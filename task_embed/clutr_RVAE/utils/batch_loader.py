import collections
import os
import re

import numpy as np
from six.moves import cPickle

from .functional import *


class BatchLoader:
    MINIGRID = "minigrid"
    MINIHACK = "minihack"
    MINIHACK_NUM_OBJECTS = 4

    blind_symbol = ''
    pad_token = '_'
    go_token = '>'
    end_token = '|'

    def __init__(self,  train_file=None, test_file=None, grid_size=None, batching="random", max_seq_len=None, env_name=None, num_objs=None):

        self.MINIHACK_NUM_OBJECTS = num_objs
        self.env_name = env_name
        self.batching = batching

        self.max_seq_len = max_seq_len
        self.grid_size = grid_size

        if self.env_name == self.MINIGRID:
            self.words_vocab_size, self.idx_to_word, self.word_to_idx = self.build_word_vocab_minigrid(
                grid_size=self.grid_size)
            self.max_word_len = np.amax([len(word) for word in self.idx_to_word])
            self.preprocess = self.preprocess_minigrid
        elif self.env_name == self.MINIHACK:
            self.words_vocab_size, self.idx_to_word, self.word_to_idx = self.build_word_vocab_minihack(
                grid_size=self.grid_size, num_objects=self.MINIHACK_NUM_OBJECTS)
            self.max_word_len = np.amax([len(word) for word in self.idx_to_word])
            self.preprocess = self.preprocess_minihack
        else:
            assert False, ""

        if train_file is not None:
            self.data_files = [train_file, test_file]
            self.preprocess(self.data_files)
            print('data have preprocessed')
        else:
            self.data_files = None

        self.cursor = [0, 0]

    def build_word_vocab_minigrid(self, grid_size):
        word_counts = grid_size

        # Mapping from index to word
        idx_to_word = [f"{num}" for num in range(1, word_counts + 1)]
        idx_to_word = idx_to_word + [self.pad_token, self.go_token, self.end_token]

        words_vocab_size = len(idx_to_word)

        # Mapping from word to index
        word_to_idx = {x: i for i, x in enumerate(idx_to_word)} #pad_token: 100, go_token: 101, end: 102

        return words_vocab_size, idx_to_word, word_to_idx


    def build_word_vocab_minihack(self, grid_size, num_objects=None):

        word_counts = grid_size

        #two vocabularies


        idx_to_word0 = [f"obj{i}" for i in range(1, self.MINIHACK_NUM_OBJECTS+1)]
        idx_to_word0 += [self.pad_token, self.go_token, self.end_token]
        words_vocab_size0 = len(idx_to_word0)
        word_to_idx0 = {x: i for i, x in enumerate(idx_to_word0)}


        idx_to_word1 = [f"{num}" for num in range(1, word_counts + 1)]
        idx_to_word1 += [self.pad_token, self.go_token, self.end_token]
        words_vocab_size1 = len(idx_to_word1)
        word_to_idx1 = {x: i for i, x in enumerate(idx_to_word1)}

        #0 for objcts 1 for location
        return (words_vocab_size0, words_vocab_size1), (idx_to_word0, idx_to_word1), (word_to_idx0, word_to_idx1)

    def preprocess_minigrid(self, data_files):
        data = [open(file, "r").read() for file in data_files]#train and test files
        data_words = [[line.split() for line in target.split('\n')] for target in data]
        self.num_lines = [len(target) for target in data_words] #[42068, 3370]

        self.word_tensor = np.array(
            [[list(map(self.word_to_idx.get, line)) for line in target] for target in data_words]) #each line of train and test files => each word to its index
        print(self.word_tensor.shape)



    def preprocess_minihack(self, data_files):
        data = [open(file, "r").read() for file in data_files]#train and test files
        data_words = [[line.split() for line in target.split('\n') if line] for target in data]
        self.num_lines = [len(target) for target in data_words]

        self.word_tensor = []
        for target in data_words:
            tar_ = []
            for line in target:
                this = []
                i = 0
                while i < len(line):
                    obj, loc = line[i], line[i + 1]
                    obj_idx, loc_idx = self.word_to_idx[0][obj], self.word_to_idx[1][loc]
                    this.append((obj_idx, loc_idx))
                    # print(obj, loc, obj_idx, loc_idx)
                    i += 2

                tar_.append(this)

            self.word_tensor.append(tar_)

        self.word_tensor = np.array(self.word_tensor)
        print(self.word_tensor.shape)

    def get_input_vec_from_str_list(self, inp):
        if self.env_name == self.MINIGRID:
            #inp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            #inp_str = [str(x) for x in inp]
            inp_vector = list(map(self.word_to_idx.get, inp))
            return inp_vector[::-1]

        elif self.env_name == self.MINIHACK:
            word_tensor = []
            i = 0
            while i < len(inp):
                obj, loc = inp[i], inp[i + 1]
                obj_idx, loc_idx = self.word_to_idx[0][obj], self.word_to_idx[1][loc]
                word_tensor.append((obj_idx, loc_idx))
                i += 2

            return word_tensor[::-1]

        else:
            assert False, "Invalid env_name in batch_loader object"

    def next_batch(self, batch_size, target_str):
        target = 0 if target_str == 'train' else 1

        if self.batching == "random":
            indexes = np.array(np.random.randint(self.num_lines[target], size=batch_size))

        elif self.batching == "sequential":
            start = self.cursor[target]
            end = start + batch_size
            if end > self.num_lines[target]:
                end = self.num_lines[target]

            indexes = np.array(list(range(start, end)))

            self.cursor[target] = end%self.num_lines[target]
        else:
            assert False, "Incorrect batching option"


        encoder_word_input = [self.word_tensor[target][index] for index in indexes]
        #encoder_character_input = [self.character_tensor[target][index] for index in indexes]
        input_seq_len = [len(line) for line in encoder_word_input]
        max_input_seq_len = np.amax(input_seq_len)

        encoded_words = [[idx for idx in line] for line in encoder_word_input]

        if self.env_name==self.MINIGRID:
            decoder_word_input = [[self.word_to_idx[self.go_token]] + line for line in encoder_word_input]
            decoder_output = [line + [self.word_to_idx[self.end_token]] for line in encoded_words]
            pad_token = self.word_to_idx[self.pad_token]
        elif self.env_name==self.MINIHACK:
            decoder_word_input = [[(self.word_to_idx[0][self.go_token], self.word_to_idx[1][self.go_token])] + line for line in encoder_word_input]
            decoder_output = [line + [(self.word_to_idx[0][self.end_token], self.word_to_idx[1][self.end_token])] for line in encoded_words]
            pad_token = (self.word_to_idx[0][self.pad_token], self.word_to_idx[1][self.pad_token])

        # sorry
        for i, line in enumerate(decoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_word_input[i] = line + [pad_token] * to_add

        #for i, line in enumerate(decoder_character_input):
        #    line_len = input_seq_len[i]
        #    to_add = max_input_seq_len - line_len
        #    decoder_character_input[i] = line + [self.encode_characters(self.pad_token)] * to_add

        for i, line in enumerate(decoder_output):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_output[i] = line + [pad_token] * to_add

        for i, line in enumerate(encoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            encoder_word_input[i] = [pad_token] * to_add + line[::-1]

        #for i, line in enumerate(encoder_character_input):
        #    line_len = input_seq_len[i]
        #    to_add = max_input_seq_len - line_len
        #    encoder_character_input[i] = [self.encode_characters(self.pad_token)] * to_add + line[::-1]

        return np.array(encoder_word_input), np.array(decoder_word_input), np.array(decoder_output)
        #return np.array(encoder_word_input), np.array(encoder_character_input), \
        #       np.array(decoder_word_input), np.array(decoder_character_input), np.array(decoder_output)


    def go_input(self, batch_size):
        if self.env_name == self.MINIGRID:
            go_word_input = [[self.word_to_idx[self.go_token]] for _ in range(batch_size)]
            return np.array(go_word_input)
        elif self.env_name == self.MINIHACK:
            go_input = [[self.word_to_idx[0][self.go_token], self.word_to_idx[1][self.go_token]] for _ in range(batch_size)]
            return np.array(go_input).reshape(-1,1,2)
            #go_word_input = [[]] for _ in range(batch_size)]

    """
    def encode_word(self, idx):
        if self.env_name == self.MINIGRID:
            result = np.zeros(self.words_vocab_size)
            result[idx] = 1
            return result
        elif self.env_name == self.MINIHACK:
            pass

    def decode_word(self, word_idx):
        if self.env_name == self.MINIGRID:
            word = self.idx_to_word[word_idx]
            return word
        elif self.env_name == self.MINIHACK:
            pass
    """

    def sample_word_from_distribution(self, distribution):

        if self.env_name == self.MINIGRID:
            ix = np.random.choice(range(self.words_vocab_size), p=distribution.ravel())
            x = np.zeros((self.words_vocab_size, 1))
            x[ix] = 1
            return self.idx_to_word[np.argmax(x)]
        elif self.env_name == self.MINIHACK:
            ix0 = np.random.choice(range(self.words_vocab_size[0]), p=distribution[0].ravel())
            x = np.zeros((self.words_vocab_size[0], 1))
            x[ix0] = 1
            obj =  self.idx_to_word[0][np.argmax(x)]

            ix1 = np.random.choice(range(self.words_vocab_size[1]), p=distribution[1].ravel())
            x = np.zeros((self.words_vocab_size[1], 1))
            x[ix1] = 1
            loc = self.idx_to_word[1][np.argmax(x)]

            return (obj, loc)




    """
    def encode_characters(self, characters):
        word_len = len(characters)
        to_add = self.max_word_len - word_len
        characters_idx = [self.char_to_idx[i] for i in characters] + to_add * [self.char_to_idx['']]
        return characters_idx
    """
