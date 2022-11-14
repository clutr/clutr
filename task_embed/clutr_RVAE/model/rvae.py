import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .encoder import Encoder

from selfModules.embedding import Embedding, ClutrEmbedding

from utils.functional import kld_coef, parameters_allocation_check, fold




class RVAE(nn.Module):
    MINIGRID = "minigrid"
    MINIHACK = "minihack"
    def __init__(self, params):
        super(RVAE, self).__init__()

        self.params = params

        #self.recons_weight = params.recons_weight

        self.embedding = ClutrEmbedding(self.params)

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)
        if self.params.enc_activation is None or self.params.enc_activation == "none":
            self.enc_activation = lambda x: x
        elif self.params.enc_activation == "tanh":
            self.enc_activation = t.tanh
        elif self.params.enc_activation == "scaled_tanh":
            self.enc_activation = lambda x: t.tanh(x)*self.params.activation_scalar
        elif self.params.enc_activation == "streched_scaled_tanh":
            self.enc_activation = lambda x: t.tanh(x/self.params.activation_scalar) * self.params.activation_scalar
        elif self.params.enc_activation == "relu":
            self.enc_activation = lambda x: t.clamp(x, min=0, max=1)
        elif self.params.enc_activation == "sigmoid":
            self.enc_activation = t.sigmoid

    def forward(self, drop_prob,
                encoder_word_input=None,
                decoder_word_input=None,
                z=None, initial_state=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        #assert parameters_allocation_check(self), \
        #    'Invalid CUDA options. Parameters should be allocated in the same memory'
        if hasattr(self.embedding, "word_embed"):
            use_cuda = self.embedding.word_embed.weight.is_cuda
        else:
            use_cuda = self.embedding.word_embed1.weight.is_cuda
        """
        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [encoder_word_input],
                                  True) \
            or (z is not None and decoder_word_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"
        """

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            #[batch_size, _] = encoder_word_input.size()
            batch_size = encoder_word_input.size()[0]
            if self.params.env_name==self.MINIGRID:
                encoder_input = self.embedding.word_embed(encoder_word_input)

            elif self.params.env_name==self.MINIHACK:
                objs = encoder_word_input[:, :, 0]
                emb_objs = self.embedding.word_embed0(objs)

                locs = encoder_word_input[:, :, 1]
                emb_locs = self.embedding.word_embed1(locs)

                encoder_input = t.cat((emb_objs, emb_locs), dim=2)


            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            if self.params.vae_type == "vae":
                z = z * std + mu
                kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

            elif self.params.vae_type == "ae":
                z = mu
                #kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
                kld = 0

            else:
                assert False, "wrong argument for vae_type"

            z = self.enc_activation(z)

        else:
            kld = None

        out, final_state = None, None
        if decoder_word_input is not None:
            if self.params.env_name == self.MINIGRID:
                decoder_input = self.embedding.word_embed(decoder_word_input)

            elif self.params.env_name == self.MINIHACK:
                objs = decoder_word_input[:, :, 0]
                emb_objs = self.embedding.word_embed0(objs)

                locs = decoder_word_input[:, :, 1]
                emb_locs = self.embedding.word_embed1(locs)

                decoder_input = t.cat((emb_objs, emb_locs), dim=2)

            #if self.params.env_name == self.MINIHACK:
            #    decoder_input = decoder_input.reshape((decoder_input.shape[0], decoder_input.shape[1], -1))
            out, final_state = self.decoder(decoder_input, z, drop_prob, initial_state)

        return out, final_state, kld, z

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            #Hack: Most likely error for the last line of the data file or data is not a mod of 32 - I dont know what :(
            if len(input) == 3 and input[0].shape[0]>0 and input[1].shape[0]>0  \
                and  input[2].shape[0]>0 and input[0].shape[1]>0 \
                and input[1].shape[1]>0 and  input[2].shape[1]>0 :
                pass
            else:
                input = batch_loader.next_batch(batch_size, 'train')

            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, decoder_word_input, target] = input

            logits, _, kld, z = self(dropout, encoder_word_input, decoder_word_input, z=None)



            if self.params.env_name == self.MINIGRID:
                logits = logits.view(-1, self.params.word_vocab_size)
                target = target.view(-1)
                cross_entropy = F.cross_entropy(logits, target)
                recons_loss = cross_entropy



            elif  self.params.env_name == self.MINIHACK:
                logits_obj = logits[:, :, 0:self.params.obj_vocab_size].view(-1, self.params.obj_vocab_size)
                target_obj = target[:, :, 0].view(-1)
                ce_obj = F.cross_entropy(logits_obj, target_obj)

                logits_loc = logits[:, :, self.params.obj_vocab_size:].view(-1, self.params.word_vocab_size)
                target_loc = target[:, :, 1].view(-1)
                ce_loc =  F.cross_entropy(logits_loc, target_loc)

                cross_entropy = ce_obj + ce_loc
                recons_loss = cross_entropy

            loss = self.params.recons_weight * recons_loss + kld_coef(i) * kld


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kld_coef(i), (t.max(z).data, t.min(z).data, t.mean(z).data)

        return train

    def validater(self, batch_loader):
        def validate(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, decoder_word_input, target] = input

            logits, _, kld, _ = self(0.,
                                  encoder_word_input, decoder_word_input,
                                  z=None)

            if self.params.env_name == self.MINIGRID:
                logits = logits.view(-1, self.params.word_vocab_size)
                target = target.view(-1)
                cross_entropy = F.cross_entropy(logits, target)
            elif self.params.env_name == self.MINIHACK:
                logits_obj = logits[:, :, 0:self.params.obj_vocab_size].view(-1, self.params.obj_vocab_size)
                target_obj = target[:, :, 0].view(-1)
                ce_obj = F.cross_entropy(logits_obj, target_obj)

                logits_loc = logits[:, :, self.params.obj_vocab_size:].view(-1, self.params.word_vocab_size)
                target_loc = target[:, :, 1].view(-1)
                ce_loc = F.cross_entropy(logits_loc, target_loc)

                cross_entropy = ce_obj + ce_loc


            return cross_entropy, kld

        return validate

    def reconstructor(self, batch_loader):
        def reconstruct(batch_size, use_cuda, seq_len, num_samples=10):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, _, _] = input
            #out, final_state, kld, z
            _, _, _, z = self(0., encoder_word_input, None,
                                  z=None)

            recons = []
            inps = []
            for i in range(min(num_samples, input[0].shape[0])):
                seed = z.detach()
                if use_cuda:
                    seed = seed.cpu()
                seed = seed.numpy()[i].reshape(1, -1)
                recon = self.sample(batch_loader, seq_len=seq_len, seed=seed, use_cuda=use_cuda)
                recons.append(recon)
                inps.append(input[0][i])

            return recons, inps

        return reconstruct

    def get_embedding(self, batch_loader):
        def embedder(batch_size, use_cuda, seq_len, num_samples=10):

            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, _, _] = input
            #out, final_state, kld, z
            _, _, _, z = self(0., encoder_word_input, None,
                                  z=None)

            recons = []
            inps = []

            for i in range(min(num_samples, input[0].shape[0])):
                seed = z.detach()
                if use_cuda:
                    seed = seed.cpu()
                seed = seed.numpy()[i].reshape(1, -1)
                recon = self.sample(batch_loader, seq_len=seq_len, seed=seed, use_cuda=use_cuda)
                recons.append(recon)
                inps.append(input[0][i])

            return recons, inps, z

        return embedder


    def sample(self, batch_loader, seq_len, seed, use_cuda, deterministic=False):
        seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        if self.params.env_name == self.MINIGRID:
            decoder_word_input_np = batch_loader.go_input(1)
        elif self.params.env_name == self.MINIHACK:
            decoder_word_input_np = batch_loader.go_input(1).reshape(1,1,-1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        #decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            #decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()
            decoder_word_input = decoder_word_input.cuda()

        result = ''

        initial_state = None

        for i in range(seq_len):
            logits, initial_state, _, _ = self(0., None,
                                            decoder_word_input,
                                            seed, initial_state)




            if self.params.env_name==self.MINIGRID:
                logits = logits.view(-1, self.params.word_vocab_size)
                prediction = F.softmax(logits)  # (1,228)

                if deterministic:
                    prediction_z = t.zeros_like(prediction)
                    prediction_z[0, t.argmax(prediction)] = 1.0
                    prediction = prediction_z
                    #print()

                word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])
                if word == batch_loader.end_token:
                    break
                result += ' ' + word
                decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            elif self.params.env_name==self.MINIHACK:
                logits = logits.view(-1, self.params.word_vocab_size+ self.params.obj_vocab_size)
                prediction_obj = F.softmax(logits[:, :self.params.obj_vocab_size]) # (1,228)
                prediction_loc = F.softmax(logits[:, self.params.obj_vocab_size:])

                if deterministic:
                    prediction_obj_z = t.zeros_like(prediction_obj)
                    prediction_obj_z[0, t.argmax(prediction_obj)] = 1.0
                    prediction_obj  = prediction_obj_z

                    prediction_loc_z = t.zeros_like(prediction_loc)
                    prediction_loc_z[0, t.argmax(prediction_loc)] = 1.0
                    prediction_loc = prediction_loc_z

                word0, word1 = batch_loader.sample_word_from_distribution([prediction_obj.data.cpu().numpy()[-1], prediction_loc.data.cpu().numpy()[-1]])
                #word1 = ae_batch_loader.sample_word_from_distribution(prediction_loc.data.cpu().numpy()[-1])

                if word1 == batch_loader.end_token:
                    break

                result += ' ' + (word0 + " " + word1)
                decoder_word_input_np = np.array([[batch_loader.word_to_idx[0][word0], batch_loader.word_to_idx[1][word1]]]).reshape(1,1,-1)

            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())

            if use_cuda:
                #decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()
                decoder_word_input = decoder_word_input.cuda()

        return result
