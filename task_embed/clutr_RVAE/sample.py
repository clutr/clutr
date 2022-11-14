import argparse
import os

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == '__main__':



    #saved_model = 'data/final/vae-recons-79-iter-1000000-latent-64-sequential-batch-train_1000000_32_trained_RVAE'


    parser = argparse.ArgumentParser(description='Sampler')

    parser.add_argument('--env-name', metavar='ENV', default="minigrid",
                        help='input_path')
    parser.add_argument('--saved-model', metavar='MODEL', default="_minigrid_trained_RVAE",
                        help='input_path')
    parser.add_argument('--max-seq-len', type=int, default=52, metavar='MAX-SEQ-LEN',
                        help='')
    parser.add_argument('--num-objs', type=int, default=4)  # for minihack
    parser.add_argument('--grid-size', type=int, default=169)
    parser.add_argument('--word-embed-size', type=int, default=300)
    parser.add_argument('--obj-embed-size', type=int, default=100)  # for minihack

    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')

    parser.add_argument('--deterministic', type=bool, default=True, metavar='deterministic',
                        help='deterministic')

    parser.add_argument('--num-sample', type=int, default=100, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--enc-activation', default="none")  # relu, tanh, relu, none, scaled_tanh, streched_scaled_tanh
    parser.add_argument('--activation-scalar', type=float, default=4)  # used for scaled_tanh, ...


    args = parser.parse_args()

    assert os.path.exists(args.saved_model), \
        'trained model not found'
    #saved_model = "_minihack_trained_RVAE"
    #max_seq_len = 87
    #env_name = "minihack"

    batch_loader = BatchLoader(grid_size=args.grid_size, max_seq_len=args.max_seq_len, env_name=args.env_name,
                               num_objs=args.num_objs)


    parameters = Parameters(batch_loader.max_seq_len,
                            word_vocab_size = batch_loader.words_vocab_size,
                            latent_variable_size=64,
                            vae_type="vae",
                            enc_activation = args.enc_activation,
                            env_name = args.env_name,
                            word_embed_size=args.word_embed_size,
                            obj_embed_size=args.obj_embed_size,
                            activation_scalar=args.activation_scalar
                            )

    rvae = RVAE(parameters)
    device = t.device('cpu')
    rvae.load_state_dict(t.load(args.saved_model, map_location=device))

    if args.use_cuda:
        rvae = rvae.cuda()

    for iteration in range(args.num_sample):
        seed = np.random.normal(size=[1, parameters.latent_variable_size])*4
        #print("")
        for _ in range(2):
            result = rvae.sample(batch_loader, args.max_seq_len, seed, args.use_cuda, deterministic=args.deterministic)
            print(f"len({len(result.split())//2})")
            print(result)
            print()
        print("-"*80)
        print()
        #print(np.min(seed), np.max(seed), np.mean(seed), result)
        #print()