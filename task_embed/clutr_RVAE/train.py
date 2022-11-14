import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=10, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False) Pass in any string except "" to enable gpu. Pass empty string for false')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result input_path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result input_path (default: '')')
    parser.add_argument('--exp-name', default='', metavar='EXP',
                        help='Exp name')
    parser.add_argument('--recons-weight', type=float, default=79)
    parser.add_argument('--latent-variable-size', type=int, default=64)
    parser.add_argument('--word-embed-size', type=int, default=300)
    parser.add_argument('--obj-embed-size', type=int, default=100) #for minihack

    parser.add_argument('--env-name', default="minigrid") #minihack, minigrid
    parser.add_argument('--grid-size', type=int, default=169) #169, minigrid, 225 minihack
    parser.add_argument('--num-objs', type=int, default=4)  # for minihack
    parser.add_argument('--max-seq-len', type=int, default=52) #52 minigrid, #87 minihack

    parser.add_argument('--batching', default="sequential") #sequential, random
    parser.add_argument('--train-file', default="data/minigrid_test.txt")  #
    parser.add_argument('--test-file', default="data/minigrid_test.txt")  #
    parser.add_argument('--vae-type', default="vae") #vae, ae
    parser.add_argument('--enc-activation', default="none") #relu, tanh, relu, none, scaled_tanh, streched_scaled_tanh
    parser.add_argument('--activation-scalar', type=float, default=4) #used for scaled_tanh, ...
    parser.add_argument('--logdir', default="rvae_logs")



    args = parser.parse_args()

    #create checkpointing/logging directories
    import os
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    print(args)
    batch_loader = BatchLoader(train_file = args.train_file, test_file=args.test_file, grid_size=args.grid_size,
                            batching=args.batching, max_seq_len=args.max_seq_len, env_name=args.env_name,
                               num_objs=args.num_objs)

    parameters = Parameters(batch_loader.max_seq_len,
                            word_vocab_size = batch_loader.words_vocab_size,
                            recons_weight=args.recons_weight,
                            latent_variable_size=args.latent_variable_size,
                            vae_type=args.vae_type,
                            enc_activation = args.enc_activation,
                            env_name = args.env_name,
                            word_embed_size = args.word_embed_size,
                            obj_embed_size = args.obj_embed_size,
                            activation_scalar = args.activation_scalar
                            )

    rvae = RVAE(parameters)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)
    reonstruct = rvae.reconstructor(batch_loader)

    ce_result = []
    kld_result = []

    checkpoint_iterations = [int(args.num_iterations*p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    print("Training Started")
    for iteration in range(args.num_iterations):

        try:
            cross_entropy, kld, coef, z_info = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)
        except:
            print(f"Error in training, iteration {iteration}. Skipping Step")
            print("-"*100)
            continue

        if iteration==0 or iteration % 100 == 0 or iteration==args.num_iterations-1:

            try:
                print('\n')
                print('------------TRAIN-------------')
                print('----------ITERATION-----------')
                print(iteration)
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy.data.cpu().numpy())
                if args.vae_type=="vae": 
                    print('-------------KLD--------------')
                    print(kld.data.cpu().numpy())
                print('-----------KLD-coef-----------')
                print(coef)
                print('------------------------------')

            except:
                print("error printing log")
            #print('-----------max, min, mean z-----------')
            #print(z_info)
            #print('------------------------------')

        
        if iteration % 1000 == 0 or iteration==args.num_iterations-1:

            try:
                cross_entropy, kld = validate(args.batch_size, args.use_cuda)

                cross_entropy = cross_entropy.data.cpu().numpy()
                if args.vae_type=="vae": 
                    kld = kld.data.cpu().numpy()

                print('\n')
                print('------------VALID-------------')
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy)
                if args.vae_type=="vae": 
                    print('-------------KLD--------------')
                    print(kld)
                print('------------------------------')

                ce_result += [cross_entropy]
                if args.vae_type=="vae": 
                    kld_result += [kld]
            except:
                print(f"error validating, iteration{iteration}")

        if iteration % 10000 == 0 or iteration==args.num_iterations-1:

            try:
                print('\n')
                print('------------SAMPLE------------')
                print('------------------------------')
                for _ in range(10):
                    seed = np.random.normal(size=[1, parameters.latent_variable_size])
                    sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)
                    print(sample)

                print('------------------------------')
            except:
                print("error sampling")
        
        
        if iteration % 10000 == 0 or iteration==args.num_iterations-1:
            print("------------RECONS-------------")
            print('------------------------------')
            recons, inps = reonstruct(args.batch_size, args.use_cuda, seq_len=52, num_samples=10)
            for i, o in zip(inps, recons):
                i = i.detach()
                if args.use_cuda:
                    i = i.cpu()

                inp_str = str(i.numpy())
                if args.env_name=="minigrid":
                    inp_str = " ".join([batch_loader.idx_to_word[x] for x in i.numpy()])

                elif args.env_name=="minihack":
                    inp_str = " ".join([" ".join([batch_loader.idx_to_word[0][idx0], batch_loader.idx_to_word[1][idx1]]) for idx0, idx1 in i.numpy()])

                print("Input: ", inp_str)
                print("Output: ", o)
                print()
            print('------------------------------')
            print('------------------------------')
            print()
        
        if iteration in checkpoint_iterations:
            t.save(rvae.state_dict(), f'{args.logdir}/{args.exp_name}_{args.env_name}_trained_RVAE_{iteration}')
            np.save(f'{args.logdir}/ce_result_{args.exp_name}_{iteration}.npy', np.array(ce_result))
            np.save(f'{args.logdir}/kld_result_npy_{args.exp_name}_{iteration}.npy', np.array(kld_result))


    t.save(rvae.state_dict(), f'{args.logdir}/{args.exp_name}_{args.env_name}_trained_RVAE')


"""
minigrid


self.word_tensor[0]/ [1]: list of sequences, sequence represented as word_index

batch:
^ takes data from self.word_tensor
=> padded to the max_input_seq_len of the batch: with go, end and pad token.
encoder_word_input: (batch, input_seq_len)       #pad, reverse(input)  
decoder_word_input: (batch, input_seq_len+1)   #go, input, pad          
decoder_output: (batch, input_seq_len+1)          #input, end, pad     (also named target)


forward:

encoder_input = embedding(encoder_word_input)      : (batch, input_seq_len       , embedding_dim)     #
decoder_input = embedding(decoder_word_input)      : (batch, input_seq_len + 1 , embedding_dim)

1) encoder_input -> context -> mu, std  -> z
2) decoder_input -> decoder(decoder_input, z, drop_prob, initial_state) -> out, final_state

returns out, final_state, kld, z

out: (batch, input_seq_len + 1, dict_size==#num of dict symbols)    #linear / logits


sample:
--------------------
minihack (Plan)

batch:

encoder_word_input: (batch, input_seq_len, 2)      
decoder_word_input: (batch, input_seq_len+1, 2)       
decoder_output: (batch, input_seq_len+1, 2)        

forward:
encoder_input = embedding(encoder_word_input)      : (batch, input_seq_len       , embedding_dim)     #  SAME  SHAPE , different way of encoding
decoder_input = embedding(decoder_word_input)      : (batch, input_seq_len + 1 , embedding_dim)     #  SAME


out: (batch, input_seq_len + 1, 2, dict_size)  ???




















"""