# Go to task_embed/CLUTR_RVAE directory and run the following
python train.py --exp-name "carracing-word_embed_300_iter_1000000_latent_64_data_1000000_scaled_tanh_scalar_4" \
  --env-name "minigrid" \
  --recons-weight 79 \
  --num-iterations 1000000 \
  --latent-variable-size 64 \
  --vae-type "vae" \
  --batching "sequential" \
  --train-file "path_to_train_dataset" \
  --test-file "path_to_test_dataset"\
  --use-cuda True \
  --word-embed-size 300 \
  --grid-size 102 \
  --max-seq-len 12 \
  --enc-activation "scaled_tanh" \
  --activation-scalar 4 \
  --logdir "log"