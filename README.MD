# CLUTR: Curriculum Learning via Unsupervised Task Representation Learning

[comment]: <[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)>

This codebase provides the implementation of CLUTR: Curriculum Learning via Unsupervised Task Representation Learning. The CLUTR algorithm is implemented on top of the PyTorch UED framework: [Dual Curriculum Design (DCD)](https://github.com/facebookresearch/dcd), which also includes [PAIRED](https://arxiv.org/abs/2012.02096). CLUTR Recurrent VAE (`task_embed/clutr_RVAE`) uses the PyTorch implementation of Samuel Bowman's [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349#) found [here](https://github.com/kefirski/pytorch_RVAE), slightly modified to use random embeddings, instead of the default word embeddings. 


## Setup
To install the necessary dependencies, run the following commands:
```
conda create --name clutr python=3.8 -y
conda activate clutr
pip install six
pip install -r requirements.txt
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
pip install pyglet==1.5.11
```


## Training

The `scripts` directory also contains the necessary scripts to train the VAE and the CLUTR algorithm. Descriptions of the arguments can be found in `arguments.py`.


## Evaluating trained agents
`eval.py` is used to evaluate agents on specific environments.  The following command evaluates a `<model>.tar` in an experiment results directory, `<xpid>`, in a base log output directory `<log_dir>` for `<num_episodes>` episodes in each of the environments named `<env_name1>`, `<env_name1>`, and `<env_name1>`, and outputs the results as a .csv in `<result_dir>`.
```shell
python -m eval \
--base_path <log_dir> \
--xpid <xpid> \
--model_tar <model>
--env_names <env_name1>,<env_name2>,<env_name3> \
--num_episodes <num_episodes> \
--result_path <result_dir>
```




