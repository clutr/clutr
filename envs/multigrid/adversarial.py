# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""An environment which is built by a learning adversary.

Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""
import random

import gym
import gym_minigrid.minigrid as minigrid
import networkx as nx
from networkx import grid_graph
import numpy as np
import torch
import random
from . import multigrid
from . import register
import os, sys
EDITOR_ACTION_SPACES = {
    'walls_none': {
        0: '-',
        1: '.',
    },
    'walls_none_goal': {
        0: '-',
        1: '.',
        2: 'g',
    },
    'walls_none_agent_goal': {
        0: '-',
        1: '.',
        2: 'a',
        3: 'g',
    },
}

# For TaskEmbed Exp
from task_embed import clutr_RVAE
from task_embed.clutr_RVAE.utils.batch_loader import BatchLoader
from task_embed.clutr_RVAE.utils.parameters import Parameters
from task_embed.clutr_RVAE.model.rvae import RVAE
import torch


class AdversarialEnv(multigrid.MultiGridEnv):
    """Grid world where an adversary build the environment the agent plays.

    The adversary places the goal, agent, and up to n_clutter blocks in sequence.
    The action dimension is the number of squares in the grid, and each action
    chooses where the next item should be placed.
    """

    def __init__(self,
                 n_clutter=50,
                 resample_n_clutter=False,
                 size=15,
                 agent_view_size=5,
                 max_steps=250,
                 goal_noise=0.,
                 random_z_dim=50,
                 choose_goal_last=False,
                 see_through_walls=True,
                 seed=0,
                 editor_actions='walls_none_agent_goal',
                 fixed_environment=False,

                 use_latent_task=False,
                 latent_task_exp=None,
                 latent_space_discrete_max=None,
                 task_ae=None,
                 ae_type=None,
                 latent_dim_max=None,
                 latent_dim_min=None,
                 cuda=None,
                 enc_activation=None,
                 latent_dim=None,
                 activation_scalar=0,
                 word_embed_size=0,
                 deterministic_vae=True
                 ):
        """Initializes environment in which adversary places goal, agent, obstacles.

        Args:
          n_clutter: The maximum number of obstacles the adversary can place.
          size: The number of tiles across one side of the grid; i.e. make a
            size x size grid.
          agent_view_size: The number of tiles in one side of the agent's partially
            observed view of the grid.
          max_steps: The maximum number of steps that can be taken before the
            episode terminates.
          goal_noise: The probability with which the goal will move to a different
            location than the one chosen by the adversary.
          random_z_dim: The environment generates a random vector z to condition the
            adversary. This gives the dimension of that vector.
          choose_goal_last: If True, will place the goal and agent as the last
            actions, rather than the first actions.
        """
        # === Task Embed Fields ===
        self.use_latent_task = use_latent_task
        self.latent_dim = latent_dim
        self.latent_task_exp = latent_task_exp
        self.latent_space_discrete_max = latent_space_discrete_max
        self.ae_type = ae_type
        self.enc_activation = enc_activation
        self.activation_scalar = activation_scalar
        self.cuda = cuda
        self.latent_dim_min = latent_dim_min
        self.latent_dim_max = latent_dim_max
        self.word_embed_size = word_embed_size
        self.deterministic_vae = deterministic_vae

        self.agent_start_pos = None
        self.goal_pos = None
        self.n_clutter = n_clutter
        self.resample_n_clutter = resample_n_clutter
        self.goal_noise = goal_noise
        self.random_z_dim = random_z_dim
        self.choose_goal_last = choose_goal_last

        # Add two actions for placing the agent and goal.
        self.n_clutter_sampled = False

        self.actual_grid_dim = size - 2
        self.actual_grid_size = (size - 2) ** 2

        super().__init__(
            n_agents=1,
            minigrid_mode=True,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,  # Set this to True for maximum speed
            competitive=True,
            seed=seed,
            fixed_environment=fixed_environment,
        )

        # Metrics
        self.reset_metrics()

        self.editor_actions = list(EDITOR_ACTION_SPACES[editor_actions].values())

        self.grid_dim = (size - 2) ** 2 # same value as self.adversary_action_dim,
        self.adversary_action_dim = (size - 2) ** 2

        # === Adversary observation and action space ===
        if self.use_latent_task:
            self.last_latent_vector = None
            if self.latent_task_exp == "singlestep_continuous":
                self.adversary_randomz_obs_space = gym.spaces.Box(
                    low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
                self.adversary_observation_space = gym.spaces.Dict(
                    {'random_z': self.adversary_randomz_obs_space})

                action_shape = (self.latent_dim,)

                self.adversary_action_space = gym.spaces.Box(
                    low=self.latent_dim_min, high=self.latent_dim_max, shape=action_shape, dtype='float32')


                # Load VAE
                if self.ae_type == "lstm_vae":
                    self.ae_seq_len = n_clutter + 2
                    self.cuda = cuda
                    # the actual range of discrete actions is [0, self.skecth_dim**2 +1] => [0, 101] for sketch_dim=10
                    # the +1 is for the skip action
                    # the 0 is for what??
                    # so we have self.skecth_dim**2 +2 discrete symbols => [0, 101]
                    # the vae generates [1, 102]
                    self.ae_batch_loader = BatchLoader(grid_size=self.actual_grid_size,
                                                       max_seq_len=self.ae_seq_len,
                                                       env_name="minigrid")

                    parameters = Parameters(self.ae_batch_loader.max_seq_len,
                                            word_vocab_size=self.ae_batch_loader.words_vocab_size,
                                            latent_variable_size=self.latent_dim,
                                            vae_type="vae",
                                            enc_activation=self.enc_activation,
                                            env_name="minigrid",
                                            word_embed_size=self.word_embed_size,
                                            activation_scalar=self.activation_scalar
                                            )
                    self.ae = RVAE(parameters)
                    import torch
                    self.ae.load_state_dict(torch.load(task_ae, map_location="cpu"))
                else:
                    raise NotImplementedError(f"{self.ae_type} not implemented")
            
            elif self.latent_task_exp == "multistep_discrete":
                self.adversary_max_steps = self.latent_dim
                self.adversary_action_space = gym.spaces.Discrete(self.latent_space_discrete_max + 1)

                self.adversary_ts_obs_space = gym.spaces.Box(
                    low=0, high=self.latent_dim, shape=(1,),
                    dtype='uint32')

                self.adversary_randomz_obs_space = gym.spaces.Box(
                    low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32) 

                self.adversary_observation_space = gym.spaces.Dict(
                    {'time_step': self.adversary_ts_obs_space,
                     'random_z': self.adversary_randomz_obs_space})

                #####LOAD VAE (Copied from singlestep continous)
                if self.ae_type == "lstm_vae":
                    self.ae_seq_len = n_clutter + 2
                    self.cuda = cuda
                    # the actual range of discrete actions is [0, self.skecth_dim**2 +1] => [0, 101] for sketch_dim=10
                    # the +1 is for the skip action
                    # the 0 is for what??
                    # so we have self.skecth_dim**2 +2 discrete symbols => [0, 101]
                    # the vae generates [1, 102]
                    self.ae_batch_loader = BatchLoader(grid_size=self.actual_grid_size,
                                                       max_seq_len=self.ae_seq_len,
                                                       env_name="minigrid")

                    parameters = Parameters(self.ae_batch_loader.max_seq_len,
                                            word_vocab_size=self.ae_batch_loader.words_vocab_size,
                                            latent_variable_size=self.latent_dim,
                                            vae_type="vae",
                                            enc_activation=self.enc_activation,
                                            env_name="minigrid",
                                            word_embed_size=self.word_embed_size,
                                            activation_scalar=self.activation_scalar
                                            )
                    self.ae = RVAE(parameters)
                    import torch
                    self.ae.load_state_dict(torch.load(task_ae, map_location="cpu"))
                else:
                    raise NotImplementedError(f"{self.ae_type} not implemented")

            else:
                raise NotImplementedError(f"{self.latent_task_exp} not implemented for latent task carracing")

        else:

            self.adversary_max_steps = self.n_clutter + 2
            self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
            self.adversary_ts_obs_space = gym.spaces.Box(
                low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
            self.adversary_randomz_obs_space = gym.spaces.Box(
                low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
            self.adversary_image_obs_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype='uint8')

            # Adversary observations are dictionaries containing an encoding of the
            # grid, the current time step, and a randomly generated vector used to
            # condition generation (as in a GAN).
            self.adversary_observation_space = gym.spaces.Dict(
                {'image': self.adversary_image_obs_space,
                 'time_step': self.adversary_ts_obs_space,
                 'random_z': self.adversary_randomz_obs_space})

        # NetworkX graph used for computing shortest input_path
        self.graph = grid_graph(dim=[size - 2, size - 2])
        self.wall_locs = []

    def _resample_n_clutter(self):
        n_clutter = np.random.randint(0, self.n_clutter)
        self.adversary_max_steps = n_clutter + 2
        self.n_clutter_sampled = True

        return n_clutter

    @property
    def processed_action_dim(self):
        return 1

    @property
    def encoding(self):
        return self.grid.encode()

    def _gen_grid(self, width, height):
        """Grid is initially empty, because adversary will create it."""
        # Create an empty grid
        self.grid = multigrid.Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

    def get_goal_x(self):
        if self.goal_pos is None:
            return -1
        return self.goal_pos[0]

    def get_goal_y(self):
        if self.goal_pos is None:
            return -1
        return self.goal_pos[1]

    def reset_metrics(self):
        self.distance_to_goal = -1
        self.n_clutter_placed = 0
        self.passable = -1
        self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

    def compute_metrics(self):
        self.n_clutter_placed = self._count_walls()
        self.compute_shortest_path()

    def reset(self):
        """Fully resets the environment to an empty grid with no agent or goal."""
        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])
        self.wall_locs = []

        self.step_count = 0
        self.adversary_step_count = 0

        if self.resample_n_clutter:
            self.n_clutter_sampled = False

        self.agent_start_dir = self._rand_int(0, 4)

        # Current position and direction of the agent
        self.reset_agent_status()

        self.agent_start_pos = None
        self.goal_pos = None

        self.done = False

        # Extra metrics
        self.reset_metrics()

        # Generate the grid. Will be random by default, or same environment if
        # 'fixed_environment' is True.
        self._gen_grid(self.width, self.height)

        if self.use_latent_task:
            if self.latent_task_exp == "singlestep_continuous":
                obs = {
                    'random_z': self.generate_random_z()
                }
                return obs
            elif self.latent_task_exp == "multistep_discrete" :

                self.env_actions = []
                obs = {
                    'time_step': [self.adversary_step_count],
                    'random_z': self.generate_random_z()
                }
                return obs
            else:
                raise NotImplementedError(f"{self.latent_task_exp} not implemented for latent task")

        else:

            image = self.grid.encode()
            obs = {
                'image': image,
                'time_step': [self.adversary_step_count],
                'random_z': self.generate_random_z()
            }

            return obs

    def reset_agent_status(self):
        """Reset the agent's position, direction, done, and carrying status."""
        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [self.agent_start_dir] * self.n_agents
        self.done = [False] * self.n_agents
        self.carrying = [None] * self.n_agents

    def reset_agent(self):
        """Resets the agent's start position, but leaves goal and walls."""
        # Remove the previous agents from the world
        for a in range(self.n_agents):
            if self.agent_pos[a] is not None:
                self.grid.set(self.agent_pos[a][0], self.agent_pos[a][1], None)

        # Current position and direction of the agent
        self.reset_agent_status()

        if self.agent_start_pos is None:
            raise ValueError('Trying to place agent at empty start position.')
        else:
            self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)

        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None
            assert self.agent_dir[a] is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos[a])
            if not (start_cell.type == 'agent' or
                    start_cell is None or start_cell.can_overlap()):
                raise ValueError('Wrong object in agent start position.')

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            actions = [int(a) for a in level.split()]

            if self.resample_n_clutter:
                self.adversary_max_steps = len(actions)

            for a in actions:
                obs, _, done, _ = self.step_adversary(a)
                if done:
                    obs = self.reset_agent()
        else:
            # reset based on encoding
            obs = self.reset_to_encoding(level)

        return obs

    def reset_to_encoding(self, encoding):
        self.grid.set_encoding(encoding, multigrid_env=self)
        self.compute_metrics()

        return self.reset_agent()

    def _clean_loc(self, x, y):
        # Remove any walls
        self.remove_wall(x, y)
        # print(f'cleaning loc {x}, {y}', flush=True)

        if isinstance(self.grid.get(x, y), minigrid.Goal):
            self.goal_pos = None
        elif isinstance(self.grid.get(x, y), multigrid.Agent):
            self.agent_start_pos = None

        self.grid.set(x, y, None)

    def _free_xy_from_mask(self, free_mask):
        free_idx = free_mask.flatten().nonzero()[0]
        free_loc = np.random.choice(free_idx)
        mask_w, mask_h = free_mask.shape
        x = free_loc % mask_w
        y = free_loc // mask_w

        return x, y

    def mutate_level(self, num_edits=1):
        """
        Mutate the current level:
          - Select num_edits locations (with replacement).
          - Take the unique set of locations, which can be < num_edits.
          - Choose a unique entity for each location.
          - Place entities in each location.
          - Place goal and agent if they do not exist.
        """
        num_tiles = (self.width - 2) * (self.height - 2)
        edit_locs = list(set(np.random.randint(0, num_tiles, num_edits)))

        action_idx = np.random.randint(0, len(self.editor_actions), len(edit_locs))
        actions = [self.editor_actions[i] for i in action_idx]

        free_mask = ~self.wall_mask
        free_mask[self.agent_start_pos[1] - 1, self.agent_start_pos[0] - 1] = False
        free_mask[self.goal_pos[1] - 1, self.goal_pos[0] - 1] = False

        for loc, a in zip(edit_locs, actions):
            x = loc % (self.width - 2) + 1
            y = loc // (self.width - 2) + 1

            self._clean_loc(x, y)

            if a == '-':
                self.put_obj(minigrid.Wall(), x, y)
                self.wall_locs.append((x - 1, y - 1))
                self.n_clutter_placed += 1
                free_mask[y - 1, x - 1] = False
            elif a == '.':
                self.remove_wall(x, y)
                self.grid.set(x, y, None)
                free_mask[y - 1, x - 1] = True
            elif a == 'a':
                if self.agent_start_pos is not None:
                    ax, ay = self.agent_start_pos
                    self.grid.set(ax, ay, None)
                    free_mask[ay - 1, ax - 1] = True

                self.place_one_agent(0, top=(x, y), size=(1, 1))
                self.agent_start_pos = np.array((x, y))
                free_mask[y - 1, x - 1] = False
            elif a == 'g':
                if self.goal_pos is not None:
                    gx, gy = self.goal_pos
                    self.grid.set(gx, gy, None)
                    free_mask[gy - 1, gx - 1] = True

                self.put_obj(minigrid.Goal(), x, y)
                self.goal_pos = np.array((x, y))
                free_mask[y - 1, x - 1] = False

        # Make sure goal exists
        if self.goal_pos is None:
            x, y = self._free_xy_from_mask(free_mask)
            free_mask[y, x] = False
            x += 1
            y += 1

            self.put_obj(minigrid.Goal(), x, y)
            self.goal_pos = np.array((x, y))

        # Make sure agent exists
        if self.agent_start_pos is None:
            x, y = self._free_xy_from_mask(free_mask)
            free_mask[y, x] = False
            x += 1
            y += 1

            self.place_one_agent(0, top=(x, y), size=(1, 1))
            self.agent_start_pos = np.array((x, y))

        # Reset meta info
        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])
        self.step_count = 0
        self.adversary_step_count = 0
        self.reset_metrics()
        self.compute_metrics()

        return self.reset_agent()

    def remove_wall(self, x, y):
        if (x - 1, y - 1) in self.wall_locs:
            self.wall_locs.remove((x - 1, y - 1))
            self.n_clutter_placed -= 1
        obj = self.grid.get(x, y)
        if obj is not None and obj.type == 'wall':
            self.grid.set(x, y, None)

    def _count_walls(self):
        wall_mask = np.array(
            [1 if isinstance(x, minigrid.Wall) else 0 for x in self.grid.grid], dtype=np.bool) \
                        .reshape(self.height, self.width)[1:-1, 1:-1]
        self.wall_mask = wall_mask

        num_walls = wall_mask.sum()

        wall_pos = list(zip(*np.nonzero(wall_mask)))
        self.wall_locs = [(x + 1, y + 1) for y, x in wall_pos]

        for y, x in wall_pos:
            self.graph.remove_node((x, y))

        return num_walls

    def compute_shortest_path(self):
        if self.agent_start_pos is None or self.goal_pos is None:
            return

        self.distance_to_goal = abs(
            self.goal_pos[0] - self.agent_start_pos[0]) + abs(
            self.goal_pos[1] - self.agent_start_pos[1])

        # Check if there is a input_path between agent start position and goal. Remember
        # to subtract 1 due to outside walls existing in the Grid, but not in the
        # networkx graph.
        self.passable = nx.has_path(
            self.graph,
            source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
            target=(self.goal_pos[0] - 1, self.goal_pos[1] - 1))
        if self.passable:
            # Compute shortest input_path
            self.shortest_path_length = nx.shortest_path_length(
                self.graph,
                source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
                target=(self.goal_pos[0] - 1, self.goal_pos[1] - 1))
        else:
            # Impassable environments have a shortest input_path length 1 longer than
            # longest possible input_path
            self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)


    def single_step(self, loc, adversary_step_count, total_steps):

        if loc >= self.grid_dim:
            raise ValueError('Position passed to step_adversary is outside the grid.')

        # Add offset of 1 for outside walls
        x = int(loc % (self.width - 2)) + 1
        y = int(loc / (self.width - 2)) + 1
        #done = False

        if self.choose_goal_last:
            should_choose_goal = adversary_step_count == total_steps - 2
            should_choose_agent = adversary_step_count == total_steps - 1
        else:
            should_choose_goal = adversary_step_count == 0
            should_choose_agent = adversary_step_count == 1

        # print(f"{self.adversary_step_count}/{self.adversary_max_steps}", flush=True)
        # print(f"goal/agent = {should_choose_goal}/{should_choose_agent}", flush=True)

        # Place goal
        if should_choose_goal:
            # If there is goal noise, sometimes randomly place the goal
            if random.random() < self.goal_noise:
                self.goal_pos = self.place_obj(minigrid.Goal(), max_tries=100)
            else:
                self.remove_wall(x, y)  # Remove any walls that might be in this loc
                self.put_obj(minigrid.Goal(), x, y)
                self.goal_pos = (x, y)

        # Place the agent
        elif should_choose_agent:
            self.remove_wall(x, y)  # Remove any walls that might be in this loc

            # Goal has already been placed here
            if self.grid.get(x, y) is not None:
                # Place agent randomly
                self.agent_start_pos = self.place_one_agent(0, rand_dir=False)
                self.deliberate_agent_placement = 0
            else:
                self.agent_start_pos = np.array([x, y])
                self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
                self.deliberate_agent_placement = 1

        # Place wall
        elif adversary_step_count < total_steps:
            # If there is already an object there, action does nothing
            if self.grid.get(x, y) is None:
                self.put_obj(minigrid.Wall(), x, y)
                self.n_clutter_placed += 1
                self.wall_locs.append((x - 1, y - 1))

        return x, y

    def get_env_vector_from_latent_vector(self, latent_vector):

        if self.ae_type == "lstm_vae":
            seq = self.ae.sample(self.ae_batch_loader, self.ae_seq_len, latent_vector.reshape(1, -1), use_cuda=False,
                                 deterministic=self.deterministic_vae)

            seq = [int(n) - 1 for n in seq.split() if n.isdigit()]

            if self.latent_task_exp == "singlestep_continuous":
                return seq[2:] + seq[:2] # MINIGRID VAE PLACES AGENT AND GOAL FIRST, but Environment vector needs to be in the end
            elif self.latent_task_exp == "multistep_discrete":
                return seq 
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def step_adversary(self, loc):
        """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.

        The action space is the number of possible squares in the grid. The squares
        are numbered from left to right, top to bottom.

        Args:
          loc: An integer specifying the location to place the next object which
            must be decoded into x, y coordinates.

        Returns:
          Standard RL observation, reward (always 0), done, and info
        """

        # Resample block count if necessary, based on first loc
        if self.resample_n_clutter and not self.n_clutter_sampled:
            if self.use_latent_task:
                #Check what this condition is for
                raise NotImplementedError()
            else:
                n_clutter = int((loc / self.adversary_action_dim) * self.n_clutter)
                self.adversary_max_steps = n_clutter + 2
                self.n_clutter_sampled = True

        if self.use_latent_task:
            self.last_latent_vector = loc
            if self.latent_task_exp == "singlestep_continuous":
                latent_vector = loc
                locs = self.get_env_vector_from_latent_vector(latent_vector)

                all_pos = set()
                for i in range(len(locs)):
                    x, y = self.single_step(locs[i], i, len(locs))
                    all_pos.add((x, y))

                while len(all_pos)<2:
                    self.single_step(random.randint(0, self.grid_dim-1), i, 2)
                    i+=1
                    all_pos.add((x, y))

                obs = {
                    'random_z': self.generate_random_z()
                }
                done = True
                self.adversary_step_count += 1

            elif self.latent_task_exp == "multistep_discrete":
                self.adversary_step_count += 1
                self.env_actions.append(loc)
                done = False

                obs = {
                    'time_step': [self.adversary_step_count],
                    'random_z': self.generate_random_z()
                }

                # End of episode
                if self.adversary_step_count >= self.adversary_max_steps:
                    done = True
                    locs = self.get_env_vector_from_latent_vector(latent_vector)

                    all_pos = set()
                    for i in range(len(locs)):
                        x, y = self.single_step(locs[i], i, len(locs))
                        all_pos.add((x, y))

                    while len(all_pos)<2:
                        self.single_step(random.randint(0, self.grid_dim-1), i, 2)
                        i+=1
                        all_pos.add((x, y))

            else:
                raise NotImplementedError()

        else:

            if self.adversary_step_count < self.adversary_max_steps:
                self.single_step(loc, self.adversary_step_count, self.adversary_max_steps)

            self.adversary_step_count += 1

            # End of episode
            if self.adversary_step_count >= self.n_clutter + 2:
                done = True
            else:
                done = False

            image = self.grid.encode()
            obs = {
                'image': image,
                'time_step': [self.adversary_step_count],
                'random_z': self.generate_random_z()
            }

        if done:
            self.reset_metrics()
            self.compute_metrics()

        return obs, 0, done, {}

    def reset_random(self):
        if self.fixed_environment:
            self.seed(self.seed_value)

        """Use domain randomization to create the environment."""
        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])

        self.step_count = 0
        self.adversary_step_count = 0

        # Current position and direction of the agent
        self.reset_agent_status()

        self.agent_start_pos = None
        self.goal_pos = None

        # Extra metrics
        self.reset_metrics()

        # Create empty grid
        self._gen_grid(self.width, self.height)

        # Randomly place goal
        self.goal_pos = self.place_obj(minigrid.Goal(), max_tries=100)

        # Randomly place agent
        self.agent_start_dir = self._rand_int(0, 4)
        self.agent_start_pos = self.place_one_agent(0, rand_dir=False)

        # Randomly place walls
        if self.resample_n_clutter:
            n_clutter = self._resample_n_clutter()
        else:
            n_clutter = int(self.n_clutter / 2)  # Based on original PAIRED logic

        for _ in range(n_clutter):
            self.place_obj(minigrid.Wall(), max_tries=100)

        self.compute_metrics()

        return self.reset_agent()


class MiniAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50)


class NoisyAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(goal_noise=0.3)


class MediumAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=30, size=10, agent_view_size=5, max_steps=200)


class GoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, fixed_environment=fixed_environment, seed=seed, max_steps=250)


class TaskEmbedGoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None,
                 use_latent_task=True, task_ae=None, ae_type=None, latent_task_exp=None,
                 latent_space_discrete_max=None, latent_dim=None,
                 latent_dim_max=None, latent_dim_min=None, cuda=None,
                 enc_activation="", activation_scalar=0, word_embed_size=0, deterministic_vae=True):
        super().__init__(choose_goal_last=True, fixed_environment=fixed_environment, seed=seed, max_steps=250,
                         use_latent_task=use_latent_task,
                         ae_type=ae_type, task_ae=task_ae,
                         latent_task_exp=latent_task_exp, latent_dim=latent_dim,
                         latent_space_discrete_max=latent_space_discrete_max,
                         latent_dim_max=latent_dim_max,
                         latent_dim_min=latent_dim_min,
                         cuda=cuda, enc_activation=enc_activation,
                         activation_scalar=activation_scalar,
                         word_embed_size=word_embed_size,
                         deterministic_vae=deterministic_vae)

class TaskEmbed25BGoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None,
                 use_latent_task=True, task_ae=None, ae_type=None, latent_task_exp=None,
                 latent_space_discrete_max=None, latent_dim=None,
                 latent_dim_max=None, latent_dim_min=None, cuda=None,
                 enc_activation="", activation_scalar=0, word_embed_size=0, deterministic_vae=True):
        super().__init__(choose_goal_last=True, fixed_environment=fixed_environment, seed=seed, max_steps=250,
                         use_latent_task=use_latent_task,
                         ae_type=ae_type, task_ae=task_ae,
                         latent_task_exp=latent_task_exp, latent_dim=latent_dim,
                         latent_space_discrete_max=latent_space_discrete_max,
                         latent_dim_max=latent_dim_max,
                         latent_dim_min=latent_dim_min,
                         cuda=cuda, enc_activation=enc_activation,
                         activation_scalar=activation_scalar,
                         word_embed_size=word_embed_size,
                         deterministic_vae=deterministic_vae,
                         n_clutter=25)

class GoalLastAdversarialEnv30(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, n_clutter=30, fixed_environment=fixed_environment, seed=seed,
                         max_steps=250)


class GoalLastAdversarialEnv60(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(choose_goal_last=True, n_clutter=60, fixed_environment=fixed_environment, seed=seed,
                         max_steps=250)


class GoalLastOpaqueWallsAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, see_through_walls=False,
            fixed_environment=fixed_environment, seed=seed, max_steps=250)


class GoalLastFewerBlocksAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, n_clutter=25,
            fixed_environment=fixed_environment, seed=seed, max_steps=250)


class GoalLastFewerBlocksAdversarialEnv_WN(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, n_clutter=25,
            fixed_environment=fixed_environment, seed=seed, max_steps=250,
            editor_actions='walls_none')


class GoalLastFewerBlocksAdversarialEnv_WNG(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, n_clutter=25,
            fixed_environment=fixed_environment, seed=seed, max_steps=250,
            editor_actions='walls_none_goal')


class GoalLastVariableBlocksAdversarialEnv_WNG(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, n_clutter=60, resample_n_clutter=True,
            fixed_environment=fixed_environment, seed=seed, max_steps=250,
            editor_actions='walls_none_goal')


class GoalLastVariableBlocksAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, n_clutter=60, resample_n_clutter=True,
            fixed_environment=fixed_environment, seed=seed, max_steps=250)


class GoalLastEmptyAdversarialEnv_WNG(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, n_clutter=0,
            fixed_environment=fixed_environment, seed=seed, max_steps=250,
            editor_actions='walls_none_goal')


class GoalLastFewerBlocksOpaqueWallsAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(
            choose_goal_last=True, n_clutter=25, see_through_walls=False,
            fixed_environment=fixed_environment, seed=seed, max_steps=250)


class MiniGoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self, fixed_environment=False, seed=None):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50,
                         choose_goal_last=True, fixed_environment=fixed_environment, seed=seed)


class FixedAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=50, size=15, agent_view_size=5, max_steps=50, fixed_environment=True)


class EmptyMiniFixedAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=0, size=6, agent_view_size=5, max_steps=50, fixed_environment=True)


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-Adversarial-v0',
    entry_point=module_path + ':AdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-TaskEmbedAdversarial-v0',
    entry_point=module_path + ':TaskEmbedGoalLastAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-TaskEmbed25BAdversarial-v0',
    entry_point=module_path + ':TaskEmbed25BGoalLastAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-MiniAdversarial-v0',
    entry_point=module_path + ':MiniAdversarialEnv',
    max_episode_steps=50,
)

register.register(
    env_id='MultiGrid-NoisyAdversarial-v0',
    entry_point=module_path + ':NoisyAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-MediumAdversarial-v0',
    entry_point=module_path + ':MediumAdversarialEnv',
    max_episode_steps=200,
)

register.register(
    env_id='MultiGrid-GoalLastAdversarial-v0',
    entry_point=module_path + ':GoalLastAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastOpaqueWallsAdversarial-v0',
    entry_point=module_path + ':GoalLastOpaqueWallsAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastFewerBlocksAdversarial-v0',
    entry_point=module_path + ':GoalLastFewerBlocksAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastFewerBlocksAdversarial-EditWN-v0',
    entry_point=module_path + ':GoalLastFewerBlocksAdversarialEnv_WN',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastFewerBlocksAdversarial-EditWNG-v0',
    entry_point=module_path + ':GoalLastFewerBlocksAdversarialEnv_WNG',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastVariableBlocksAdversarialEnv-v0',
    entry_point=module_path + ':GoalLastVariableBlocksAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastVariableBlocksAdversarialEnv-Edit-v0',
    entry_point=module_path + ':GoalLastVariableBlocksAdversarialEnv_WNG',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastEmptyAdversarialEnv-Edit-v0',
    entry_point=module_path + ':GoalLastEmptyAdversarialEnv_WNG',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-GoalLastFewerBlocksOpaqueWallsAdversarial-v0',
    entry_point=module_path + ':GoalLastFewerBlocksOpaqueWallsAdversarialEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-MiniGoalLastAdversarial-v0',
    entry_point=module_path + ':MiniGoalLastAdversarialEnv',
    max_episode_steps=50,
)

register.register(
    env_id='MultiGrid-FixedAdversarial-v0',
    entry_point=module_path + ':FixedAdversarialEnv',
    max_episode_steps=50,
)

register.register(
    env_id='MultiGrid-EmptyMiniFixedAdversarial-v0',
    entry_point=module_path + ':EmptyMiniFixedAdversarialEnv',
    max_episode_steps=50,
)

register.register(
    env_id='MultiGrid-GoalLastAdversarialEnv30-v0',
    entry_point=module_path + ':GoalLastAdversarialEnv30',
    max_episode_steps=50,
)

register.register(
    env_id='MultiGrid-GoalLastAdversarialEnv60-v0',
    entry_point=module_path + ':GoalLastAdversarialEnv60',
    max_episode_steps=50,
)
