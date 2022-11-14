import gym_minigrid
import numpy as np
from gym_minigrid import minigrid
import gym

def xlate_grid(grid):
    ret = []
    for el in grid:
        if el is None:
            v = 0
        elif isinstance(el, gym_minigrid.minigrid.Wall):
            v = 1
        elif isinstance(el, gym_minigrid.minigrid.Goal):
            v = "G"
        else:
            v = "A"
        ret.append(v)
    return ret


def set_grid(env, grid):
    #print("wall count:", sum([l == 1 for l in grid]))
    for loc in range(len(grid)):
        y = (loc) // (env.width)
        x = loc % env.width

        if grid[loc] == 1:
            env.put_obj(minigrid.Wall(), x, y)
        elif grid[loc] == "G":
            env.put_obj(minigrid.Goal(), x, y)
        elif grid[loc] == "A":
            env.place_agent_at_pos(0, np.array([x, y]), rand_dir=False)

"""
updated version is in clutr/results/minigrid_analysis/tsne_on_minigrid_envs.py
def get_param_vec_from_raw_grid(grid, width = 15):

    '
    visually e.g when rendered with minigrid env:
    x,y == 0,0 is upper left... x grows down, y grows right

    loc to (x,y) ->  (loc//width, loc%width)
    example:
    agent = 153, goal = 106
    agent = (10, 3)
    goal = (106//15, 106%15) =>  (7,1)

    :param grid:
    :param width:
    :return:
    '''
    goal = None
    agent = None
    obstacles = []

    for loc in range(len(grid)):
        y = (loc) // (width)
        x = loc % width

        # skip boundaries
        if x in [0, width - 1] or y in [0, width - 1]: continue
        #print(grid[loc])
        if grid[loc] == 1:
            obstacles.append(loc)
        elif grid[loc] == "G":
            goal = loc
        elif grid[loc] == "A":
            agent = loc

    obstacles.sort()
    param_vec = [agent, goal] + obstacles
    return param_vec
"""


