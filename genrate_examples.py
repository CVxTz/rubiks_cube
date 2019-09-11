import json
from collections import Counter
from multiprocessing import Pool
from random import choice
from uuid import uuid4

import numpy as np
import pycuber as pc
from pycuber.solver import CFOPSolver
from tqdm import tqdm

action_map = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11,
              'F2': 12, 'B2': 13, 'U2': 14, 'D2': 15, 'L2': 16, 'R2': 17, "F2'": 18, "B2'": 19, "U2'": 20, "D2'": 21,
              "L2'": 22, "R2'": 23}
inv_action_map = {v: k for k, v in action_map.items()}
color_map = {'green': 0, 'blue': 1, 'yellow': 2, 'red': 3, 'orange': 4, 'white': 5}

color_list_map = {'green': [1, 0, 0, 0, 0, 0], 'blue': [0, 1, 0, 0, 0, 0], 'yellow': [0, 0, 1, 0, 0, 0],
                  'red': [0, 0, 0, 1, 0, 0], 'orange': [0, 0, 0, 0, 1, 0], 'white': [0, 0, 0, 0, 0, 1]}


def flatten(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.append(x[i][j].colour)
    return flat


def flatten_1d_b(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.extend(color_list_map[x[i][j].colour])
    return flat


def order(data):
    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    probs = [float(c) / len(data) for c in counts.values()]

    return max(probs)


def perc_solved_cube(cube):
    flat = flatten(cube)
    perc_side = [order(flat[i:(i + 9)]) for i in range(0, 9 * 6, 9)]
    return np.mean(perc_side)


def gen_sample(forward_transform, backward_transform):
    cube = pc.Cube()

    forward = pc.Formula(forward_transform)

    cube(forward)

    backward = pc.Formula(backward_transform)

    sample_X = []
    sample_Y = []
    cubes = []

    for s in backward:
        sample_X.append(flatten_1d_b(cube))
        sample_Y.append(action_map[s.name])
        cubes.append(cube.copy())
        cube(s.name)

    return sample_X, sample_Y, cubes


def write_sample(i=0):
    n_steps = 10 + np.random.randint(0, 20)

    cube = pc.Cube()

    forward = " ".join([choice(list(action_map.keys())) for _ in range(n_steps)])

    cube(forward)

    solver = CFOPSolver(cube)

    backward = solver.solve(suppress_progress_messages=True)

    sample_X, sample_Y, cubes = gen_sample(forward, backward)

    d = {"forward": forward,
         "backward": " ".join([a.name for a in backward]),
         "sample_X": sample_X,
         "sample_Y": sample_Y}

    json.dump(d, open('samples/%s.json' % uuid4().hex, "w"), indent=1)

    return 1


p = Pool(4)

for a in tqdm(p.map(write_sample, range(100000))):
    print(a)
