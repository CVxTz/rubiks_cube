from random import choice
import pycuber as pc
import numpy as np
from collections import Counter
from pycuber.solver import CFOPSolver
import time

action_map = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11}
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
    perc_side = [order(flat[i:(i+9)]) for i in range(0, 9*6, 9)]
    return np.mean(perc_side)


def gen_sample(n_steps=6):

    cube = pc.Cube()

    transformation = [choice(list(action_map.keys())) for _ in range(n_steps)]

    my_formula = pc.Formula(transformation)

    cube(my_formula)

    my_formula.reverse()

    sample_X = []
    sample_Y = []
    cubes = []

    for s in my_formula:
        sample_X.append(flatten_1d_b(cube))
        sample_Y.append(action_map[s.name])
        cubes.append(cube.copy())
        cube(s.name)
        print(perc_solved_cube(cube))

    return sample_X, sample_Y, cubes


sample_X, sample_Y, cubes = gen_sample(n_steps=10)

sample_X = np.array(sample_X)
sample_Y = np.array(sample_Y)

print(sample_X.shape, sample_Y.shape)

for x in cubes:
    print(x)

    s = time.time()
    solver = CFOPSolver(x)

    solution = solver.solve(suppress_progress_messages=True)

    print(time.time()-s, solution)