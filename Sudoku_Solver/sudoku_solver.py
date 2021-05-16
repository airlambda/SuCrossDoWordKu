import numpy as np
from copy import deepcopy
import sys
import threading

"""
A 16x16 Sudoku solver using backtracking and recursion. 
"""
threading.stack_size(67108864) # 64MB stack
sys.setrecursionlimit(2 ** 20) # something real big
                               # you actually hit the 64MB limit first
                               # going by other answers, could just use 2**32-1

starter = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, 15],
                    [-1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 12, 14, 13],
                    [-1, -1, -1, -1,  0, -1, -1, -1, -1, -1, 15, 12, 11, -1, -1, -1],
                    [-1, -1, -1, -1, -1, 12, -1, 11, 10, 13, 14, -1, -1, -1,  0, -1],
                    [-1, -1, -1, -1, -1, 11, 10, 13, -1, -1, -1, -1, -1,  0, -1, -1],
                    [-1, -1, 12, 10, 15, 14, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1],
                    [11, 14, 13, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [13, 12, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 15, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 10, 11, 14, 13, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1],
                    [-1, -1, -1, -1, 12, 10, 11, -1, -1, -1,  0, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 12, 10, 13, 14, -1, -1, -1, -1, -1,  0],
                    [-1, -1, -1, -1, -1, -1, -1, -1,  0, 15, 12, 10, -1, -1, -1, -1],
                    [-1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, 11, 14, 15, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, 10, 11, 12]])

def get_subgrids(grid):
    subgrids = []
    for box_i in range(4):
        for box_j in range(4):
            subgrid = []
            for i in range(4):
                for j in range(4):
                    subgrid.append(grid[4 * box_i + i, 4 * box_j + j])
            subgrids.append(subgrid)
    return np.array(subgrids)

def get_candidates(grid):
    def subgrid_index(i, j):
        return (i // 4) * 4 + j // 4
    subgrids = get_subgrids(grid)
    trial = []
    for i in range(16):
        row_candidates = []
        for j in range(16):
            if grid[i][j] == -1:
                row = set(grid[i,:])
                col = set(grid[:,j])
                sub = set(subgrids[subgrid_index(i,j)])
                common = row | col | sub
                candidates = set(range(16)) - common
                row_candidates.append(list(candidates))
            else:
                row_candidates.append([grid[i, j]])
        trial.append(row_candidates)
    return trial

def fill_singles(grid, candidates=None):
    grid = grid.copy()
    if not candidates:
        candidates = get_candidates(grid)
    any_fill = True
    while any_fill:
        any_fill = False
        for i in range(16):
            for j in range(16):
                if len(candidates[i][j]) == 1 and grid[i][j] == -1:
                    grid[i][j] = candidates[i][j][0]
                    candidates = merge(get_candidates(grid),
                                       candidates)
                    any_fill = True
    return grid

def merge(candidates_1, candidates_2):
    candidates_min = []
    for i in range(16):
        row = []
        for j in range(16):
            if len(candidates_1[i][j]) < len(candidates_2[i][j]):
                row.append(candidates_1[i][j][:])
            else:
                row.append(candidates_2[i][j][:])
        candidates_min.append(row)
    return candidates_min

def is_solution(grid):
    if np.all(np.sum(grid, axis=1) == 120) and \
       np.all(np.sum(grid, axis=0) == 120) and \
       np.all(np.sum(get_subgrids(grid), axis=1) == 120):
        return True
    return False

def is_valid_grid(grid):
    candidates = get_candidates(grid)
    for i in range(16):
        for j in range(16):
            if len(candidates[i][j]) == 0:
                return False
    return True

def filter_candidates(grid):
    test_grid = grid.copy()
    candidates = get_candidates(grid)
    filtered_candidates = deepcopy(candidates)
    for i in range(16):
        for j in range(16):
            if grid[i, j] == -1:
                for candidate in candidates[i][j]:
                    test_grid[i, j] = candidate
                    if not is_valid_grid(fill_singles(test_grid)):
                        filtered_candidates[i][j].remove(candidate)
                    test_grid[i, j] = -1
    return filtered_candidates

def make_guess(grid, candidates=None):
    grid = grid.copy()
    if not candidates:
        candidates = get_candidates(grid)
    # Getting the shortest number of candidates > 1:
    min_len = sorted(list(set(map(
       len, np.array(candidates, dtype=object).reshape(1,256)[0]))))[1]
    for i in range(16):
        for j in range(16):
            if len(candidates[i][j]) == min_len:
                for guess in candidates[i][j]:
                    grid[i, j] = guess
                    solution = filtered_solve(grid)
                    if solution is not None:
                        return solution
                    # Discarding a wrong guess
                    grid[i, j] = -1

def filtered_solve(grid):
    candidates = filter_candidates(grid)
    grid = fill_singles(grid, candidates)
    if is_solution(grid):
        return grid
    if not is_valid_grid(grid):
        return None
    print(grid)
    return make_guess(grid, candidates)

thread = threading.Thread(target=filtered_solve(starter))
thread.start()