import copy
import numpy as np
from copy import deepcopy
import math
import random

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

def subgrid_index(i, j):
    """
    Returns the index of the subgrid given an (i, j) row, column reference
    """
    return (i // 4) * 4 + j // 4

def get_subgrids(grid):
    """
    Returns all subgrids from a given grid
    """
    subgrids = []
    for box_i in range(4):
        for box_j in range(4):
            subgrid = []
            for i in range(4):
                for j in range(4):
                    subgrid.append(grid[4 * box_i + i][4 * box_j + j])
            subgrids.append(subgrid)
    return np.array(subgrids)

def random_solve(grid):
    """
    Checks what numbers are already in the grid, creates a list
    of available numbers, then fills in the grid according to that list.
    The restriction is that there should be 16 of each value in the entire grid.
    """
    existing_numbers = []
    available_numbers = []
    output = np.zeros((16, 16), dtype=int)
    
    # Find out what numbers are already used up
    for i in range(16):
        for j in range(16):
            if grid[i][j] != -1:
                existing_numbers.append(grid[i][j])
    
    # Create list of numbers to be filled in
    for num in range(16):
        occurrences = existing_numbers.count(num)
        available_cells = 16 - occurrences
        if available_cells > 0:
            available_numbers += available_cells * [num]
    random.shuffle(available_numbers)

    # Fill in numbers
    available_number_iterator = iter(available_numbers)
    for i in range(16):
        for j in range(16):
            if grid[i][j] == -1:
                output[i][j] = next(available_number_iterator)
            else:
                output[i][j] = grid[i][j]
    
    return output


def evaluate_energy(grid):
    """
    Calculate the energy of a given Sudoku board. 0 is solved, 1 is the worst score.
    Each box in the board can have a max score of 3 corresponding to:
    +1 : non-unique element in row
    +1 : non-unique element in column
    +1 : non-unique element in subgrid
    """
    norm_constant = 16 * 16 * 3
    energy = norm_constant
    subgrids = get_subgrids(grid)

    # Find num unique elements on all rows
    for i in range(16):
        row_contents = [x for x in grid[i][:] if x >= 0]
        unique_in_row = len(set(row_contents))
        energy -= unique_in_row
    
    # Find num unique elements on all columns
    for j in range(16):
        col_contents = [y for y in grid[:][j] if y >= 0]
        unique_in_col = len(set(col_contents))
        energy -= unique_in_col
    
    # Find num unique elements in subgrid
    for k in range(16):
        subgrid_contents = [z for z in subgrids[k] if z >= 0]
        unique_in_subgrid = len(set(subgrid_contents))
        energy -= unique_in_subgrid
    
    energy /= norm_constant
    return energy  


def create_variant_grid(grid):
    """
    Chooses 2 cells at random to swap
    """
    while True:
        y1 = random.randint(0, 15)
        x1 = random.randint(0, 15)
        if grid[y1][x1] != -1:
            swap_value_1 = grid[y1][x1]
            break
    while True:
        y2 = random.randint(0, 15)
        x2 = random.randint(0, 15)
        if grid[y2][x2] != -1 and x2 != x1 and y2 != y1 :
            swap_value_2 = grid[y2][x2]
            break
    
    # Create the new variant grid
    new_grid = copy.deepcopy(grid)
    new_grid[y1][x1] = swap_value_2
    new_grid[y2][x2] = swap_value_1

    return new_grid

def acceptance_probability(energy_current, energy_variant, temperature):
    """
    Returns a float between 0.0 and 1.0 depending on:
    cost_current vs. cost_variant.
    If cost_variant is less than cost current, a 1.0 is returned
    so that the variant is the new best guess.
    """
    if energy_variant < energy_current:
        return 1.0
    else:
        return math.exp((energy_current - energy_variant) / temperature)

def main(grid):
    # Save the current best estimate
    best = copy.deepcopy(grid)

    # Set up
    random.seed()
    INITIAL_TEMP = 10000
    STOP_TEMP = 0.75
    COUNTER_ITER = 100
    COOLDOWN_RATE = 0.0001
    temperature = INITIAL_TEMP
    counter = 0

    while temperature > STOP_TEMP:

        # Randomly solve current grid
        solved_grid = random_solve(grid)

        # Generate new variant
        variant_state = create_variant_grid(solved_grid)

        # Determine the energy of the current grid and the variant grid
        energy_current = evaluate_energy(best)
        energy_variant = evaluate_energy(variant_state)

        # Accept the variant state if it passes some probability threshold
        random_value = random.uniform(0, 1)
        if acceptance_probability(energy_current, energy_variant, temperature) > random_value:
            solved_grid = copy.deepcopy(variant_state)
        
        # Check to see if the new lowest energy is less than the lowest energy of the best
        if evaluate_energy(best) > evaluate_energy(solved_grid):
            best = copy.deepcopy(solved_grid)

        # Update console
        energy_best = evaluate_energy(best)
        energy_current = evaluate_energy(solved_grid)
        if counter % COUNTER_ITER == 0:
            print(f"Best energy: {energy_best} | Current energy: {energy_current} | Temperature: {temperature} | Counter: {counter}")
        if energy_best == 0.0:
            break

        # Keep on lowering the temp if the solution has not been found
        temperature *= (1 - COOLDOWN_RATE)
        counter += 1

    # Summarise
    if energy_best == 0.0:
        print(best)
    else:
        print("Could not find solution to sudoku")
        print(best)

if __name__ == "__main__":
    main(starter)