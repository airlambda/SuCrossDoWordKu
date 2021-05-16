import copy
import numpy as np
import math
import random
import multiprocessing
import sys
import time

"""
A hopefully more optimal hexadecimal Sudoku solver using homogenous simulated annealing,
and parallelism.
Works best with pypy3.
"""

# Define starting Sudoku array
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

"""
The following nomenclature will be used:
- Grid, used to refer to the full 16 x 16 problem
- Subgrid, used to refer to the 4 x 4 grids within a Grid, which need to contain numbers 0 to 15.
- Cell, used to refer to one value in the Grid
"""

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
                    subgrid.append(grid[4 * box_i + i, 4 * box_j + j])
            subgrids.append(subgrid)
    return np.array(subgrids)

def count_free_cells(subgrid):
    """
    Returns the number of non-fixed cells in a subgrid
    """
    arr = np.delete(subgrid, np.where(subgrid == -1))
    return 16 - len(np.unique(arr))

def evaluate_markov_chain_length(grid):
    """
    Returns the value of the Markov Chain length for the 
    homogenous simulated annealing process
    """
    subgrids = get_subgrids(grid)
    ml = 0
    for i in range(16):
        ml += count_free_cells(subgrids[i])
    return (ml ** 2)

def random_solve_subgrid(grid):
    """
    Takes initial grid and does a naive solve per subgrid
    Ensures that there exists one of each number (0-15) in each subgrid
    """
    subgrid = get_subgrids(grid)
    
    for i in range(len(subgrid)):
        current_subgrid = subgrid[i]
        available_numbers = list((range(0, 16)))
        existing_numbers = [x for x in current_subgrid if x >= 0]
        available_numbers = [number for number in available_numbers if not number in existing_numbers]
        random.shuffle(available_numbers)

        # Fill out the grid on a subgrid basis
        [start_x, start_y] = divmod(i, 4)
        list_to_subgrid(available_numbers, start_x*4, start_y*4, grid)
    
    return grid
        
def list_to_subgrid(input_list, start_x, start_y, grid):
    """
    Takes an input list of ints, starting (X,Y) position, grid and fills out a subgrid
    """
    x = 0
    y = 0
    available_number_iterator = iter(input_list)
    for i in range(16):
        x = i % 4
        if x == 0 and i !=0: 
            y += 1
        if grid[start_x + x, start_y + y] == -1:
            grid[start_x + x, start_y + y] = next(available_number_iterator)
        else:
            continue
    return grid

def neighborhood_operator(starting_grid, grid):
    """
    Randomly swaps 2 cells in the same subgrid.
    Returns the output (transformed) grid, and the locations of the two cells
    that were swapped
    """
    swap = True
    output = copy.deepcopy(grid)

    while swap:
        i = random.randint(0, 15)
        j = random.randint(0, 15)

        # Check if original starting grid had a fixed value at cell(i, j)
        if starting_grid[i][j] != -1:
            continue
        else:
            first_subgrid = subgrid_index(i ,j)
            k = random.randint(0, 15)
            l = random.randint(0, 15)
            if (subgrid_index(k, l) == first_subgrid and starting_grid[k, l] == -1 and grid[i, j] != grid[k, l]):
                output[i, j], output[k, l] = grid[k, l], grid[i, j]
                cell_1_location = (i, j)
                cell_2_location = (k, l)
                swap = False
    
    return output, cell_1_location, cell_2_location

def evaluate_initial_cost(grid):
    """
    Evaluates the number of values that are not present for 
    every row and column and then sums them up
    Returns the "cost", which is the sum of every non-present value 
    for each row and column, as well as the scores for each row and column
    in list form
    """
    row_scores = []
    column_scores = []

    for row in range(16):
        row_scores.append(16 - len(set(grid[row, :])))
    
    for column in range(16):
        column_scores.append(16 - len(set(grid[:, column])))

    total = sum(row_scores) + sum(column_scores)

    return total, row_scores, column_scores

def update_cost(grid, row_scores, column_scores, cell_1_row, cell_1_column, cell_2_row, cell_2_column):
    """ 
    Runs delta-evaluation on initial cost according to swapped cells
    instead of redoing main evaluate_initial_cost routine again.
    Returns new cost, row and column lists as evaluate_initial_cost
    """
    row_scores[cell_1_row] = (16 - len(set(grid[cell_1_row, :])))
    row_scores[cell_2_row] = (16 - len(set(grid[cell_2_row, :])))

    column_scores[cell_1_column] = (16 - len(set(grid[:, cell_1_column])))
    column_scores[cell_2_column] = (16 - len(set(grid[:, cell_2_column])))

    total = sum(row_scores) + sum(column_scores)

    return total, row_scores, column_scores

def acceptance_probability(cost_current, cost_variant, temperature):
    """
    Returns a float between 0.0 and 1.0 depending on:
    cost_current vs. cost_variant.
    If cost_variant is less than cost current, a 1.0 is returned
    so that the variant is the new best guess.
    """
    if cost_variant < cost_current:
        return 1.0
    else:
        return math.exp((cost_current - cost_variant) / temperature)

def main(starting_grid):

    # Set up
    random.seed()
    INITIAL_TEMP = 500
    STOP_TEMP = 0.75
    COUNTER_MOD = 1000
    COOLDOWN_RATE = 0.05
    MAX_MARKOV_CHAIN_LENGTH = evaluate_markov_chain_length(starting_grid)
    MAX_NUM_MARKOV_CHAINS_WITHOUT_IMPROVEMENT = 20
    temperature = INITIAL_TEMP
    counter = 0
    ml = 0
    consecutive_chains_without_improvement = 0
    original = copy.deepcopy(starting_grid)

    # Starting out, our first solve is our best (best = current)
    current = random_solve_subgrid(starting_grid)
    cost_current, row_scores_current, column_scores_current = evaluate_initial_cost(current)

    best = copy.deepcopy(current)
    cost_best, row_scores_best, column_scores_best = evaluate_initial_cost(best)

    while temperature > STOP_TEMP:
        while ml <= MAX_MARKOV_CHAIN_LENGTH:

            # Create variant and find the cost of the variant state
            variant, cell_1_location, cell_2_location = neighborhood_operator(original, current)
            cost_variant, row_scores_variant, column_scores_variant = update_cost(variant, row_scores_current, column_scores_current, cell_1_location[0], cell_1_location[1], cell_2_location[0], cell_2_location[1])
            
            # Accept the variant state if:
            # a) The cost is less than the current cost
            # b) It passes a probabilistic threshold defined by the temperature
            prob_threshold = random.uniform(0, 1)
            if acceptance_probability(cost_current, cost_variant, temperature) > prob_threshold:
                current = copy.deepcopy(variant)
                cost_current = cost_variant
                row_scores_current = row_scores_variant
                column_scores_current = column_scores_variant
            
            # Now check to see if the current version of the grid has a lower
            # cost than the previous best version of the grid
            if cost_best > cost_current:
                best = copy.deepcopy(current)
                cost_best = cost_current
                row_scores_best = row_scores_current
                column_scores_best = column_scores_current
                consecutive_chains_without_improvement = 0
            
            # Update console
            if counter % COUNTER_MOD == 0:
                print(f"Lowest cost: {cost_best} | Current cost: {cost_current} | Temperature: {temperature} | Counter: {counter} | Markov Chain Length: {ml} | Num Markov Chains: {consecutive_chains_without_improvement}")
            if cost_best == 0:
                break

            ml += 1
            counter += 1
    
        # Check for reheat condition
        if consecutive_chains_without_improvement >= MAX_NUM_MARKOV_CHAINS_WITHOUT_IMPROVEMENT:
            temperature = INITIAL_TEMP
            current = random_solve_subgrid(starting_grid)
            cost_current, row_scores_current, column_scores_current = evaluate_initial_cost(current)
            cost_best = cost_current
            consecutive_chains_without_improvement = 0
            ml = 0
            print("Reheat condition triggered")
        else:
        # Keep on lowering the temp if the solution has not been found
            temperature *= (1 - COOLDOWN_RATE)
            ml = 0
            consecutive_chains_without_improvement += 1
    
    # Summarise
    if cost_best == 0:
        print("Sudoku solved:\n")
        print(best)
        return best
    else:
        print("Could not find solution to sudoku")
        print(best)
        return 0

def multiprocess_function(event, starting_grid):
    x = main(starting_grid)
    if x == 0:
        print("This process failed to solve a sudoku")
        event.set()

if __name__ == "__main__":
    processes = []
    event = multiprocessing.Event()
    for i in range(5):
        p = multiprocessing.Process(target=multiprocess_function,
                                    args=(event,starter,))
        p.start()
        processes.append(p)
    
    while True:
        if event.is_set():
            for i in processes:
                i.terminate()
                print(f"Process {i} has terminated")
            sys.exit(1)
        time.sleep(2)