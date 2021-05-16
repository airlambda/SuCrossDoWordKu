# SuCrossDoWordKu
A terrible idea for a cross between a crossword and a sudoku

Playable here: https://docs.google.com/spreadsheets/d/1bi8H_LWVdBgSS4h8ewq9jVXrmFexhOH1y8kbynvFFEI/edit?usp=sharing

Having said idea meant creating a way of solving a 16 x 16 hexadecimal sudoku.

Certain cells are "fixed" and placed manually due to making it make sense from a crossword point of view.

# Word_List
Word_List contains a search of words in the English dictionary that can be spelled out using one of A-F only. Run `valid_words.py` and change the constraints to see other words with only a single instance of a subset of letters in them.

# Sudoku_Solver
This folder contains 3 separate ways of trying to solve a 16 x 16 hexadecimal sudoku.
- `sudoku_solver.py` is the original method based off backtracking and recursion. Sadly, with how sparse the grid was and the recursion limits on Python, this code does not really get too far in solving a 16x16 sudoku.
- `sudoku_solver_annealing.py` is a naive first attempt at a metaheuristic simulated annealing algorithm to solve the 16 x 16 sudoku problem. 
- `sudoku_solver_annealing_optimal.py` is an improved attempt using homogenous simulated annealing. The ideas were sourced from [this](http://rhydlewis.eu/papers/META_CAN_SOLVE_SUDOKU.pdf) paper by Rhyd Lewis. I also made an attempt at incorporating multiprocessing since the problem is easily parallelisable.
