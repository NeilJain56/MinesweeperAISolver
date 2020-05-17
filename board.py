import random

# Generates a board given a dimension (d) and number of mines (n)
# where every cell is either a * --> mine or positive integer --> clue
def generateBoard(dim, bombs):

    # If the number of mines is greater than the number of cells on the board,
    # make the number of mines equal to the number of cells on the board.
    if bombs > dim*dim:
        bombs = dim * dim

    # Initialize the board to all 0
    board = [[0 for p in range(dim)] for k in range(dim)]

    while bombs > 0:

        # Generate a random coordinate on the board for a potential mine
        mineX = random.randint(0, dim - 1)
        mineY = random.randint(0, dim - 1)

        # If the random coordinate is on a mine, generate a new random coordinate
        # until we land on a cell that is a 0.
        while board[mineX][mineY] == "*":
            mineX = random.randint(0, dim - 1)
            mineY = random.randint(0, dim - 1)

        # Place a mine (1) on the valid random coordinate
        board[mineX][mineY] = "*"

        # Reduce the number of mines needed to be placed by 1
        bombs = bombs - 1

    for i in range(dim):
        for j in range(dim):
            # If it is a mine, do not generate a clue about the surrounding cells
            if board[i][j] == "*":
                continue
            # Iterate through all possible neighbors
            neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1)]

            # Only if the neighbor is in the bounds of the graph and the neighbor is a mine
            # add 1 to the clue
            board[i][j] = sum([1 for (i,j) in neighbors if checkPoint(i, j, dim) and board[i][j] == "*"])

    return board

# Helper function to check if a certain point is between 0 and the graphs dimensions
def checkPoint(x, y, dim):
    if (0 <= x < dim) and (0 <= y < dim):
        return True
    return False

# Helper Function used to print boards in a pretty way
def printMap(graph):
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            print(graph[i][j], end=" ")
        print("")

