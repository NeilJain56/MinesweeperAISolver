import board
import matrix
import numpy as np
import random
from collections import deque
from copy import deepcopy
import m as mat

class MinRiskAgent():

    def __init__(self, board, mines):

        # Original Board
        self.board = board

        # Saves matrices we have solved
        self.eq = {}

        # Expected Number of Squares that can be worked out
        self.knownCells = {}

        # Total number of squares stepped on without knowing what they are
        self.risk = 0

        # Probabilistic Knowledge Base
        self.pkb = {}

        # Board the agent sees
        self.agentBoard = []

        # Total mines in a grid
        self.mines = mines

        # Number of mines exploded
        self.minesHit = 0

        # Set of all cells flagged as mines
        self.minesFlagged = set()

        # Set of cells that were visited
        self.visited = set()

        # List of cells that will be opened next
        self.moveList = []
        
        # Initialize agentBoard to all unknowns (?)
        self.agentBoard = [["?" for j in range(len(board))] for i in range(len(board))]
                        
        # Initialize PKB to 0.5 for every cell given that we don't know how 
        # many mines there are
        for i in range(len(board)):
            for j in range(len(board)):
                self.pkb[(i,j)] = 0.5

        # Initialize KnownCells Map to 0 for every cell given that we don't have 
        # any info
        for i in range(len(board)):
            for j in range(len(board)):
                self.knownCells[(i,j)] = 0

    # Method used to update list of active unknowns
    def updateAllUnknownCells(self, unknownCells):
        for cell in unknownCells:
            self.updateUnknownCell(cell)

    # Method used to calculate the risk for a specific cell
    def updateUnknownCell(self, unknownCell):
        grid = deepcopy(self.agentBoard)
        
        clues = self.getSetOfClues(unknownCell, grid)

        if len(clues) == 0:
            return

        x = unknownCell[0]
        y = unknownCell[1]

        grid[x][y] = "*"
        cellPKB = self.buildSubMatrixForRisk(clues, grid)
        R = self.countKnownCells(cellPKB)

        grid[x][y] = "S"
        cellPKB = self.buildSubMatrixForRisk(clues, grid)
        S = self.countKnownCells(cellPKB)

        q = self.pkb[unknownCell]
        expectedCells = float(q)*float(R) + float((1-q))*float(S)
        self.knownCells[unknownCell] = expectedCells

    # Gets the set of related clues around one cell
    def getSetOfClues(self, cell, grid):
        dim = len(grid)
        visited = [[False for p in range(dim)] for k in range(dim)]        
        visited[cell[0]][cell[1]] = True
        currentSet = []
        queue = deque()
        queue.append(cell)

        while len(queue) != 0:
            node = queue.popleft()
            for neighbor in self.getNeighbors(node):
                if (isinstance(grid[neighbor[0]][neighbor[1]], int)) and (not visited[neighbor[0]][neighbor[1]]):
                    currentSet.append(neighbor)
                    queue.append(neighbor)
                    visited[neighbor[0]][neighbor[1]] = True
                elif grid[neighbor[0]][neighbor[1]] == "?" and isinstance(grid[node[0]][node[1]], int) and not visited[neighbor[0]][neighbor[1]]:
                    for n in self.getNeighbors(neighbor):
                        if (isinstance(grid[n[0]][n[1]], int)) and (not visited[n[0]][n[1]]):
                            currentSet.append(n)
                            queue.append(n)
                            visited[n[0]][n[1]] = True
        return currentSet

    # Builds a submatrix for a list of clues
    def buildSubMatrixForRisk(self, clues, grid):
        count = 0
        colToCell = {}
        cellToCol = {}
        clueNeighbors = {}

        # Map columns to cells 
        for clue in clues:
            neighbors = self.getNeighbors(clue)
            clueNeighbors[clue] = neighbors
            for neighbor in neighbors:
                if neighbor not in cellToCol and grid[neighbor[0]][neighbor[1]] == "?": 
                    colToCell[count] = neighbor
                    cellToCol[neighbor] = count
                    count = count + 1

        m_size = len(colToCell)

        # m is the matrix that will contain the system of equations
        m = []

        # For every clue find the neighbors and for every neighbor that is unknown
        # place a 1 in the equation. Subtract from the clue that is going to be on 
        # left side of the equation if there are any neighbors that are flagged as 
        # mines or exploded as mines. 
        for clue in clueNeighbors.keys():
            neighbors = clueNeighbors[clue]
            mineNeighbors = 0
            newRow = [0 for j in range(m_size+1)]
            for neighbor in neighbors:
                x = neighbor[0]
                y = neighbor[1]
                if grid[x][y] == "?":
                    newRow[cellToCol[(x, y)]] = 1
                elif grid[x][y] == "*" or grid[x][y] == "F":
                    mineNeighbors = mineNeighbors + 1
            newRow[m_size] = grid[clue[0]][clue[1]] - mineNeighbors

            # If the newRow to be added is all 0, no need to add it in. 
            if not self.allZero(newRow):
                m.append(newRow)

        cellPkb = {}
        if len(m) != 0:
            # Get the probabilites for every clue adjacent cell
            currPkb = mat.get_probabilities(np.array(m))

            # Convert the column numbers to cells (column) -> (i,j)
            for key in currPkb:
               cellPkb[colToCell[key]] = currPkb[key]
            
        return cellPkb

    # Counts the number of cells that are 0 or 1 value in PKB 
    def countKnownCells(self, knownMap):
        count = 0
        for key in knownMap:
            if knownMap[key] == 1 or knownMap[key] == 0:
                count = count + 1
        return count

    # Method used to update the PKB using the sets of related clues
    def buildForBatch(self, clueList):
        for clues in clueList:
            self.buildSubMatrix(clues)

    # Builds the system of equations given a set of clues
    def buildSubMatrix(self, clues):
        count = 0
        colToCell = {}
        cellToCol = {}
        clueNeighbors = {}

        # Map columns to cells 
        for clue in clues:
            neighbors = self.getNeighbors(clue)
            clueNeighbors[clue] = neighbors
            for neighbor in neighbors:
                if neighbor not in cellToCol and self.agentBoard[neighbor[0]][neighbor[1]] == "?": 
                    colToCell[count] = neighbor
                    cellToCol[neighbor] = count
                    count = count + 1

        m_size = len(colToCell)

        # m is the matrix that will contain the system of equations
        m = []

        # For every clue find the neighbors and for every neighbor that is unknown
        # place a 1 in the equation. Subtract from the clue that is going to be on 
        # left side of the equation if there are any neighbors that are flagged as 
        # mines or exploded as mines. 
        for clue in clueNeighbors.keys():
            neighbors = clueNeighbors[clue]
            mineNeighbors = 0
            newRow = [0 for j in range(m_size+1)]
            for neighbor in neighbors:
                x = neighbor[0]
                y = neighbor[1]
                if self.agentBoard[x][y] == "?":
                    newRow[cellToCol[(x, y)]] = 1
                elif self.agentBoard[x][y] == "*" or self.agentBoard[x][y] == "F":
                    mineNeighbors = mineNeighbors + 1
            newRow[m_size] = self.agentBoard[clue[0]][clue[1]] - mineNeighbors

            # If the newRow to be added is all 0, no need to add it in. 
            if not self.allZero(newRow):
                m.append(newRow)

        if len(m) != 0 and not self.checkMatrix(clues, m):
            self.saveMatrix(clues, m)

            # Get the probabilites for every clue adjacent cell
            currPkb = mat.get_probabilities(np.array(m))
            cellPkb = {}

            # Convert the column numbers to cells (column) -> (i,j)
            for key in currPkb:
               cellPkb[colToCell[key]] = currPkb[key]        

            # Update the PKB using cellPKB
            for key in cellPkb:
                self.pkb[key] = cellPkb[key]

    # Saves every matrix we build and solve for to our EQ map.
    def saveMatrix(self, clues, m):
        tClue = tuple(clues)
        if tClue not in self.eq:
            self.eq[tClue] = m
        elif tClue in self.eq and self.eq[tClue] != m:
            self.eq[tClue] = m

    # Returns true if a matrix already exists in our EQ map
    # This reduces runtime by avoiding recalculation of clue sets we have already 
    # calculated for.
    def checkMatrix(self, clues, m):
        tClue = tuple(clues)
        if tClue in self.eq and self.eq[tClue] == m:
            return True
        return False

    # Flags all the cells that have a 1 probability of being a mine as a "F"
    def updateAgentBoardFromPKB(self):
        # Iterate through the whole PKB and find which cells have probability 1
        for key in self.pkb:
            if key not in self.visited and self.pkb[key] == 1:
                self.visited.add(key)
                self.minesFlagged.add(key)
                # An "F" on the agent board ia mine that is flagged and will
                # not be clicked on.
                self.agentBoard[key[0]][key[1]] = "F"
        
    # Returns a list of lists of clue cells, where one list in the larger list
    # represents a set of clues that share neighbors and should be evaluated
    # together. This drastically improves the runtime!
    def getRelatedSets(self):
        dim = len(self.board)
        visited = [[False for p in range(dim)] for k in range(dim)]
        relatedSets = []
        for cell in self.visited:
            if isinstance(self.agentBoard[cell[0]][cell[1]], int) and not visited[cell[0]][cell[1]] and self.atLeastOneUknownNeighbor(cell):
                visited[cell[0]][cell[1]] = True
                currentSet = []
                currentSet.append(cell)
                queue = deque()
                queue.append(cell)

                while len(queue) != 0:
                    node = queue.popleft()
                    for neighbor in self.getNeighbors(node):
                        if (isinstance(self.agentBoard[neighbor[0]][neighbor[1]], int)) and (not visited[neighbor[0]][neighbor[1]]) and self.sharesUnknownNeighbors(neighbor, node):
                            currentSet.append(neighbor)
                            queue.append(neighbor)
                        elif self.agentBoard[neighbor[0]][neighbor[1]] == "?" and not visited[neighbor[0]][neighbor[1]]:
                            for n in self.getNeighbors(neighbor):
                                if (isinstance(self.agentBoard[n[0]][n[1]], int)) and (not visited[n[0]][n[1]]):
                                    currentSet.append(n)
                                    queue.append(n)
                                    visited[n[0]][n[1]] = True
                        visited[neighbor[0]][neighbor[1]] = True
                relatedSets.append(currentSet)
        return relatedSets
    
    # Returns true if two cells share at least one unknown neighbor
    def sharesUnknownNeighbors(self, n, m):
        n = set(self.getNeighbors(n))
        m = set(self.getNeighbors(m))

        for k in n:
            if k in m and self.agentBoard[k[0]][k[1]] == "?":
                return True
        return False

        # Helper function that returns true if a cell has at least one unknown neighbor
    
    # Helper function that returns true if a cell has at least one unknown neighbor
    def atLeastOneUknownNeighbor(self, node):
        for n in self.getNeighbors(node):
            if self.agentBoard[n[0]][n[1]] == "?":
                return True
        return False

    # Calculates the next move by randomly choosing from all the cells 
    # that have a minimum probability, but if there are multiple cells
    # with 0 probability we return all of them to improve runtime. 
    def nextMove(self):
        safe, flags = self.basicInference()

        if len(safe) != 0 or len(flags) != 0:
            #print("Basic Inference")
            self.moveList.extend(safe)
            self.updateAgentBoardFromPKB()
            return safe

        # Get the sets of related clues 
        clues = self.getRelatedSets()

        # Recalculate the PKB using these sets of clues
        self.buildForBatch(clues)

        # Update the agent board after we have recalculated the PKB
        self.updateAgentBoardFromPKB()

        mins = []
        minP = 1

        # Get the minimum probability from the PKB
        for key in self.pkb:
            if key not in self.visited and self.pkb[key] < minP:
                minP = self.pkb[key]

        # Find all cells that have this minimum probability from the PKB
        if minP > 0 and minP < 1:
            for key in self.pkb:
                if key not in self.visited:
                    mins.append(key)
        else:
            for key in self.pkb:
                if key not in self.visited and self.pkb[key] == minP:
                    mins.append(key)
        
        nextCell = None

        #print("Min P: " + str(minP))
        # Add all cells with 0 probability to the moveList and return these
        # as the nextCells to be hit
        if minP == 0:
            mins = self.expandZeros(mins)
            self.moveList.extend(mins)
            nextCell = mins
        # If the minimum probability stays a 1, this means that the rest of 
        # the cells are flagged as mines and should not be queried and the 
        # game is over.  
        elif minP == 1:
            return "Game Over!"
        # If the minP is some number between 0 and 1, than we choose our nextMove
        # randomly. 
        else:
            self.risk = self.risk + 1
            nextCell = self.getMaxKnownCell(mins)
            nextCell = self.expandZeros([nextCell])
            self.moveList.extend(nextCell)
        return nextCell

    # If the next cell we click is a cell where the clue is a 0, we proceed
    # to open up all the surrounding neighbors and continue this cycle until 
    # we have exhausted all 0 clues in the surroundings. We use Breadth-First
    # Search to expand the 0 clues.
    def expandZeros(self, cells):
        moves = []
        dim = len(self.board)
        visited = [[False for p in range(dim)] for k in range(dim)]
        queue = deque()
        queue.extend(cells)

        for cell in cells:
            visited[cell[0]][cell[1]] = True
        
        while len(queue) != 0:
            node = queue.popleft()
            moves.append(node)

            if self.board[node[0]][node[1]] == 0:
                for n in self.getNeighbors(node):
                    if not visited[n[0]][n[1]]:
                        queue.append(n)
                        visited[n[0]][n[1]] = True
        return moves

    def getMaxKnownCell(self, cells):
        self.updateAllUnknownCells(cells)
        maxK = 0
        maxKnownCells = []

        # Get the maximum known cells
        for key in self.knownCells:
            if key not in self.visited and self.knownCells[key] > maxK:
                maxK = self.knownCells[key]

        for key in self.knownCells:
            if key not in self.visited and self.knownCells[key] == maxK:
                maxKnownCells.append(key)

        size = len(maxKnownCells)
        return maxKnownCells[random.randint(0, size - 1)]

    # Performs basic inference on the whole board to reduce the amount
    # of advanced inference needed.
    def basicInference(self):
        dim = len(self.board)
        flaggedMines = []
        safeCells = []

        for i in range(dim):
            for j in range(dim):
                if isinstance(self.agentBoard[i][j], int):
                    clue = self.agentBoard[i][j]
                    mineNeighbors = 0
                    safeNeighbors = 0
                    unknownNeighbors = []
                    totalNeighbors = 0

                    for n in self.getNeighbors((i,j)):
                        totalNeighbors = totalNeighbors + 1
                        if self.agentBoard[n[0]][n[1]] == "*" or self.agentBoard[n[0]][n[1]] == "F":
                            mineNeighbors = mineNeighbors + 1
                        elif isinstance(self.agentBoard[n[0]][n[1]], int):
                            safeNeighbors = safeNeighbors + 1
                        else:
                            unknownNeighbors.append(n)

                    if clue - mineNeighbors == len(unknownNeighbors):
                        flaggedMines.extend(unknownNeighbors)
                    elif totalNeighbors - clue - safeNeighbors == len(unknownNeighbors):
                        safeCells.extend(unknownNeighbors)

        for m in flaggedMines:
            self.pkb[m] = 1

        return self.expandZeros(safeCells), flaggedMines

    # Opens a given cell and calls functions to recalculate the PKB based on 
    # this new information. 
    def makeMove(self):
        while len(self.moveList) != 0:
            coord = self.moveList.pop(0)
            x = coord[0]
            y = coord[1]
            self.visited.add((x, y))
            self.agentBoard[x][y] = self.board[x][y]
            # If we open a mine, update the minesHit attribute and 
            # the PKB.
            if self.agentBoard[x][y] == "*":
                self.minesHit = self.minesHit + 1
                self.pkb[(x, y)] = 1
            # If we open a clue, update the PKB.
            else:
                self.pkb[(x, y)] = 0

    # Method to simplify solving the board and returns the score. 
    def solve(self):
        nextCells = self.nextMove()
        while nextCells != "Game Over!":
            self.makeMove()
            nextCells = self.nextMove()
        return self.getScore()

    # Helper method that checks if a row is all 0
    def allZero(self, row):
        for i in range(len(row)):
            if row[i] != 0:
                return False
        return True

    # Gets the score of the agent by returning totalMines - minesHit
    def getScore(self):
        return self.mines - self.minesHit

    # Get all valid neighbors
    def getNeighbors(self, coord):
        x = coord[0]
        y = coord[1]
        # Iterate through all possible neighbors
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1),
                    (x - 1, y - 1)]

        return [(i,j) for (i,j) in neighbors if self.checkPoint(i, j)]

    # Helper function to check if a certain point is between 0 and the graphs dimensions
    def checkPoint(self, x, y):
        if (0 <= x < len(self.board)) and (0 <= y < len(self.board)):
            return True
        return False
