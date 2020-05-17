import random
from collections import deque


class BasicAgent:

    # initialize the fields we will use
    # the board is the actual minesweeper board with all the data, it is only accessed to get values if a move is made
    # info map is a map from coordinate to the info we have about that cell see KnowledgePoint for more info
    # visited is a list of all visited cells
    # mines is a list of all mines discovered, this includes mines discovered safely or through exploding
    # the score is how many mines are identified safely, starts at the number of mines but is decremented when a bomb is exploded
    # changes contains lists of changes associated with a specific move, used for visualization
    # opencoords is used to generate random moves more efficiently
    def __init__(self, board, mines):
        self.board = board
        # visited is a map from coordinates to the info gotten from the coordinate
        self.infoMap = {}
        self.visited = []
        self.mines = []
        self.score = mines
        self.totalMines = mines
        self.changes = []
        self.opencoords = []

        dim = len(board)
        # for every coordinate create a new KnowledgePoint to keep track of a cells info
        # also add that coordinate to the list of possible moves
        for i in range(dim):
            for j in range(dim):
                coord = (i,j)
                self.infoMap[coord] = KnowledgePoint(len(self.__getNeighbors(coord)))
                self.opencoords.append(coord)

    # prints the board the agent sees
    def printBoard(self):
        dim = len(self.board)
        for i in range(dim):
            for j in range(dim):
                print(self.infoMap[(i,j)].clue, end=" ")
            print("")

    # Returns the board the agent sees
    def getBoard(self):
        dim = len(self.board)
        agentBoard = [["?" for i in range(dim)] for j in range(dim)]
        for i in range(dim):
            for j in range(dim):
                agentBoard[i][j] = self.infoMap[(i, j)].clue
        return agentBoard

    # Execute a given move and update the knowledge base
    def __makeMove(self, coord):
        x = coord[0]
        y = coord[1]

        # get the clue at the coordinate
        clue = self.board[x][y]
        # update the cells info with the clue
        self.infoMap[coord].clue = clue

        # create a new list of changes for the current move and append the picked cell as the first change
        changes = []
        changes.append(coord)

        # if the clue is a * then a bomb is exploded update the score and list of mines
        if clue == "*":
            self.score -= 1
            self.mines.append(coord)

        # else we found a clue and we must update the visited array and update the clues neighbors
        # append this coordinate to the list of current changes
        else:
            self.visited.append(coord)
            neighbors = self.__getNeighbors(coord)
            for neighbor in neighbors:
                self.infoMap[neighbor].hiddenNeighbors -= 1
                self.infoMap[neighbor].safeNeighbors += 1
                changes.append(neighbor)

        # add the current list of changes to the list of all changes
        self.changes.append(changes)
        # update the rest of the board based on the current change
        self.__propogateChanges(coord)

    # this method contains the logic to update the board based on a move
    def __propogateChanges(self, point):
        dim = len(self.board)

        # create a queue to keep track of cells that have changed and need to propagate their changes to their neighbors
        queue = deque()
        queue.append(point)
        # if the initial move exploded a bomb queue the neighbors of the mine to have changes propagated
        if self.infoMap[point].clue != "*":
            neighbors = self.__getNeighbors(point)
            for neighbor in neighbors:
                if neighbor in self.visited:
                    queue.append(neighbor)

        # keep applying changes until the queue is empty
        # this implementation takes a bfs like approach
        while(len(queue) != 0):
            coord = queue.popleft()
            x = coord[0]
            y = coord[1]

            # get the neighbors of the current coordinate
            neighbors = self.__getNeighbors(coord)

            clue = self.infoMap[coord].clue

            # If the clue at a given spot is a bomb we should update all neighbors to receive this knowledge
            # and then queue all those neighbors to propagate their changes
            if clue == "*" or clue == "F":
                if coord in self.visited:
                    continue
                self.visited.append(coord)
                for neighbor in neighbors:
                    if neighbor not in self.changes[-1]:
                        self.changes[-1].append(neighbor)
                    self.infoMap[neighbor].hiddenNeighbors -= 1
                    self.infoMap[neighbor].minesFound += 1
                    info = self.infoMap[neighbor]
                    queue.append(neighbor)

            # Else if it is not hidden and we have info on the cell try to get more information
            elif clue != "?":
                info = self.infoMap[coord]

                # If there are no hiddenNeighbors then no additional info can be received
                if info.hiddenNeighbors == 0:
                    continue

                # If all hiddenNeighbors are determined to be bombs, mark them as such and propagate those changes to
                # all neighbors
                if (clue - info.minesFound) == info.hiddenNeighbors:
                    #print("all bombs")
                    for neighbor in neighbors:
                        if neighbor not in self.visited:
                            if neighbor not in self.changes[-1]:
                                self.changes[-1].append(neighbor)
                            self.infoMap[neighbor].clue = "F"
                            if neighbor not in self.mines:
                                self.mines.append(neighbor)
                            queue.append(neighbor)
                            info = self.infoMap[neighbor]

                # If all hiddenNeighbors are determined to be safe mark them as such and do not propagate as there
                # are no changes to propagate
                elif (len(neighbors) - clue - info.safeNeighbors) == info.hiddenNeighbors:
                    for neighbor in neighbors:
                        if neighbor not in self.visited:
                            self.infoMap[neighbor].isSafe = True
                            info = self.infoMap[neighbor]

    # If there is a safe move execute it, else choose a move at random
    def nextMove(self):
        # If all the bombs have been found then the game is over
        if len(self.mines) == self.totalMines:
            return "Game Over"

        # Check if there is a safe move if so execute that move
        for coord, knowledge in self.infoMap.items():
            if knowledge.isSafe and not (coord in self.visited):
                self.__makeMove(coord)
                return "safe move"
            else:
                continue

        # if there is no logical next step choose a point at random
        coord = self.__randomPoint()
        while coord in self.visited:
            coord = self.__randomPoint()

        self.__makeMove(coord)
        return "random move"

    def solve(self):
        while self.nextMove() != "Game Over":
            pass
        return self.score

    # Get a random point
    def __randomPoint(self):
        index = random.randint(0, len(self.opencoords) - 1)
        return self.opencoords.pop(index)

    # Get all valid neighbors
    def __getNeighbors(self, coord):
        x = coord[0]
        y = coord[1]
        # Iterate through all possible neighbors
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1),
                     (x - 1, y - 1)]

        validNeighbors = [(i,j) for (i,j) in neighbors if self.__checkPoint(i, j)]

        return validNeighbors

    # Helper function to check if a certain point is between 0 and the graphs dimensions
    def __checkPoint(self, x, y):
        if (0 <= x < len(self.board)) and (0 <= y < len(self.board)):
            return True
        return False

    # Returns a list of the most recent changes
    def getLatestChanges(self):
        if(len(self.changes) != 0):
            return self.changes[-1]
        return []

# Class to model the knowledge base
class KnowledgePoint:

    def __init__(self, neighbors):
        self.clue = "?"
        self.safeNeighbors = 0
        self.hiddenNeighbors = neighbors
        self.minesFound = 0
        self.isSafe = False
