import board
from advancedAgent import AdvancedAgent
from minRiskAgent import MinRiskAgent
from minCostAgent import MinCostAgent
from improvedAgent import ImprovedAgent
from collections import deque

def main():
    dim = 10
    mines = 30
    b = board.generateBoard(dim, mines)

    board.printMap(b)
    print("")
    advanced = ImprovedAgent(b, mines)
    board.printMap(advanced.agentBoard)
    move = advanced.nextMove()
    while move != "Game Over!":
        print("Move:" + str(move))
        input("Click enter to make the next move!")
        advanced.makeMove()
        board.printMap(b)
        print("")
        board.printMap(advanced.agentBoard)
        move = advanced.nextMove()
    print("Score: " + str(advanced.getScore()))
    print("Cost: " + str(advanced.minesHit))
    print("Risk: " + str(advanced.risk))

main()