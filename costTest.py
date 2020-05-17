import board
from advancedAgent import AdvancedAgent
from minCostAgent import MinCostAgent
from minRiskAgent import MinRiskAgent
from improvedAgent import ImprovedAgent

def main():
    dim = 10
    runs = 5
    mineStepSize = 10
    startMines = 10
    endMines = 100
    for mines in range(startMines, endMines, mineStepSize):
        advanced_cost = 0
        min_cost_cost = 0
        min_risk_cost = 0
        improved_cost = 0
        for i in range(runs):
            myBoard = board.generateBoard(dim, mines)
            #board.printMap(myBoard)
            #print("")
            advancedAgent = AdvancedAgent(myBoard, mines)
            minCostAgent = MinCostAgent(myBoard, mines)
            minRiskAgent = MinRiskAgent(myBoard, mines)
            improvedAgent = ImprovedAgent(myBoard, mines)
            advancedAgent.solve()
            minCostAgent.solve()
            minRiskAgent.solve()
            #improvedAgent.solve()
            advanced_cost = advanced_cost + advancedAgent.minesHit
            min_cost_cost = min_cost_cost + minCostAgent.minesHit
            min_risk_cost = min_risk_cost + minRiskAgent.minesHit
            improved_cost = improved_cost + improvedAgent.minesHit
        print("Advanced Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Cost: " + str(advanced_cost/runs))
        print("Min Cost Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Cost: " + str(min_cost_cost/runs))
        print("Min Risk Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Cost: " + str(min_risk_cost/runs))
        print("Improved Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Cost: " + str(improved_cost/runs))
        print("")

main()