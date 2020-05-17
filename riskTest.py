import board
from advancedAgent import AdvancedAgent
from minCostAgent import MinCostAgent
from minRiskAgent import MinRiskAgent

def main():
    dim = 10
    runs = 5
    mineStepSize = 10
    startMines = 10
    endMines = 100
    for mines in range(startMines, endMines, mineStepSize):
        advanced_risk = 0
        min_cost_risk = 0
        min_risk_risk = 0
        for i in range(runs):
            myBoard = board.generateBoard(dim, mines)
            advancedAgent = AdvancedAgent(myBoard, mines)
            minCostAgent = MinCostAgent(myBoard, mines)
            minRiskAgent = MinRiskAgent(myBoard, mines)
            advancedAgent.solve()
            minCostAgent.solve()
            minRiskAgent.solve()
            advanced_risk = advanced_risk + advancedAgent.risk
            min_cost_risk = min_cost_risk + minCostAgent.risk
            min_risk_risk = min_risk_risk + minRiskAgent.risk
        print("Advanced Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Risk: " + str(advanced_risk/runs))
        print("Min Cost Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Risk: " + str(min_cost_risk/runs))
        print("Min Risk Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Risk: " + str(min_risk_risk/runs))
        print("")

main()