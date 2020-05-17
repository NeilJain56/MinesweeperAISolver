import board
from advancedAgent import AdvancedAgent
from minCostAgent import MinCostAgent
from minRiskAgent import MinRiskAgent

def main():
    dim = 10
    runs = 5
    mineStepSize = 10
    startMines = 10
    endMines = 50
    for mines in range(startMines, endMines, mineStepSize):
        infer_score = 0
        min_cost_score = 0
        min_risk_score = 0
        for i in range(runs):
            myBoard = board.generateBoard(dim, mines)
            inferAgent = AdvancedAgent(myBoard, mines)
            minCostAgent = MinCostAgent(myBoard, mines)
            infer_score = infer_score + inferAgent.solve()
            min_cost_score = min_cost_score + minCostAgent.solve()
        print("Inference Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Score: " + str(infer_score/runs))
        print("Min Cost Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Score: " + str(min_cost_score/runs))
        print("Min Risk Algorithm at Mine Density: " + str(mines/(dim*dim)))
        print("Avg Score: " + str(min_cost_score/runs))
        print("")

main()