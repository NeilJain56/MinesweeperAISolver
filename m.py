import numpy as np
from timeit import default_timer as timer

# This function reduces a given matrix to row echelon form
def row_echelon(A):
    # r is the num of rows and c is the num of cols
    r, c = A.shape

    # Start at the first row and first col
    row = 0
    col = row

    # While we still have rows to check keep looping
    while row < r:
        # Done checks if we have found our next row to apply the reduction
        done = False
        for j in range(row, c):
            if not done:
                for i in range(row, r):
                    # Check if the element at question is non zero
                    if A[i, j] != 0:
                        done = True
                        # Set the col to the one we just found so we have a reference
                        col = j
                        # If the col with the next left most non zero value is not the 
                        # current row then swap those rows
                        if i != row:
                            temp = A[i].copy()
                            A[i] = A[row].copy()
                            A[row] = temp
                            break
                        break
            else:
                break

        # If we did not find a left most non zero col then we do not want to 
        # change any rows and should continue instead
        if A[row, col] != 0:
            A[row] = A[row] / A[row, col]
            A[row+1:] -= A[row] * A[row+1:, col:col+1]
        row = row + 1

    # Return the row reduced matrix
    return A

# Gets the first non-zero number in a row 
# Returns false if all zeros in the row
def firstNum(row):
    r = len(row) - 1
    for i in range(r):
        if row[i] != 0:
            return i, True
    return False, False

# Gets all possible valid variable assignments 
def get_solutions(A):

    # Put matrix A into row_echelon form
    A = row_echelon(A)

    # If there are any -1's as the first element in a row,
    # flip the sign of every element in the row. 
    # This simplifies evaluating a variable assignment.
    A = remove_negatives(A)

    #print("ROW")
    #print(np.array(A))
    
    # Use topological sorting to get the order in which 
    # variables should be evaluated. Free variables are variables
    # that are not dependent on any other variable.
    free_variables, indegrees, edges = topological_sort(A)
    
    #print("INDEGREES")
    #print(indegrees)

    #print("EDGES")
    #print(edges)

    # To increase effeciency, if there are any free variables 
    # that have rows in the matrix than just automatically assign 
    # them the value of the number all the way to the right. 
    free_var_assignment = default_free_assignment(A)
    #print("FREE ASSIGN")
    #print(free_var_assignment)

    # Remove the free variables that are already assigned to reduce the number of 
    # actual free variable and increase effeciency. 
    free_variables = getActualFree(free_variables, free_var_assignment)
    free_variables = sort_free(free_variables)
    #print("FREE")
    #print(free_variables)
    #print("FREE VARS: " + str(len(free_variables)))
    if len(free_variables) > 15:
        return []
    
    start = timer()
    validAssignments = backtracking(free_variables, A, free_var_assignment, indegrees, edges)
    end = timer()
    #print("BACKTRACK TIME: " + str(end - start))
    return validAssignments

# When a free variable column has a 1 in a row where all other columns are
# 0, then we can assign that free variable the value all the way to the right.
def default_free_assignment(A):
    cols = len(A[0])
    rows = len(A)
    assign = {}
    for r in range(rows):
        j, valid = firstNum(A[r])
        if valid == False:
            break

        if allZero(A[r][j+1:cols-1]):
            assign[j] = A[r][cols-1]
    return assign

# Remove the already assigned free variables to reduce the number of free variables
# that need to be assigned.
def getActualFree(free_vars, free_var_assignment):
    free_vars = set(free_vars)
    for f in free_var_assignment:
        free_vars.remove(f)
    return list(free_vars)

# Gets probability for each cell being a mine
def get_probabilities(A):

    # Get all possible valid assignments for matrix A
    validAssignments = get_solutions(A)
    total_assignments = len(validAssignments)
    probabilities = {}

    # This for loop calculates probabilities[key] = sum(assignment[key]) in every
    # valid assignment.
    for assignment in validAssignments:
        if assignment != None:
            for key in assignment:
                if key not in probabilities:
                    probabilities[key] = 0
                if assignment[key] == 1:
                    probabilities[key] = probabilities[key] + 1
    
    # Divide each value by the total assignments, to get the probability
    # that a cell is a mine.
    for key in probabilities:
        probabilities[key] = float(probabilities[key])/float(total_assignments)
    return probabilities

# Evaluates an assignment of all variables by using free variable assignment
# and returns the full assignment if it is valid
def evaluate_assignment(free_vars, free_assignment, A, ordering, free_var_assignment):
    cols = len(A[0])

    # Copy the free variable default assignment 
    assignment = free_var_assignment.copy()

    # Copy the free variable assignment to the assignment dict
    for f in range(len(free_vars)):
        assignment[free_vars[f]] = free_assignment[f]

    # Remove all the free variables from the ordering array
    ordering = ordering[len(assignment):]
    
    # Evaluate the ordering in order
    for t in ordering:

        # Find the row with the Tth column as the first 1 in the row
        realRow = find_row(A, t)
        sum_vars = 0

        # Iterate through the row and add to sum_vars by multiplying 
        # the assignment by the value in the matrix
        for i in range(t+1, cols-1):
            if A[realRow][i] != 0:
                sum_vars = sum_vars + A[realRow][i]*assignment[i]

        # If at any point, an assignment has a value that is not 0 and not 1
        # we return False because this is an invalid assignment.
        if A[realRow][cols-1] - sum_vars != 0 and A[realRow][cols-1] - sum_vars != 1:
            return False

        assignment[t] = A[realRow][cols-1] - sum_vars

    return assignment

def sort_free(free_vars):
    return sorted(free_vars, reverse=True)

def backtracking(free_vars, A, free_var_assignment, indegrees, edges):
    validAssignments = []
    totalT = 0
    # Copy the free variable default assignment 
    assignment = free_var_assignment.copy()

    if len(free_vars) == 0:
        #print("NO_------------------------------------------")
        return [evaluate_with_no_free_vars(assignment, edges, indegrees, A)]

    queue = [[assignment, "0", 0], [assignment, "1", 0]]
    inDict = {}
    inDict[0] = indegrees
    #print("FREE VARS")
    #print(free_vars)

    while len(queue) != 0:
        node = queue.pop(0)
        #print("NODE: " + str(node))
        temp_assignment = node[0].copy()
        num = node[1]
        level = node[2]
        
        #print("Level: " + str(level))
        #print("Length of Num: " + str(len(num)))
        
        for i in range(len(num)):
            temp_assignment[free_vars[i]] = int(num[i])

        #print("BEFORE: " + str(temp_assignment))
        s = timer()
        temp_assignment = evaluate_partial(temp_assignment.copy(), edges, inDict[level].copy(), A, free_vars[level])
        e = timer()
        totalT = totalT + (e - s)

        #print("AFTER: " + str(temp_assignment))
        if temp_assignment != None:
            t_assign = temp_assignment[0].copy()
            
            if level == len(free_vars) - 1:
                validAssignments.append(t_assign)

            ins = temp_assignment[1].copy()
            inDict[level + 1] = ins

            zero = num + "0"
            one = num + "1"

            if level < len(free_vars) - 1:
                queue.append([t_assign.copy(), zero, level + 1])
                queue.append([t_assign.copy(), one, level + 1])

    #print("NUM OF VALID: " + str(len(validAssignments)))
    #print("EVAL TIME: " + str(totalT))
    return validAssignments

def evaluate_with_no_free_vars(assignment, edges, indegrees, A):
    queue = []
    queue.extend(list(assignment.keys()))
    finalAssignment = assignment.copy()

    while len(queue) != 0:
        node = queue.pop(0)
        if node not in assignment:
            val = calc(A, node, finalAssignment)
            if val != 0 and val != 1:
                return None

            finalAssignment[node] = val

        for n in edges[node]:
            indegrees[n] = indegrees[n] - 1
            # Only enqueue when indegrees is 0
            if indegrees[n] == 0 and n not in assignment:
                queue.append(n)

    return finalAssignment

def evaluate_partial(assignment, edges, indegrees, A, newVar):
    queue = [newVar]
    finalAssignment = assignment.copy()
    
    #print("NEW")
    while len(queue) != 0:
        #print("QUEUE")
        #print(queue)
        node = queue.pop(0)
        if node not in assignment:
            val = calc(A, node, finalAssignment)
            if val != 0 and val != 1:
                return None

            finalAssignment[node] = val

        for n in edges[node]:
            indegrees[n] = indegrees[n] - 1
            # Only enqueue when indegrees is 0
            if indegrees[n] == 0 and n not in assignment:
                queue.append(n)
                
    #print("FINAL: " + str(finalAssignment))
    return finalAssignment, indegrees

def calc(A, t, assignment):
    cols = len(A[0])

    # Find the row with the Tth column as the first 1 in the row
    realRow = find_row(A, t)
    sum_vars = 0

    # Iterate through the row and add to sum_vars by multiplying 
    # the assignment by the value in the matrix
    for i in range(t+1, cols-1):
        if A[realRow][i] != 0:
            sum_vars = sum_vars + A[realRow][i]*assignment[i]

    return A[realRow][cols-1] - sum_vars

# Returns binary numbers from 0 to 2^n where n is the number 
# of free variables to test all possible combinations    
def possible_assignments(free_vars):
    num_of_vars = len(free_vars)
    #print("Free Vars: " + str(num_of_vars))
    result = []
    for i in range(0, 2**num_of_vars):
        binary = bin(i)[2:]
        paddedBin = "0" * (num_of_vars - len(binary)) + binary
        result.append([int(paddedBin[i]) for i in range(len(paddedBin))])

    return result

# Gives order of variables that should be evaluated based off of 
# row echelon form of array A
def topological_sort(A):
    edges = {}
    rows = len(A)
    col = len(A[0])

    # Initialize edges for every column except for the last one 
    # because the last one has right hand side of equations and
    # does not represent a variable.
    for i in range(col - 1):
        edges[i] = set()
    
    # For every row, add an edge from t to every column j when 
    # j has a non-zero value in it. This means that the value of t is
    # dependent on j.
    for i in range(rows):
        j, valid = firstNum(A[i])
        if valid == False:
            break

        for t in range(j+1, col - 1):
            if A[i][t] != 0:
                edges[t].add(j)
    
    # Initialize indegrees for topological sort
    indegrees = {}
    for i in range(col - 1):
        indegrees[i] = 0

    # Add 1 to indegree, for every edge that goes into e.
    for key in edges: 
        for e in edges[key]:
            indegrees[e] = indegrees[e] + 1
    

    # All variables with 0 indegrees are free variables.
    free_vars = [i for i in indegrees if indegrees[i] == 0 and check_col(A, i)]
    
    return free_vars, indegrees, edges

# Makes all leading -1's positive by flipping every value in the row
def remove_negatives(A):
    rows = len(A)
    col = len(A[0])
    for i in range(rows):
        j, valid = firstNum(A[i])
        if valid == False:
            break
        
        if A[i][j] == -1:
            A[i] = [A[i][k]*-1 for k in range(j, col)]
    return A

# Finds a row with a 1 in the Tth column as the first 1 in the row
def find_row(A, t):
    for i in range(len(A)):
        if A[i][t] == 1 and allZero(A[i][0:t]):
            return i

# Checks if a certain column is all 0 which is used to see if a 
# variable is a free variable in conjunction with if it has no 
# dependencies
def check_col(A, j):
    for i in range(len(A)):
        if A[i][j] != 0:
            return True
    return False
 
# Returns true if every element in the row is a zero
def allZero(row):
    for i in range(len(row)):
        if row[i] != 0:
            return False
    return True
