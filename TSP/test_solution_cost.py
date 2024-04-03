from tsp import *
import math

if __name__ == '__main__':
    edge_with_weights = [(0, 1, 1), (0, 2, 1.41), (0, 3, 2.23), (1, 2, 1), (1, 3, 1.41), (2, 3, 1)]
    A = 1213
    B = 1213
    tsp = TSP(edge_with_weights, A, B, show_graph=False, save_graph=True)
    print(tsp.weight)
    n = len(tsp.weight) - 1
    fx = tsp.get_pair_coeff_var()
    solutions = generate_binary_string(length=9)
    # print(len(solutions))

    solution_cost = dict()
    for solution in solutions:
        cost = calculate_cost(fx, solution)
        solution_cost[solution] = round(cost, 2)

    solution_cost_sorted = sorted(solution_cost.items(), key=lambda item: item[1])

    for i in solution_cost_sorted[:math.factorial(n)]:
        print(i[0], i[1])