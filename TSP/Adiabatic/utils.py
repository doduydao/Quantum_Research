import matplotlib.pyplot as plt
from tsp import calculate_cost


def evaluate_H(fx, solutions) -> float:
    """
    Estimate the cost of solutions, that are made by H

    :param fx: cost fucntion
    :type fx: list
    :param solutions: all states and counted corresponding
    :type solutions: dictionary
    :return: Expectation of cost
    :rtype: float
    """
    energy = 0
    for state, count in solutions.items():
        cost = calculate_cost(fx, state)
        energy += cost * count
    total = sum(list(solutions.values()))
    return energy / total


def compare_cost_by_iter(solution_iters, fx):
    info = []
    energys = []
    for i in range(len(solution_iters)):
        states = solution_iters[i]
        energy = round(evaluate_H(fx, states),2)
        print("iter:", i, 'energy:', energy)
        energys.append(energy)
        info.append([i, energy, states])

    x = range(len(energys))
    plt.clf()
    plt.plot(x, energys)
    # Loop through data and add text labels with a small offset
    # for i, v in enumerate(energys):
    #     plt.text(energys[i], v + 0.2, f"{v}", ha="center")  # ha for horizontal alignment
    plt.xlabel("Iters")
    plt.ylabel("Estimation of average cost")
    # plt.legend()

    plt.savefig("line_plot.png")
    # plt.show()
    best_result = sorted(info, key=lambda item: item[1])[0]
    return best_result