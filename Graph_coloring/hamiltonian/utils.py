import matplotlib.pyplot as plt
from .graph_coloring import calculate_cost


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
    total = 0
    for state, value in solutions.items():
        cost = calculate_cost(fx, state)
        # print(state, count, cost)
        energy += cost * value[1]
        total += value[1]
    return energy / total

def count_color(state, n):
    colors = set()
    step = int(len(state) / n)
    for i in range(0, len(state), step):
        colors.add(state[i:i + step])
    return len(colors)

def compare_cost_by_iter(solution_iters, fx, no_nodes):
    info = []
    energys = []
    for i in range(len(solution_iters)):
        states = solution_iters[i]
        new_states = dict()
        for state, shot in states.items():
            no_color = count_color(state, no_nodes)
            new_states[state] = [no_color, shot]

        states = {i[0]:i[1] for i in sorted(new_states.items(), key=lambda item: item[1][1], reverse=True)}

        energy = round(evaluate_H(fx, states),2)

        print("iter:", i, 'energy:', energy, 'states:', states)
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

def inversion_affichage(counts) -> dict:
    return {k[::-1]: v for k, v in counts.items()}
