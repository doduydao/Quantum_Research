from .tsp import calculate_cost

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

def compare_cost_by_iter(solution_iters, fx):
    info = []
    for i in range(len(solution_iters)):
        states = solution_iters[i]
        iter = states[0]
        states = states[1]

        distribution_cost = dict()
        no_shots = sum(list(states.values()))
        energy = 0
        for state, shot in states.items():
            cost_by_state = round(calculate_cost(fx, state), 2)
            prob = shot / no_shots * 100
            energy += cost_by_state * prob


            if cost_by_state not in distribution_cost:
                distribution_cost[cost_by_state] = prob
            else:
                distribution_cost[cost_by_state] += prob

        distribution_cost = {i[0]: round(i[1],2) for i in sorted(distribution_cost.items(), key=lambda item: item[0])}
        info.append([iter, round(energy, 2), distribution_cost])
    # info = sorted(info, key=lambda item: item[1])
    return info

def inversion_affichage(counts) -> dict:
    return {k[::-1]: v for k, v in counts.items()}


def calculate_cumulative_prob(data):
    cumulative_data = dict()
    keys = list(data.keys())
    cumulative_data[keys[0]] = data[keys[0]]
    for k in keys[1:]:
        probs = data[k]
        cumulative_probs = [probs[0]]
        for p in probs[1:]:
            cumulative_probs.append(cumulative_probs[-1]+p)
        cumulative_data[k] = cumulative_probs
    return cumulative_data