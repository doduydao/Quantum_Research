def calculate_no_conflict(state, n, edges):
    colors = list()
    step = int(len(state) / n)
    for i in range(0, len(state), step):
        colors.append(state[i:i + step])

    no_conflict = 0
    for i in range(len(colors)):
        for j in range(len(colors)):
            if i != j and ((i, j) in edges or (j, i) in edges) :
                for k in range(step):
                    if colors[i][k] == colors[j][k] and int(colors[j][k])==1:
                        no_conflict+=1
    return no_conflict


def calculate_cost(fx, solution):
    xs = [int(char) for char in solution]
    cost = 0
    for e in fx[:-1]:
        w = e[0]
        x_ip = xs[e[1][0]]
        for i in range(1, len(e[1])):
            x_ip *= xs[e[1][i]]
        cost += w * x_ip
    cost += fx[-1]
    return cost



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
    tmp = 0
    for i in range(0, len(state), step):
        tmp += int(state[i:i + step])
        colors.add(int(state[i:i + step]))
    # print(colors)

    no_color_used = 0

    tmp = str(tmp)
    for i in tmp:
        if int(i) != 0:
            no_color_used+=1

    return no_color_used

def penalty_part(state, n):
    colors = list()
    step = int(len(state) / n)
    for i in range(0, len(state), step):
        colors.append(state[i:i + step])

    no_conflit = 0

    for color in colors:
        tmp = 0
        for i in color:
            tmp += int(i)
        tmp = abs(tmp -1)
        no_conflit+=tmp
    return no_conflit


def compare_cost_by_iter(solution_iters, fx, no_nodes, egdes, P, C):
    # print(fx)

    info = []
    for i in range(len(solution_iters)):
        states = solution_iters[i]
        iter = states[0]
        states = states[1]
        distribution_no_colors = dict()
        distribution_cost = dict()
        no_shots = sum(list(states.values()))
        energy = 0
        for state, shot in states.items():
            no_conflict = calculate_no_conflict(state, no_nodes, egdes)
            # cost_by_state_2 = calculate_cost(fx, state)

            no_colors = count_color(state, no_nodes)

            cost_by_state = P * penalty_part(state, no_nodes) + C * no_conflict + no_colors
            # penatly_part =

            prob = shot / no_shots * 100
            # print("state:", state, cost_by_state, no_conflict, prob)

            energy += cost_by_state * prob

            if no_colors not in distribution_no_colors:
                distribution_no_colors[no_colors] = prob
            else:
                distribution_no_colors[no_colors] += prob
            if cost_by_state not in distribution_cost:
                distribution_cost[cost_by_state] = prob
            else:
                distribution_cost[cost_by_state] += prob

        distribution_no_colors = {i[0]: round(i[1],2) for i in sorted(distribution_no_colors.items(), key=lambda item: item[0])}
        distribution_cost = {i[0]: round(i[1],2) for i in sorted(distribution_cost.items(), key=lambda item: item[0])}
        info.append([iter, round(energy, 2), distribution_no_colors, distribution_cost])
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



