import matplotlib.pyplot as plt
import pandas as pd


def calculate_no_conflict(state, n, edges):
    colors = list()
    step = int(len(state) / n)
    for i in range(0, len(state), step):
        colors.append(state[i:i + step])

    no_conflict = 0
    for i in range(len(colors)):
        for j in range(len(colors)):
            if i != j and ((i, j) in edges or (j, i) in edges):
                for k in range(step):
                    if colors[i][k] == colors[j][k] and int(colors[j][k]) == 1:
                        no_conflict += 1
    return no_conflict

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
            no_color_used += 1

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
        tmp = abs(tmp - 1)
        no_conflit += tmp
    return no_conflit


def calculate_cost(state, B, C, no_nodes, egdes):
    no_conflict = calculate_no_conflict(state, no_nodes, egdes)
    no_colors = count_color(state, no_nodes)
    penalty = penalty_part(state, no_nodes)
    return B * penalty + C * no_conflict + no_colors


def evaluate_H(solutions, B, C, no_nodes, egdes) -> float:
    """
    Estimate the cost of solutions, that are made by H
    """
    energy = 0
    total = 0

    for state, value in solutions.items():
        cost = calculate_cost(state, B, C, no_nodes, egdes)
        energy += cost * value
        total += value
    return energy / total

def compare_cost_by_iter(solution_iters, no_nodes, edges, B, C):
    info = []
    for i in range(len(solution_iters)):
        states = solution_iters[i]
        iter = states[0]
        states = states[1]
        distribution_cost = dict()
        no_shots = sum(list(states.values()))
        for state, shot in states.items():
            cost_by_state = calculate_cost(state, B, C, no_nodes, edges)
            # prob = shot / no_shots * 100

            if cost_by_state not in distribution_cost:
                distribution_cost[cost_by_state] = shot
            else:
                distribution_cost[cost_by_state] += shot

        distribution_cost = {i[0]: round(i[1], 2) for i in sorted(distribution_cost.items(), key=lambda item: item[0])}

        file_name = "iter_" + str(iter) + ".txt"
        with open(file_name, 'w') as file:
            for k, v in distribution_cost.items():
                line = str(k) + "\t" + str(v) + "\n"
                file.write(line)

        info.append([iter, distribution_cost])
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
            cumulative_probs.append(cumulative_probs[-1] + p)
        cumulative_data[k] = cumulative_probs
    return cumulative_data


def has_K_color(state, K, edges, n):
    no_conflict = calculate_no_conflict(state, n, edges)
    no_colors = count_color(state, n)

    if no_conflict == 0 and no_colors == K:
        return True
    else:
        return False


def is_solution(state, K, n):
    colors = list()
    step = int(len(state) / n)
    for i in range(0, len(state), step):
        colors.append(state[i:i + step])

    for color in colors:
        tmp = []
        for i in range(K):
            tmp.append(int(color[i]))
        if sum(tmp) != 1:
            return False
    return True
    # return False


def create_chart(name, results, is_export_data=True):
    # results = [[iter, energy, distribution_no_colors, distribution_cost],..]
    data_std_cost = dict()
    data_estimation_cost = dict()
    iters = []
    for result_iter in results:
        iter = result_iter[0]
        iters.append(iter)
        cost_shot = result_iter[1]
        costs = list(cost_shot.keys())
        shots = list(cost_shot.values())
        no_shots = sum(shots)

        data_estimation_cost['cost'] = costs
        data_estimation_cost["iter " + str(iter)] = [round(i / no_shots * 100, 2) for i in shots]
        mean = no_shots / len(costs)
        
        std = [(x - mean) ** 2 / no_shots for x in shots]
        data_std_cost['cost'] = costs
        data_std_cost["std " + str(iter)] = std

    cumulative_std_cost = calculate_cumulative_prob(data_std_cost)
    cumulative_estimation_cost = calculate_cumulative_prob(data_estimation_cost)

    if is_export_data:
        df = pd.DataFrame.from_dict(data_std_cost)
        df.to_csv(name + '/result_std_cost.csv', index=False, sep='\t')
        print(name + "/result_std_cost.csv file created successfully!")

        df = pd.DataFrame.from_dict(data_estimation_cost)
        df.to_csv(name + '/result_distribution_cost.csv', index=False, sep='\t')
        print(name + "/result_distribution_cost.csv file created successfully!")

        df = pd.DataFrame.from_dict(cumulative_std_cost)
        df.to_csv(name + '/result_cumulative_std_cost.csv', index=False, sep='\t')
        print(name + "/result_cumulative_std_cost.csv file created successfully!")

        df = pd.DataFrame.from_dict(cumulative_estimation_cost)
        df.to_csv(name + '/result_cumulative_estimation_cost.csv', index=False, sep='\t')
        print(name + "/result_cumulative_estimation_cost.csv file created successfully!")

    plt.clf()
    plt.figure(figsize=(8, 6))  # Set chart size
    keys = list(data_std_cost.keys())[1:]
    df = pd.DataFrame(data_std_cost)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['cost'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['cost'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Standard Deviation of cost')
    plt.xlabel('Cost')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(name + '/Standard Deviation of cost.PNG')

    plt.clf()
    plt.figure(figsize=(8, 6))  # Set chart size
    keys = list(data_estimation_cost.keys())[1:]
    df = pd.DataFrame(data_estimation_cost)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['cost'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['cost'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Probability of cost')
    plt.xlabel('Cost')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    plt.savefig(name + '/Probability of cost.PNG')

    # cummulative proba
    plt.clf()
    plt.figure(figsize=(8, 6))  # Set chart size
    keys = list(cumulative_std_cost.keys())[1:]
    df = pd.DataFrame(cumulative_std_cost)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['cost'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['cost'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Cumulative standard deviation of cost')
    plt.xlabel('Cost')
    plt.ylabel('Cumulative standard deviation')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(name + '/Cumulative standard deviation of cost.PNG')

    plt.clf()
    plt.figure(figsize=(8, 6))  # Set chart size
    keys = list(cumulative_estimation_cost.keys())[1:]
    df = pd.DataFrame(cumulative_estimation_cost)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['cost'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['cost'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Cumulative probability of cost')
    plt.xlabel('Cost')
    plt.ylabel('Cumulative probability')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(name + '/Cumulative probability of cost.PNG')
