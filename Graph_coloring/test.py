from hamiltonian import *


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


def has_K_color(state, K, edges, n):
    # no_conflict = calculate_no_conflict(state, n, edges)
    no_colors = count_color(state, n)

    # if no_conflict == 0 and no_colors == K:
    if no_colors == K:
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
    no_conflict = calculate_no_conflict(state, n, edges)
    if no_conflict != 0:
        return False
    return True
    # return False


if __name__ == '__main__':
    # make a graph
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 1000
    P = A
    C = 10
    no_shots = 2048
    T = 100

    gcp = Graph_Coloring(edges, K=K, A=A, node_size=500, show_graph=False, save_graph=True)

    states = generate_binary_string(len(gcp.nodes) * K)
    print(f"Probability of 1 state is {1 / len(states)*100:.2f}%")

    no_state_has_K_colors = 0

    states_has_K_colors = []
    for state in states:
        if has_K_color(state, 3, edges, len(gcp.nodes)):
            no_state_has_K_colors += 1
            # print(state)
            states_has_K_colors.append(state)
    prob_3_color = no_state_has_K_colors / len(states)
    print("no_state_has_K_colors", no_state_has_K_colors)
    print(f"Probability of state has 3 color is {prob_3_color * 100:.2f}% in all states")

    print()
    no_state_is_solution = 0
    for state in states_has_K_colors:
        if is_solution(state, K, len(gcp.nodes)):
            no_state_is_solution += 1

            cost_by_state = (P * penalty_part(state, len(gcp.nodes))
                             + C * calculate_no_conflict(state, len(gcp.nodes), edges)
                             + count_color(state, len(gcp.nodes))
                             )
            print("state:", state, "cost:", cost_by_state)
    prob_is_solution = no_state_is_solution / len(states)
    print("no_state_is_solution:", no_state_is_solution)
    print(f"Probability of 1 solution is {prob_is_solution * 100:.2f}% in all states")
    print(f"Probability of 1 solution is {no_state_is_solution/no_state_has_K_colors * 100:.2f}% in all states have 3 colors")

    sample = 5000
    print(prob_is_solution * sample)

    # print(has_K_color('111000000000', 3, edges,len(gcp.nodes)))

    # print(6*1/len(states))

    # print(len(states))

    # print("Cost function:")
    # prettyprint(gcp.get_cost_function())
    #
    # print("Hamiltonian simplification:")
    # prettyprint(gcp.simply_Hamiltonian())

    # state = '000111011011'
    # is_solution(state, K)

    # conflict = calculate_no_conflict(state, len(gcp.nodes), gcp.edges)
    # print(conflict, A * penalty_part(state, len(gcp.nodes)))
    # print(calculate_cost(gcp.get_pair_coeff_var(), state))
    # print(72/2593*100)