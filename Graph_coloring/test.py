from hamiltonian import *

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
    print(no_conflit)
    return no_conflit



if __name__ == '__main__':
    # make a graph
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 1000
    no_shots = 2048
    T = 100

    gcp = Graph_Coloring(edges,K=K, A=A, node_size=500, show_graph=False, save_graph=True)

    # print("Cost function:")
    # prettyprint(gcp.get_cost_function())
    #
    # print("Hamiltonian simplification:")
    # prettyprint(gcp.simply_Hamiltonian())

    state = '000111011011'
    conflict = calculate_no_conflict(state, len(gcp.nodes), gcp.edges)
    print(conflict, A * penalty_part(state, len(gcp.nodes)))
    print(calculate_cost(gcp.get_pair_coeff_var(), state))