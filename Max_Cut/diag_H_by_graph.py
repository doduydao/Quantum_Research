import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def diag_Zs(no_qubit):
    Z = np.array([[1, 0], [0, -1]])
    Id = np.eye(2)
    Zs = []
    for j in range(no_qubit):
        Z_i = None
        for i in range(no_qubit):
            if i == j:
                Z_i = np.kron(Z_i, Z) if Z_i is not None else Z
            else:
                Z_i = np.kron(Z_i, Id) if Z_i is not None else Id
        Zs.append(Z_i)

    diag_Zs = [np.diag(Z_i) for Z_i in Zs]
    return diag_Zs

def diag_H(G):
    nodes = G.nodes
    edges = G.edges

    no_qubit = len(nodes)
    Zs = diag_Zs(no_qubit)
    print(len(Zs))
    H = None
    H_z = np.array([1.0]* (2**no_qubit)) * 1/2 * len(edges)

    for edge in edges:
        i = edge[0]
        j = edge[1]
        diag_Z_i = Zs[i]
        diag_Z_j = Zs[j]

        if H is None:
            H = diag_Z_i * diag_Z_j
        else:
            H += diag_Z_i * diag_Z_j
    return H_z - 1/2 * H


def make_solution(diag_H):
    num_qubits = int(len(diag_H) ** 0.5)

    for i, value in enumerate(diag_H):
        binary_state = format(i, f'0{num_qubits}b')
        print(f"|{binary_state}>: {int(value)}")



if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4)])
    nx.draw(G, with_labels=True, alpha=0.8, node_size=1000, font_size=20)
    plt.savefig("Graph.png", format="PNG")
    H = diag_H(G)
    print(H)
    make_solution(H)