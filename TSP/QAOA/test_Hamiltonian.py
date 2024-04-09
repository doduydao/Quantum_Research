from tsp2 import TSP
if __name__ == '__main__':

    # edge_with_weights = [(0, 1, 1), (0, 2, 1.41), (0, 3, 2.23), (1, 2, 1), (1, 3, 1.41), (2, 3, 1)]
    edge_with_weights = [(0, 1, 48), (0, 2, 91), (1, 2, 63)]
    A = 1231
    B = 1000
    tsp = TSP(edge_with_weights, A=A, B=B, node_size=500, show_graph=False, save_graph=True)
    H_Z_P, H_z = tsp.get_pair_coeff_gate()
    print("Hamiltonian: ")
    print(tsp.Hamiltonian)
    print()
    print("H_z:", H_z)
    print("H_Z_P:", H_Z_P)
    print()