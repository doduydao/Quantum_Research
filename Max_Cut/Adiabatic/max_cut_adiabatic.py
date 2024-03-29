from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer, transpile
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
import networkx as nx


def black_box(G, T):
    nodes = G.nodes
    egdes = G.edges

    no_qubits = len(nodes)

    qreg_q = QuantumRegister(no_qubits, 'q')
    creg_c = ClassicalRegister(no_qubits, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.reset(qreg_q)
    circuit.h(qreg_q)

    theta_x = 1
    theta_z = 0
    delta = theta_x / T
    h_z = len(egdes) / 2
    for iter in range(8):
        # print(iter + 1, " ", str(theta_x))
        circuit.rx(theta_x, qreg_q)
        # Rotation of (-1/2).Z_i.Z_j
        t = -0.5 / h_z
        for edge in egdes:
            i = edge[0]
            j = edge[1]
            circuit.cx(qreg_q[i], qreg_q[j])
            circuit.rz(2 * t * theta_z, qreg_q[j])
            circuit.cx(qreg_q[i], qreg_q[j])

        # update angle
        theta_x -= delta
        theta_z += delta

    circuit.measure(qreg_q, creg_c)
    return circuit


if __name__ == '__main__':
    # make a graph
    # G = nx.Graph()
    G = nx.barabasi_albert_graph(20, 11)
    # G.add_nodes_from([0, 1, 2])
    # G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    # G.add_nodes_from([0, 1, 2, 3, 4])
    # G.add_edges_from([(0, 1), (1, 2), (0, 2), (0, 3), (1, 4), (2,4), (1,3)])
    nx.draw(G, with_labels=True, alpha=0.8, node_size=500)
    plt.savefig("Graph.png", format="PNG")
    T = 10
    circuit = black_box(G, T)
    # circuit.draw("mpl")
    # plt.show()

    backend = Aer.get_backend('qasm_simulator')
    NUM_SHOTS = 1000
    job = execute(circuit, backend, shots=NUM_SHOTS)  # NUM_SHOTS
    result = job.result()

    print("time_taken = ", result.time_taken)
    counts = result.get_counts(circuit)
    counts = {k[::-1]: v for k, v in counts.items()}  # Reverse qubit order
    print("count = ", counts)
    fig_counted = plot_histogram(counts)
    fig_proba = plot_distribution(counts, title="Qasm Distribution")
    fig_counted.savefig('counted_result.png', bbox_inches='tight')
    fig_proba.savefig('proba_result.png', bbox_inches='tight')
    # end = time.time()

    # time_taken = end - start
    # print(time_taken)
    # ts.append(result.time_taken)


    import time
    # ts = []
    # for _ in range(10):
    #     # start = time.time()
    #     T = 10000
    #     circuit = black_box(G, T)
    #     # circuit.draw("mpl")
    #     # plt.show()
    #
    #     backend = Aer.get_backend('qasm_simulator')
    #     NUM_SHOTS = 1000
    #     job = execute(circuit, backend, shots=NUM_SHOTS)  # NUM_SHOTS
    #     result = job.result()
    #
    #     print("time_taken = ", result.time_taken)
    #     counts = result.get_counts(circuit)
    #     counts = {k[::-1]: v for k, v in counts.items()}  # Reverse qubit order
    #     print("count = ", counts)
    #     # fig_counted = plot_histogram(counts)
    #     # fig_proba = plot_distribution(counts, title="Qasm Distribution")
    #     # fig_counted.savefig('counted_result.png', bbox_inches='tight')
    #     # fig_proba.savefig('proba_result.png', bbox_inches='tight')
    #     # end = time.time()
    #
    #     # time_taken = end - start
    #     # print(time_taken)
    #     ts.append(result.time_taken)
    # print('avg_time:', sum(ts)/len(ts))