from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import math
import numpy as np
from scipy.optimize import minimize
import networkx as nx
import matplotlib.pyplot as plt


def inversion_affichage(counts):
    return {k[::-1]: v for k, v in counts.items()}


def H_P(gamma, qreg_q, edges):
    qc = QuantumCircuit(qreg_q)
    h_z = len(edges) / 2
    t = 1 / h_z

    # arc i - j
    for edge in edges:
        i = edge[0]
        j = edge[1]
        qc.cx(qreg_q[i], qreg_q[j])
        qc.rz(2 * gamma * t, qreg_q[j])
        qc.cx(qreg_q[i], qreg_q[j])
        qc.barrier()
    return qc


def H_D(beta, qreg_q):
    qc = QuantumCircuit(qreg_q)
    qc.rx(2 * beta, qreg_q)
    return qc


def cost_func(solution):
    cut_value = 0
    l = len(solution)
    m = math.ceil(len(solution) / 2)
    for i in range(0, l - 1):
        for j in range(i + 1, l):
            if solution[i] != solution[j]:
                if (i in range(0, m)) and (j in range(m, l)):
                    cut_value += 1
    return -cut_value


def evaluate_H(counts):
    energy = 0
    total_counts = 0
    for state, count in counts.items():
        cost = cost_func(state)
        energy += cost * count
        total_counts += count
    return energy / total_counts


def H(no_qubits, beta, gamma, p_qc, edges):
    qreg_q = QuantumRegister(no_qubits, 'q')
    creg_c = ClassicalRegister(no_qubits, 'c')
    qc = QuantumCircuit(qreg_q, creg_c)

    # Apply Hardamard gate
    qc.h(qreg_q)
    qc.barrier()

    for i in range(0, p_qc):
        qp = H_P(gamma[i], qreg_q, edges)
        qd = H_D(beta[i], qreg_q)
        qc.append(qp, [q for q in qreg_q])
        qc.append(qd, [q for q in qreg_q])

    qc.barrier()
    qc.measure(qreg_q, creg_c)
    return qc


def minimizer(p, no_qubits, edges):
    quantum_simulator = Aer.get_backend("qasm_simulator")

    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = H(no_qubits, beta, gamma, p, edges)
        job = execute(qc, quantum_simulator, seed_simulator=10, shots=2048)
        result = job.result()
        counts = result.get_counts()
        counts = inversion_affichage(counts)  # Reverse qubit order
        return evaluate_H(counts)

    return f


def black_box(G, depth_of_circuit, optim_method, no_iters_optim):
    nodes = G.nodes
    edges = G.edges
    no_qubits = len(nodes)
    angles_objective = minimizer(depth_of_circuit, no_qubits, edges)
    theta = np.random.rand(depth_of_circuit * 2)
    res_sample = minimize(angles_objective,
                          theta,
                          method=optim_method,
                          options={'maxiter': no_iters_optim, 'disp': True}
                          )
    print(res_sample)
    optimal_theta = res_sample['x']
    beta = optimal_theta[:depth_of_circuit]
    gamma = optimal_theta[depth_of_circuit:]
    print("angles beta : ", beta)
    print("angles gamma : ", gamma)

    qc = H(no_qubits, beta, gamma, depth_of_circuit, edges)
    return qc


if __name__ == '__main__':
    # make a graph
    G = nx.Graph()
    # G.add_nodes_from([0, 1, 2, 3, 4])
    # G.add_edges_from([(0, 1), (1, 2), (0, 2), (0, 3), (1, 4), (2, 4), (1, 3)])
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    nx.draw(G, with_labels=True, alpha=0.8, node_size=500)
    plt.savefig("Graph.png", format="PNG")

    optim_method = 'COBYLA'
    no_iters_optim = 50
    depth_of_circuit = 5
    circuit = black_box(G, depth_of_circuit, optim_method, no_iters_optim)
    # circuit.draw("mpl")
    # plt.show()

    backend = Aer.get_backend('qasm_simulator')
    NUM_SHOTS = 1000
    job = execute(circuit, backend, shots=NUM_SHOTS)  # NUM_SHOTS
    result = job.result()

    print("time_taken = ", result.time_taken)
    counts = result.get_counts(circuit)
    counts = inversion_affichage(counts)  # Reverse qubit order
    print("count = ", counts)
    fig_counted = plot_histogram(counts)
    fig_proba = plot_distribution(counts, title="Qasm Distribution")
    fig_counted.savefig('histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('distribution_optimal.png', bbox_inches='tight')
