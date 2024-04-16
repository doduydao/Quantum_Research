from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
from hamiltonian import *
import pandas as pd

backend = Aer.get_backend('qasm_simulator')


def H(gcp, T, no_shots):
    """
    Create quantum circuit of Hamiltonian by H_P, which is created in gcp object.

    :param gcp: cp object
    :type gcp: gcp Object
    :param T: number of iterations
    :type T: int
    :return: quantum circuit
    :rtype:
    """
    H_z_p, H_z = gcp.get_pair_coeff_gate()

    no_qubits = len(gcp.nodes) * gcp.K
    bit_strings = generate_binary_string(no_qubits)

    delta_t = 1 / T
    num_str = str(delta_t)
    split_num = num_str.split(".")
    num_decimals = len(split_num[1])

    states = []
    for iter in [1, T]:
        print('iter:', iter, 'delta_t:', delta_t)
        t = 1
        qreg_q = QuantumRegister(no_qubits, 'q')
        creg_c = ClassicalRegister(no_qubits, 'c')
        qc = QuantumCircuit(qreg_q, creg_c)
        qc.reset(qreg_q)
        qc.h(qreg_q)  # Apply Hadamard gate
        qc.barrier()

        for _ in range(iter):
            qc.rx(-2 * t, qreg_q)
            for h in H_z_p:
                w = h[0] / H_z
                idx = h[1]
                if len(idx) == 1:
                    qc.rz(2 * (1-t) * w, qreg_q[idx[0]])
                else:
                    i = idx[0]
                    j = idx[1]
                    qc.cx(qreg_q[i], qreg_q[j])
                    qc.rz(2 * (1-t) * w, qreg_q[j])
                    qc.cx(qreg_q[i], qreg_q[j])
            t = round(t - delta_t, num_decimals)
        qc.measure(qreg_q, creg_c)
        solutions = find_solution(qc, no_shots, bit_strings)
        states.append([iter, solutions])
        # print(qc)
    return states


def find_solution(circuit, no_shots, bit_strings):
    """
    :param circuit:
    :type circuit:
    :param no_shots:
    :type no_shots:
    :return:
    :rtype:
    """

    job = execute(circuit, backend, seed_simulator=10, shots=no_shots)  # NUM_SHOTS
    result = job.result()
    solutions = result.get_counts(circuit)
    solutions = inversion_affichage(solutions)  # Reverse qubit order
    for bit in bit_strings:
        if bit not in solutions:
            solutions[bit] = 0
    return solutions


def run(gcp, T, no_shots, P, C):
    states = H(gcp, T, no_shots)
    fx = gcp.get_pair_coeff_var()

    results = compare_cost_by_iter(states, fx, len(gcp.nodes), gcp.edges, P, C)
    solutions = states[-1][-1]
    redundants = []

    for k, v in solutions.items():
        if v == 0:
            redundants.append(k)
    for k in redundants:
        del solutions[k]

    fig_counted = plot_histogram(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('Adiabatic/histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('Adiabatic/distribution_optimal.png', bbox_inches='tight')

    # distribution = dict()
    # for state, shot in solutions.items():
    #     no_conflict = calculate_no_conflict(state, len(gcp.nodes), gcp.edges)
    #     no_colors = count_color(state, len(gcp.nodes))
    #     cost_by_state = P * penalty_part(state, len(gcp.nodes)) + C * no_conflict + no_colors
    #     prob = shot / no_shots * 100
    #     distribution[state] = [cost_by_state, prob]
        # print('state:',state, 'cost:', cost_by_state, 'prob', prob)

    return results


if __name__ == '__main__':
    # make a graph
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 1000
    P = A
    C = 100
    no_shots = 2048
    T = 100

    gcp = Graph_Coloring(edges,K=K, A=A, node_size=500, show_graph=False, save_graph=True)
    results = run(gcp, T, no_shots, P=P, C=C)
    create_chart(results, is_export_data=True)