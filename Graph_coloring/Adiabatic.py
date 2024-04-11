from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
from hamiltonian import *

backend = Aer.get_backend('qasm_simulator')


def H(gcp, T, no_shots, show_iter=False):
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

    if show_iter:
        start = 0
        end = T+1
        step = 1
    else:
        start = T
        end = T+1
        step = 1


    for i in range(start, end, step):
        print('iter:', i, 'delta_t:', delta_t)
        t = 1
        qreg_q = None
        creg_c = None
        qc = None
        qreg_q = QuantumRegister(no_qubits, 'q')
        creg_c = ClassicalRegister(no_qubits, 'c')
        qc = QuantumCircuit(qreg_q, creg_c)
        qc.reset(qreg_q)
        qc.h(qreg_q)  # Apply Hadamard gate
        qc.barrier()
        for _ in range(i):
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
        solutions = find_solution(qc, no_shots, bit_strings, qreg_q, creg_c)
        states.append(solutions)

    return states


def find_solution(circuit, no_shots, bit_strings, qreg_q, creg_c):
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
    # print("time_taken = ", result.time_taken)
    solutions = result.get_counts(circuit)
    solutions = inversion_affichage(solutions)  # Reverse qubit order
    for state in bit_strings:
        if state not in solutions:
            solutions[state] = 0
    return solutions



def run(gcp, T, no_shots, show_result_of_iter_optim, show_iter):
    states = H(gcp, T, no_shots, show_iter)
    fx = gcp.get_pair_coeff_var()

    results = compare_cost_by_iter(states, fx, no_nodes=len(gcp.nodes))
    solutions = results[-1]
    states =  dict()
    redundants = []

    for k, v in solutions.items():
        if v[1] == 0:
            redundants.append(k)
        states[k] = v[1]
    for k in redundants:
        del solutions[k]

    if show_iter:
        print("best_iter: %.d, estimate_average_cost: %.2f" % (results[0], results[1]))
    else:
        print("best_iter: %.d, estimate_average_cost: %.2f" % (T, results[1]))
        print(solutions)

    fig_counted = plot_histogram(states, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(states, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('distribution_optimal.png', bbox_inches='tight')


def create_table(gcp, T, no_shots):
    table = run(gcp, T, no_shots, show_result_of_iter_optim=True)
    table = sorted(table.items(), key=lambda item: item[0])
    table = {i[0]: i[1] for i in table}
    comparison_file = "comparison.txt"
    with open(comparison_file, 'w') as file:
        for k, v in table.items():
            v_st = []
            for i in v:
                if i == "":
                    v_i = str(0)
                else:
                    # i = round(i / 100, 4)
                    v_i = str(i).replace('.', ',')
                v_st.append(v_i)
            st = '\t'.join(v_st)
            k = str(k).replace('.', ',')
            file.write(k + '\t' + st + "\n")

if __name__ == '__main__':
    # make a graph
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 100
    no_shots = 2048
    T = 100
    gcp = Graph_Coloring(edges,K=K, A=A, node_size=500, show_graph=False, save_graph=True)
    run(gcp, T, no_shots, show_result_of_iter_optim=True, show_iter=False)
