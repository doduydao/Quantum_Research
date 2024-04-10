from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
from tsp import TSP, calculate_cost, inversion_affichage, generate_binary_string
import math
from utils import *
backend = Aer.get_backend('qasm_simulator')
# def H_P(qreg_q, delta_t, H_z_p, H_z) -> QuantumCircuit:
#     """
#     Create H_P part of Hamiltonian
#
#     :param qreg_q: Quantum register
#     :type qreg_q: QuantumRegister()
#     :param delta_t: step delta_t
#     :type delta_t: float
#     :param H_z_p: H_z_p part of H_P
#     :type H_z_p: list
#     :param H_z: H_z part of H_P
#     :type H_z: float
#     :return: Quantum circuit of H_P part
#     :rtype: QuantumCircuit
#     """
#     qc = QuantumCircuit(qreg_q)
#     for h in H_z_p:
#         w = h[0] / H_z
#         idx = h[1]
#         if len(idx) == 1:
#             qc.rz(2 * delta_t * w, qreg_q[idx[0]])
#         else:
#             i = idx[0]
#             j = idx[1]
#             qc.cx(qreg_q[i], qreg_q[j])
#             qc.rz(2 * delta_t * w, qreg_q[j])
#             qc.cx(qreg_q[i], qreg_q[j])
#     qc.barrier()
#     return qc


# def H_D(qreg_q, delta_t) -> QuantumCircuit:
#     """
#     Create H_D part of Hamiltonian
#
#     :param qreg_q: Quantum register
#     :type qreg_q: QuantumRegister()
#     :param delta_t: step delta_t
#     :type delta_t: float
#     :return: Quantum circuit of H_D part
#     :rtype: QuantumCircuit()
#     """
#     qc = QuantumCircuit(qreg_q)
#
#     qc.barrier()
#     return qc



def H(tsp, T, no_shots, show_iter=False):
    """
    Create quantum circuit of Hamiltonian by H_P, which is created in TSP object.

    :param tsp: TSP object
    :type tsp: TSP Object
    :param T: number of iterations
    :type T: int
    :return: quantum circuit
    :rtype:
    """
    H_z_p, H_z = tsp.get_pair_coeff_gate()

    no_qubits = (len(tsp.weights) - 1) ** 2
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



def run(tsp, T, no_shots, show_result_of_iter_optim, show_iter):
    states = H(tsp, T, no_shots, show_iter)
    fx = tsp.get_pair_coeff_var()

    best_result = compare_cost_by_iter(states, fx)
    print("best_iter: %.d, estimate_average_cost: %.2f" % (best_result[0], best_result[1]))
    solutions = best_result[-1]
    fig_counted = plot_histogram(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('distribution_optimal.png', bbox_inches='tight')

    # table = dict()
    # start = T
    # step = T
    # if show_result_of_iter_optim:
    #     start = 0
    #
    # for no_iters_optim in range(start, T + step, step):
    #

    #     counted = dict()
    #     distribution = dict()
    #     solution_dict = dict()
    #     for k, v in solutions.items():
    #         cost = round(calculate_cost(fx, k), 6)
    #         if cost not in counted:
    #             counted[cost] = v
    #         else:
    #             counted[cost] += v
    #
    #         if cost not in distribution:
    #             distribution[cost] = v / no_shots * 100
    #         else:
    #             distribution[cost] += v / no_shots * 100
    #
    #         solution_dict[k] = [v / no_shots * 100, cost]
    #
    #     distribution = sorted(distribution.items(), key=lambda item: item[0])
    #     counted = sorted(counted.items(), key=lambda item: item[0])
    #     counted_sorted_by_count = sorted(counted, key=lambda item: item[1], reverse=True)
    #     solution_dict = sorted(solution_dict.items(), key=lambda item: item[1][1])
    #
    #     print("distribution:")
    #     print(distribution)
    #     print("counted_sorted_by_count")
    #     print(counted_sorted_by_count)
    #     print()
    #     print("solution with probability")
    #     for s in solution_dict[:math.factorial(len(tsp.weights) - 1)]:
    #         print(s[0], "\t", round(s[1][1], 2), "\t", round(s[1][0], 2))
    #     print()
    #     distribution = {i[0]: i[1] for i in distribution}
    #
    #     if no_iters_optim == 0:
    #         cost = list(distribution.keys())
    #         proba = list(distribution.values())
    #         for k in range(len(cost)):
    #             if cost[k] not in table:
    #                 table[cost[k]] = [proba[k]]
    #             else:
    #                 table[cost[k]] += [proba[k]]
    #     if no_iters_optim == T:
    #         cost = list(distribution.keys())
    #         proba = list(distribution.values())
    #         for k in range(len(cost)):
    #             if cost[k] not in table:
    #                 table[cost[k]] = [0, proba[k]]
    #             else:
    #                 table[cost[k]] += [proba[k]]
    #         for k, v in table.items():
    #             if len(v) == 1:
    #                 table[k].append(0)
    #
    # return table

def create_table(tsp, T, no_shots):
    table = run(tsp, T, no_shots, show_result_of_iter_optim=True)
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
    edge_with_weights = [(0, 1, 1), (0, 2, 1.41), (0, 3, 2.23), (1, 2, 1), (1, 3, 1.41), (2, 3, 1)]
    A = 10
    B = 10
    no_shots = 2048
    T = 100
    tsp = TSP(edge_with_weights, A=A, B=B, node_size=500, show_graph=False, save_graph=True)
    run(tsp, T, no_shots, show_result_of_iter_optim=True, show_iter=True)
    # circuit = H(tsp, T)
    # solutions = find_solution(circuit, no_shots)
    # print(solutions)
    # create_table(tsp, T, no_shots)
