import numpy
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tsp import TSP, filter_solution
import math


def inversion_affichage(counts) -> dict:
    return {k[::-1]: v for k, v in counts.items()}


def H_P(qreg_q, gamma, H_z_p, H_z) -> QuantumCircuit:
    """
    Create H_P part of Hamiltonian

    :param qreg_q: Quantum register
    :type qreg_q: QuantumRegister()
    :param gamma: angle
    :type gamma: float
    :param H_z_p: H_z_p part of H_P
    :type H_z_p: list
    :param H_z: H_z part of H_P
    :type H_z: float
    :return: Quantum circuit of H_P part
    :rtype: QuantumCircuit
    """
    qc = QuantumCircuit(qreg_q)
    for h in H_z_p:
        w = h[0] / H_z
        idx = h[1]
        if len(idx) == 1:
            qc.rz(2 * gamma * w, qreg_q[idx[0]])
        else:
            i = idx[0]
            j = idx[1]
            qc.cx(qreg_q[i], qreg_q[j])
            qc.rz(2 * gamma * w, qreg_q[j])
            qc.cx(qreg_q[i], qreg_q[j])
    qc.barrier()
    return qc


def H_D(qreg_q, beta) -> QuantumCircuit:
    """
    Create H_D part of Hamiltonian

    :param qreg_q: Quantum register
    :type qreg_q: QuantumRegister()
    :param beta: angle
    :type beta: float
    :return: Quantum circuit of H_D part
    :rtype: QuantumCircuit()
    """
    qc = QuantumCircuit(qreg_q)
    qc.rx(2 * beta, qreg_q)
    qc.barrier()
    return qc


def calculate_cost(fx, state) -> float:
    """
    Calculate cost of tour (state)

    :param fx: cost function
    :type fx: list. ex: [[coef,[x_0_0, x_0_1]],...,[], offset]
    :param state: one state can be a solution
    :type state: string
    :return: cost
    :rtype: float
    """
    xs = [int(char) for char in state]
    cost = 0
    for e in fx[:-1]:
        w = e[0]
        x_ip = xs[e[1][0]]
        for i in range(1, len(e[1])):
            x_ip *= xs[e[1][i]]
        cost += w * x_ip
    cost += fx[-1]
    return cost


def evaluate_H(fx, solutions) -> float:
    """
    Estimate the cost of solutions, that are made by H

    :param fx: cost fucntion
    :type fx: list
    :param solutions: all states and counted corresponding
    :type solutions: dictionary
    :return: Expectation of cost
    :rtype: float
    """
    energy = 0
    for state, count in solutions.items():
        cost = calculate_cost(fx, state)
        energy += cost * count
    total = sum(list(solutions.values()))
    return energy / total


def H(tsp, beta, gamma, p_qc) -> QuantumCircuit:
    """
    Create quantum circuit of Hamiltonian by H_P, which is created in TSP object.

    :param tsp: TSP object
    :type tsp: TSP Object
    :param beta: angles beta
    :type beta: list
    :param gamma: angles gamma
    :type gamma: list
    :param p_qc: circuit depth
    :type p_qc: int
    :return: quantum circuit
    :rtype: QuantumCircuit()
    """
    no_qubits = (len(tsp.weights) - 1) ** 2
    qreg_q = QuantumRegister(no_qubits, 'q')
    creg_c = ClassicalRegister(no_qubits, 'c')
    qc = QuantumCircuit(qreg_q, creg_c)
    qc.h(qreg_q)  # Apply Hadamard gate
    qc.barrier()
    H_z_p, H_z = tsp.get_pair_coeff_gate()

    for i in range(p_qc):
        h_p = H_P(qreg_q, gamma[i], H_z_p, H_z)
        h_d = H_D(qreg_q, beta[i])
        qc.append(h_p, qreg_q)
        qc.append(h_d, qreg_q)

    qc.barrier()
    qc.measure(qreg_q, creg_c)
    return qc


def function_minimize(tsp, p_qc, no_shots):
    """
    Objective function of H

    :param tsp: TSP object
    :type tsp: TSP Object
    :param p_qc: circuit depth
    :type p_qc: int
    :return:
    :rtype: function
    """
    quantum_simulator = Aer.get_backend("qasm_simulator")
    fx = tsp.get_pair_coeff_var()

    def f(theta):
        beta = theta[:p_qc]
        gamma = theta[p_qc:]
        qc = H(tsp, beta, gamma, p_qc)
        job = execute(qc, quantum_simulator, seed_simulator=10, shots=no_shots)
        result = job.result()
        counts = result.get_counts()
        counts = inversion_affichage(counts)  # Reverse qubit order
        return evaluate_H(fx, counts)

    return f


def minimizer(angles_objective, p_qc, optim_method, no_iters_optim):
    """
    Find best (beta, gamma) angles

    :param angles_objective: minimize function
    :type angles_objective: function
    :param p_qc: circuit depth
    :type p_qc: int
    :param optim_method: optimization method
    :type optim_method: string
    :param no_iters_optim: number of iteration
    :type no_iters_optim: int
    :param no_shots: number of samples
    :type no_shots: int
    :return: best beta, gamma angle
    :rtype: list, list
    """
    # initial_point = np.zeros(p_qc * 2)
    initial_point = np.random.rand(p_qc * 2)
    res_sample = minimize(angles_objective,
                          initial_point,
                          method=optim_method,
                          options={'maxiter': no_iters_optim,
                                   'disp': True}
                          )
    print(res_sample)
    optimal_theta = res_sample['x']
    beta = optimal_theta[:p_qc]
    gamma = optimal_theta[p_qc:]
    return beta, gamma
def make_circuit(tsp, p_qc, optim_method, no_iters_optim, no_shots) -> QuantumCircuit:
    """
    Create quantum circuit for Hamiltonian of TSP

    :param tsp: TSP object
    :type tsp: TSP Object
    :param p_qc: circuit depth
    :type p_qc: int
    :param optim_method: optimization method
    :type optim_method: string
    :param no_iters_optim: number of iteration
    :type no_iters_optim: int
    :param no_shots: number of samples
    :type no_shots: int
    :return: quantum circuit
    :rtype: QuantumCircuit
    """
    angles_objective = function_minimize(tsp, p_qc, no_shots)
    beta, gamma = minimizer(angles_objective, p_qc, optim_method, no_iters_optim)
    # theta = numpy.array([2.819630E+00,  -1.148550E-01,  5.240590E-01,   7.516110E-01,   5.030128E-03, 4.223003E-03 ,  8.949918E-03,   2.736550E-01,   4.036071E-04,   1.346727E+00, 1.446576E-01,   1.137636E+00,   8.016028E-02,   6.019606E-01,   2.279614E+00, 1.366261E+00,   1.086591E+00,   6.453142E-01,   7.103665E-01,   2.199607E+00])
    # theta = numpy.array([3.368268E-01, 5.997952E-01, 1.591153E-01, 9.593521E-01, 4.316945E-01, -9.036192E-02, 5.965993E-02, 7.554193E-01, 2.655019E-01, 1.418550E-01,  4.493724E-01, 8.504383E-01, 6.524912E-01, 4.687341E-02, 8.285875E-01,  5.884106E-01, 1.929075E-01, 1.270839E+00, 6.292845E-01, 3.055395E-01,  3.143851E-01, 4.368999E-01, 5.340962E-01, 2.562147E-01, 1.645299E+00,  5.213757E-01, 2.814088E-01, 2.571498E-01, 4.272136E-01, 2.624298E-01])
    # theta = np.random.rand(p_qc * 2)
    qc = H(tsp, beta, gamma, p_qc)
    return qc


def find_solution(circuit, no_shots):
    """

    :param circuit:
    :type circuit:
    :param no_shots:
    :type no_shots:
    :return:
    :rtype:
    """
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, seed_simulator=10, shots=no_shots)  # NUM_SHOTS
    result = job.result()
    # print("time_taken = ", result.time_taken)
    solutions = result.get_counts(circuit)
    solutions = inversion_affichage(solutions)  # Reverse qubit order
    solutions = filter_solution(solutions, no_shots)
    fig_counted = plot_histogram(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('distribution_optimal.png', bbox_inches='tight')

    return solutions


def make_distribution(distribution, no_iters_optim):
    k = distribution.keys()
    v = distribution.values()
    print("make_distribution")
    print(k)
    print(v)
    plt.close('all')
    plt.plot(k, v)
    plt.xlabel('Cost')
    plt.ylabel('Probability')
    plt.title("Distribution")
    file_name = "images/Distribution_" + str(no_iters_optim) + ".png"
    plt.savefig(file_name)


def accumulate_distribution(distribution, no_iters_optim):
    v = list(distribution.values())
    value = [v[0]]
    for i in range(1, len(v)):
        accumlate_proba = value[i - 1] + v[i]
        value.append(accumlate_proba)
    # print(key)
    k = distribution.keys()
    print("accumulate_distribution")
    print(k)
    print(value)
    plt.close('all')
    plt.plot(k, value)
    plt.xlabel('Cost')
    plt.ylabel('Probability')
    plt.title("Accumulate distribution")
    file_name = "images/accumulate_distribution_" + str(no_iters_optim) + ".png"
    plt.savefig(file_name)


def run(tsp, p_qc, optim_method, no_shots, total, show_result_of_iter_optim):
    table = dict()

    start = total
    step = total
    if show_result_of_iter_optim:
        start = 0

    for no_iters_optim in range(start, total + step, step):
        circuit = make_circuit(tsp, p_qc, optim_method, no_iters_optim, no_shots)
        # circuit.draw("mpl")
        # plt.show()
        solutions = find_solution(circuit, no_shots)
        # print(solutions)
        fx = tsp.get_pair_coeff_var()
        counted = dict()
        distribution = dict()
        solution_dict = dict()
        for k, v in solutions.items():
            cost = round(calculate_cost(fx, k), 6)
            if cost not in counted:
                counted[cost] = v
            else:
                counted[cost] += v

            if cost not in distribution:
                distribution[cost] = v / no_shots * 100
            else:
                distribution[cost] += v / no_shots * 100

            solution_dict[k] = [v / no_shots * 100, cost]

        distribution = sorted(distribution.items(), key=lambda item: item[0])
        counted = sorted(counted.items(), key=lambda item: item[0])
        counted_sorted_by_count = sorted(counted, key=lambda item: item[1], reverse=True)
        solution_dict = sorted(solution_dict.items(), key=lambda item: item[1][1])
        # distribution_sorted_by_count = sorted(distribution.items(), key=lambda item: item[0])
        print("distribution:")
        print(distribution)
        # print()
        # print("counted")
        # print(counted)
        # print()
        print("counted_sorted_by_count")
        print(counted_sorted_by_count)
        print()
        print("solution with probability")
        for s in solution_dict[:math.factorial(len(tsp.weights) - 1)]:
            print(s[0], "\t", round(s[1][1], 2), "\t", round(s[1][0], 2))
        print()
        distribution = {i[0]: i[1] for i in distribution}
        # counted = {i[0]: i[1] for i in counted}

        # make_distribution(distribution, no_iters_optim)
        # print()
        # accumulate_distribution(distribution, no_iters_optim)

        if no_iters_optim == 0:
            cost = list(distribution.keys())
            proba = list(distribution.values())
            for k in range(len(cost)):
                if cost[k] not in table:
                    table[cost[k]] = [proba[k]]
                else:
                    table[cost[k]] += [proba[k]]
        if no_iters_optim == total:
            cost = list(distribution.keys())
            proba = list(distribution.values())
            for k in range(len(cost)):
                if cost[k] not in table:
                    table[cost[k]] = [0, proba[k]]
                else:
                    table[cost[k]] += [proba[k]]
            for k, v in table.items():
                if len(v) == 1:
                    table[k].append(0)

    return table


if __name__ == '__main__':
    # make a graph
    edge_with_weights = [(0, 1, 1), (0, 2, 1.41), (0, 3, 2.23), (1, 2, 1), (1, 3, 1.41), (2, 3, 1)]
    # edge_with_weights = [(0, 1, 48), (0, 2, 91), (1, 2, 63)]
    # A = max([i[2] for i in edge_with_weights])**2
    # B = A
    A = 10000
    B = 10000
    no_shots = 2048
    p_qc = 10
    no_iters_optim = 500
    optim_method = 'cobyla'

    tsp = TSP(edge_with_weights, A=A, B=B, node_size=500, show_graph=False, save_graph=True)
    # H_Z_P, H_z = tsp.get_pair_coeff_gate()
    # print("Hamiltonian: ")
    # print(tsp.Hamiltonian)
    # print()
    # print("H_z:", H_z)
    # print("H_Z_P:", H_Z_P)
    # print()

    table = run(tsp, p_qc, optim_method, no_shots, no_iters_optim, show_result_of_iter_optim=True)
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
