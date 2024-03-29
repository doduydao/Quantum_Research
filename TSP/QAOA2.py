import numpy
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# from tsp2 import TSP, find_best_solution, make_order, filter_solution
from tsp import TSP, find_best_solution, make_order, filter_solution


def inversion_affichage(counts):
    return {k[::-1]: v for k, v in counts.items()}


def H_P(qreg_q, gamma, H_Z_P, H_z):
    qc = QuantumCircuit(qreg_q)
    for h in H_Z_P:
        w = h[0] / H_z
        indices = h[1]
        if len(indices) == 1:
            qc.rz(2 * gamma * w, qreg_q[indices[0]])
        else:
            i = indices[0]
            j = indices[1]
            qc.cx(qreg_q[i], qreg_q[j])
            qc.rz(2 * gamma * w, qreg_q[j])
            qc.cx(qreg_q[i], qreg_q[j])
    qc.barrier()
    return qc


def H_D(qreg_q, beta):
    qc = QuantumCircuit(qreg_q)
    qc.rx(2 * beta, qreg_q)
    qc.barrier()
    return qc


def calculate_cost(fx, solution):
    xs = [int(char) for char in solution]
    cost = 0
    for e in fx[:-1]:
        w = e[0]
        x_ip = xs[e[1][0]]
        for i in range(1, len(e[1])):
            x_ip *= xs[e[1][i]]
        cost += w * x_ip
    cost += fx[-1]
    return cost


def evaluate_H(fx, counts, no_shots):
    energy = 0
    for state, count in counts.items():
        cost = calculate_cost(fx, state)
        energy += cost * count
    print(energy/no_shots)
    return energy/no_shots


def H(tsp, beta, gamma, p_qc):
    # no_qubits = (len(tsp.weights)) ** 2
    no_qubits = (len(tsp.weights) - 1) ** 2
    qreg_q = QuantumRegister(no_qubits, 'q')
    creg_c = ClassicalRegister(no_qubits, 'c')
    qc = QuantumCircuit(qreg_q, creg_c)

    # Apply Hardamard gate
    qc.h(qreg_q)
    qc.barrier()
    H_Z_P, H_z = tsp.get_pair_coeff_gate()
    # print(H_Z_P)

    for i in range(p_qc):
        h_p = H_P(qreg_q, gamma[i], H_Z_P, H_z)
        h_d = H_D(qreg_q, beta[i])
        qc.append(h_p, qreg_q)
        qc.append(h_d, qreg_q)

    qc.barrier()
    qc.measure(qreg_q, creg_c)
    return qc


def minimizer(tsp, p_qc, no_shots):
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
        return evaluate_H(fx, counts, no_shots)
    return f


def make_circuit(tsp, p_qc, optim_method, no_iters_optim, no_shots):
    angles_objective = minimizer(tsp, p_qc, no_shots)
    # theta = numpy.array([2.819630E+00,  -1.148550E-01,  5.240590E-01,   7.516110E-01,   5.030128E-03, 4.223003E-03 ,  8.949918E-03,   2.736550E-01,   4.036071E-04,   1.346727E+00, 1.446576E-01,   1.137636E+00,   8.016028E-02,   6.019606E-01,   2.279614E+00, 1.366261E+00,   1.086591E+00,   6.453142E-01,   7.103665E-01,   2.199607E+00])
    theta = np.random.rand(p_qc * 2)
    res_sample = minimize(angles_objective,
                          theta,
                          method=optim_method,
                          options={'maxiter': no_iters_optim, 'disp': True}
                          )
    print(res_sample)
    optimal_theta = res_sample['x']
    beta = optimal_theta[:p_qc]
    gamma = optimal_theta[p_qc:]
    # print("angles beta : ", beta)
    # print("angles gamma : ", gamma)

    qc = H(tsp, beta, gamma, p_qc)
    return qc


def find_solution(circuit, no_shots):
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, seed_simulator=10, shots=no_shots)  # NUM_SHOTS
    result = job.result()
    # print("time_taken = ", result.time_taken)

    solutions = result.get_counts(circuit)
    solutions = inversion_affichage(solutions)  # Reverse qubit order

    fig_counted = plot_histogram(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('distribution_optimal.png', bbox_inches='tight')

    return solutions, result


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
        solutions, result = find_solution(circuit, no_shots)
        print(solutions)
        fx = tsp.get_pair_coeff_var()
        counted = dict()
        distribution = dict()
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

        distribution = sorted(distribution.items(), key=lambda item: item[0])
        counted = sorted(counted.items(), key=lambda item: item[0])
        counted_sorted_by_count = sorted(counted, key=lambda item: item[1], reverse=True)
        # distribution_sorted_by_count = sorted(distribution.items(), key=lambda item: item[0])
        print("distribution:")
        print(distribution)
        print()
        print("counted")
        print(counted)
        print()

        print("counted_sorted_by_count")
        print(counted_sorted_by_count)
        print()

        distribution = {i[0]: i[1] for i in distribution}
        counted = {i[0]: i[1] for i in counted}

        make_distribution(distribution, no_iters_optim)
        print()
        accumulate_distribution(distribution, no_iters_optim)

        if no_iters_optim == 0 or no_iters_optim == total:
            cost = list(distribution.keys())
            proba = list(distribution.values())
            for k in range(len(cost)):
                if cost[k] not in table:
                    table[cost[k]] = [proba[k]]
                else:
                    table[cost[k]] += [proba[k]]
    return table


if __name__ == '__main__':
    # make a graph
    edge_with_weights = [(0, 1, 1), (0, 2, 1.41), (0, 3, 2.23), (1, 2, 1), (1, 3, 1.41), (2, 3, 1)]
    # edge_with_weights = [(0, 1, 48), (0, 2, 91), (1, 2, 63)]
    A = 1213
    B = 1213
    no_shots = 2048
    p_qc = 10
    no_iters_optim = 150
    optim_method = 'cobyla'

    tsp = TSP(edge_with_weights, A=A, B=B, node_size=500, show_graph=False, save_graph=True)
    H_Z_P, H_z = tsp.get_pair_coeff_gate()
    print("Hamiltonian: ")
    print(tsp.Hamiltonian)
    print()
    print("H_z:", H_z)
    print("H_Z_P:", H_Z_P)
    print()

    table = run(tsp, p_qc, optim_method, no_shots, no_iters_optim, show_result_of_iter_optim=True)
    table = sorted(table.items(), key=lambda item: item[0])
    table = {i[0]: i[1] for i in table}
    comparison_file = "comparison.txt"
    with open(comparison_file, 'w') as file:
        for k, v in table.items():
            v_st = [str(i).replace('.', ',') for i in v]
            st = '\t'.join(v_st)
            k = str(k).replace('.', ',')
            file.write(k + '\t' + st + "\n")