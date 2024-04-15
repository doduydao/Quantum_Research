from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from hamiltonian import *
import pandas as pd


backend = Aer.get_backend('qasm_simulator')
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
    # print("time_taken = ", result.time_taken)
    solutions = result.get_counts(circuit)
    solutions = inversion_affichage(solutions)  # Reverse qubit order
    for state in bit_strings:
        if state not in solutions:
            solutions[state] = 0
    return solutions

def run(tsp, p_qc, optim_method, no_shots, no_iters_optim, show_iter):
    no_qubits = (len(tsp.weights)-1) ** 2
    bit_strings = generate_binary_string(no_qubits)

    states = []

    if show_iter:
        start = 1
        end = no_iters_optim + 1
        step = no_iters_optim - 1
    else:
        start = no_iters_optim
        end = no_iters_optim + 1
        step = 1

    for no_iters in range(start, end, step):
        print('no_iters:', no_iters)
        angles_objective = function_minimize(tsp, p_qc, no_shots)
        beta, gamma = minimizer(angles_objective, p_qc, optim_method, no_iters)
        qc = H(tsp, beta, gamma, p_qc)
        solutions = find_solution(qc, no_shots, bit_strings)
        states.append([no_iters, solutions])

    fx = tsp.get_pair_coeff_var()
    results = compare_cost_by_iter(states, fx)

    solutions = states[-1][1]
    redundants = []

    for k, v in solutions.items():
        if v == 0:
            redundants.append(k)
    for k in redundants:
        del solutions[k]

    fig_counted = plot_histogram(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('QAOA/histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('QAOA/distribution_optimal.png', bbox_inches='tight')

    return results
def create_chart(results, is_export_data=True):
    # results = [[iter, energy, distribution_no_colors, distribution_cost],..]
    data_distribution_cost = dict()
    data_average_cost = dict()
    average_costs = []
    iters = []
    for result_iter in results:
        iter = result_iter[0]
        iters.append(iter)
        average_cost = result_iter[1]
        average_costs.append(average_cost)
        distribution_cost = result_iter[2]
        data_distribution_cost['cost'] = list(distribution_cost.keys())
        data_distribution_cost["iter "+str(iter)] = list(distribution_cost.values())

    data_average_cost['iters'] = iters
    data_average_cost['average_costs'] = average_costs

    cumulative_distribution_cost = calculate_cumulative_prob(data_distribution_cost)

    if is_export_data:
        df = pd.DataFrame.from_dict(data_distribution_cost)
        df.to_csv('QAOA/result_distribution_cost.csv', index=False, sep='\t')
        print("Adiabatic/result_distribution_cost.csv file created successfully!")

        df = pd.DataFrame.from_dict(cumulative_distribution_cost)
        df.to_csv('Adiabatic/result_cumulative_distribution_cost.csv', index=False, sep='\t')
        print("QAOA/result_cumulative_distribution_cost.csv file created successfully!")

        df = pd.DataFrame.from_dict(data_average_cost)
        df.to_csv('QAOA/result_average_cost.csv', index=False, sep='\t')
        print("QAOA/result_average_cost.csv file created successfully!")

    plt.clf()
    plt.figure(figsize=(8, 6))  # Set chart size
    keys = list(data_distribution_cost.keys())[1:]
    df = pd.DataFrame(data_distribution_cost)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['cost'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['cost'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Probability of cost')
    plt.xlabel('Cost')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    plt.savefig('QAOA/Probability of cost.PNG')


    # cummulative proba
    plt.clf()
    plt.figure(figsize=(8, 6))  # Set chart size
    keys = list(cumulative_distribution_cost.keys())[1:]
    df = pd.DataFrame(cumulative_distribution_cost)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['cost'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['cost'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Cumulative probability of cost')
    plt.xlabel('Cost')
    plt.ylabel('Cumulative probability')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('QAOA/Cumulative probability of cost.PNG')


if __name__ == '__main__':
    # make a graph
    edge_with_weights = [(0, 1, 1), (0, 2, 1.41), (0, 3, 2.23), (1, 2, 1), (1, 3, 1.41), (2, 3, 1)]
    A = 1000
    B = 1000
    tsp = TSP(edge_with_weights, A=A, B=B, node_size=500, show_graph=False, save_graph=True)
    no_shots = 2048
    p_qc = 15
    no_iters_optim = 500
    optim_method = 'cobyla'
    results = run(tsp, p_qc, optim_method, no_shots, no_iters_optim, show_iter=True)
    create_chart(results, is_export_data=True)

