from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
from scipy.optimize import minimize
from hamiltonian import *
import pandas as pd

backend = Aer.get_backend('qasm_simulator')
def inversion_affichage(counts) -> dict:
    return {k[::-1]: v for k, v in counts.items()}


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
    qc.rx(-2 * beta, qreg_q)
    qc.barrier()
    return qc

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

def H(gcp, beta, gamma, p_qc) -> QuantumCircuit:
    """
    Create quantum circuit of Hamiltonian by H_P, which is created in GCP object.

    :param gcp: GCP object
    :type gcp: GCP Object
    :param beta: angles beta
    :type beta: list
    :param gamma: angles gamma
    :type gamma: list
    :param p_qc: circuit depth
    :type p_qc: int
    :return: quantum circuit
    :rtype: QuantumCircuit()
    """
    no_qubits = len(gcp.nodes) * gcp.K
    qreg_q = QuantumRegister(no_qubits, 'q')
    creg_c = ClassicalRegister(no_qubits, 'c')
    qc = QuantumCircuit(qreg_q, creg_c)
    qc.h(qreg_q)  # Apply Hadamard gate
    qc.barrier()
    H_z_p, H_z = gcp.get_pair_coeff_gate()

    for i in range(p_qc):
        h_p = H_P(qreg_q, gamma[i], H_z_p, H_z)
        h_d = H_D(qreg_q, beta[i])
        qc.append(h_d, qreg_q)
        qc.append(h_p, qreg_q)

    qc.barrier()
    qc.measure(qreg_q, creg_c)
    return qc


def function_minimize(gcp, p_qc, no_shots, B, C):
    """
    Objective function of H

    :param gcp: GCP object
    :type gcp: GCP Object
    :param p_qc: circuit depth
    :type p_qc: int
    :return:
    :rtype: function
    """
    quantum_simulator = Aer.get_backend("qasm_simulator")
    def f(theta):
        beta = theta[:p_qc]
        gamma = theta[p_qc:]
        qc = H(gcp, beta, gamma, p_qc)
        job = execute(qc, quantum_simulator, seed_simulator=10, shots=no_shots)
        result = job.result()
        counts = result.get_counts()
        counts = inversion_affichage(counts)  # Reverse qubit order
        return evaluate_H(counts, B, C, len(gcp.nodes), gcp.edges)

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
    # initial_point = np.random.rand(p_qc * 2)

    initial_point = np.asarray([8.817230E-01, 1.687994E+00, 5.426246E-01, 3.096648E-01, 5.537904E-01,2.076591E-01, 2.113032E-01,-3.302992E-01, 1.538847E-01, 2.205492E-01,1.014123E+00, 9.122375E-01, 7.894404E-01, 3.998747E-01, 9.120732E-01,2.787024E+00, 1.581086E+00, 8.330237E-01, 8.494271E-01, 1.436904E-01])

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
    for state in bit_strings:
        if state not in solutions:
            solutions[state] = 0
    return solutions


def run(gcp, p_qc, optim_method, no_shots, no_iters_optim, show_iter, B, C):
    no_qubits = len(gcp.nodes) * gcp.K
    bit_strings = generate_binary_string(no_qubits)

    states = []

    for no_iters in [1, no_iters_optim]:
        print('no_iters:', no_iters)
        angles_objective = function_minimize(gcp, p_qc, no_shots, B, C)
        beta, gamma = minimizer(angles_objective, p_qc, optim_method, no_iters)
        qc = H(gcp, beta, gamma, p_qc)
        solutions = find_solution(qc, no_shots, bit_strings)
        states.append([no_iters, solutions])

    results = compare_cost_by_iter(states, len(gcp.nodes), gcp.edges, B, C)

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

    solutions = {i[0]: i[1] for i in sorted(solutions.items(), key=lambda x: x[1])}
    for state, shot in solutions.items():
        cost_by_state = calculate_cost(state, B, C, len(gcp.nodes),gcp.edges)
        prob = shot / no_shots * 100
        print('state:', state, 'cost:', cost_by_state, 'prob', prob)


    return results


if __name__ == '__main__':
    # make a graph
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 1000
    B = 1000
    C = 100
    gcp = Graph_Coloring(edges, K=K, A=A, node_size=500, show_graph=False, save_graph=False)
    
    no_shots = 2048
    p_qc = 10
    no_iters_optim = 2000
    optim_method = 'cobyla'

    results = run(gcp, p_qc, optim_method, no_shots, no_iters_optim, show_iter=True, B=B, C=C)
    create_chart(name="QAOA", results=results, is_export_data=True)


