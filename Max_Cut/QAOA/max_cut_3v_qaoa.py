from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import math
import numpy as np
from scipy.optimize import minimize


def inversion_affichage(counts):
    return {k[::-1]: v for k, v in counts.items()}


def H_P(gamma, qreg_q):
    qc = QuantumCircuit(qreg_q)
    h_z = 6 / 2
    t = 1 / h_z
    # arc 1 - 2
    qc.cx(qreg_q[0], qreg_q[1])
    qc.rz(2 * gamma * t, qreg_q[1])
    qc.cx(qreg_q[0], qreg_q[1])
    qc.barrier()
    # arc 1 - 3
    qc.cx(qreg_q[0], qreg_q[2])
    qc.rz(2 * gamma * t, qreg_q[2])
    qc.cx(qreg_q[0], qreg_q[2])
    qc.barrier()
    # arc 2 - 3
    qc.cx(qreg_q[1], qreg_q[2])
    qc.rz(2 * gamma * t, qreg_q[2])
    qc.cx(qreg_q[1], qreg_q[2])
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


def H(no_qubits, beta, gamma, p_qc):
    qreg_q = QuantumRegister(no_qubits, 'q')
    creg_c = ClassicalRegister(no_qubits, 'c')
    qc = QuantumCircuit(qreg_q, creg_c)

    # Apply Hardamard gate
    qc.h(qreg_q)
    qc.barrier()

    for i in range(0, p_qc):
        qp = H_P(gamma[i], qreg_q)
        qd = H_D(beta[i], qreg_q)
        qc.append(qp, [qreg_q[0], qreg_q[1], qreg_q[2]])
        qc.append(qd, [qreg_q[0], qreg_q[1], qreg_q[2]])

    qc.barrier()
    qc.measure(qreg_q, creg_c)
    return qc


def minimizer(p, no_qubits):
    quantum_simulator = Aer.get_backend("qasm_simulator")

    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = H(no_qubits, beta, gamma, p)
        job = execute(qc, quantum_simulator, seed_simulator=10, shots=2048)
        result = job.result()
        counts = result.get_counts()
        counts = inversion_affichage(counts)  # Reverse qubit order
        return evaluate_H(counts)

    return f


quantum_simulator = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 1000
no_qubits = 3
p = 5

angles_objective = minimizer(p, no_qubits)
theta = np.random.rand(p * 2)
res_sample = minimize(angles_objective,
                      theta,
                      method='COBYLA',
                      options={'maxiter': 100, 'disp': True}
                      )
print(res_sample)
optimal_theta = res_sample['x']
beta = optimal_theta[:p]
gamma = optimal_theta[p:]
print("angles beta : ", beta)
print("angles gamma : ", gamma)

qc = H(no_qubits, beta, gamma, p)
job = execute(qc, quantum_simulator, shots=NUM_SHOTS)
result = job.result()
print("time_taken = ", result.time_taken)
counts = result.get_counts()
counts = inversion_affichage(counts)
print(counts)
fig_histogram = plot_histogram(counts)
fig_distribution = plot_distribution(counts)
fig_histogram.savefig('histogram_optimal.png')
fig_distribution.savefig('distribution_optimal.png')
