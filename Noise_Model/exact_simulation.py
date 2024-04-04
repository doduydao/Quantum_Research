from qiskit.primitives import BackendEstimator, Estimator
from qiskit_aer import Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt
# backend = Aer.get_backend('qasm_simulator')
# backend_estimator = BackendEstimator(backend)
from qiskit.circuit import Parameter
import numpy as np
from TSP.tsp import *

# observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])
# print(observable)
# qc = QuantumCircuit(2)
# qc.h(0)
# qc.x(0)
# qc.h(1)
# qc.x(1)
# # qc.cx(0, 1)
# print(qc.draw(style="iqp"))
# # plt.show()
# estimator = Estimator()
# job = estimator.run(qc, observable)
# result = job.result()
# exp_value = result.values[0]
# print(exp_value)
#
# theta = Parameter('θ')
# param_qc = QuantumCircuit(2)
# param_qc.rx(theta, 0)
# param_qc.rx(theta, 1)
# # param_qc.cx(0,1)
# print(param_qc.draw(style="iqp"))
# parameter_values = [[0], [np.pi / 6], [np.pi / 2]]
#
# job = estimator.run([param_qc] * 3, [observable] * 3, parameter_values=parameter_values)
# values = job.result().values
#
# for i in range(3):
#     print(f"Parameter: {parameter_values[i][0]:.5f}\t Expectation value: {values[i]}")

if __name__ == '__main__':

    edge_with_weights = [(0, 1, 1),
                         (0, 2, 1.41),
                         (0, 3, 2.23),
                         (1, 2, 1),
                         (1, 3, 1.41),
                         (2, 3, 1)]

    A = 1000
    B = 1000
    tsp = TSP(edge_with_weights, A, B)

    qubit_op, offset = tsp.to_ising

    gamma = Parameter('gamma')
    beta = Parameter('beta')

    H_z_p, H_z = tsp.get_pair_coeff_gate()
    qreg_q = QuantumRegister(9, 'q')
    creg_c = ClassicalRegister(9, 'c')
    qc = QuantumCircuit(qreg_q, creg_c)

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
    qc.rx(2 * beta, qreg_q)
    p_qc = 3



    parameter_values = [[0, 0], [np.pi / 6, np.pi / 6], [np.pi / 2, np.pi / 2]]

    estimator = Estimator()
    job = estimator.run([qc] * 3, [qubit_op] * 3, parameter_values=parameter_values)
    values = job.result().values
    for i in range(3):
        print(f"Parameter: {parameter_values[i][0]:.5f}\t Expectation value: {values[i]}")

    # print("Offset:", offset)
    # print("Qubit operators:", qubit_op)
    # paulis = qubit_op.paulis
    # coeffs = qubit_op.coeffs
    # for i in range(len(paulis)):
    #     print(paulis[i], "\t", coeffs[i])

    # fx = tsp.get_pair_coeff_var()
    #
    # solutions = generate_binary_string(length=9)
    # print(len(solutions))
    #
    # solution_cost = dict()
    # for solution in solutions:
    #     cost = calculate_cost(fx, solution)
    #     solution_cost[solution] = round(cost, 6)
    #     # print(solution, cost)
    # solution_cost = sorted(solution_cost.items(), key=lambda item: item[1])
    # solution_cost = {i[0]: i[1] for i in solution_cost}
    # for k, v in solution_cost.items():
    #     print("cost of state", k,":", v)

    # solution = "010001100"
    # order = make_order(solution)
    # print(order)
    # tsp.draw_tsp_solution(order)



