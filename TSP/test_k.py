# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

def draw_tsp_solution(G, order, colors, pos):
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)


# Generating a graph of 3 nodes
n = 3
num_qubits = n**2
tsp = Tsp.create_random_instance(n, seed=123)
adj_matrix = nx.to_numpy_array(tsp.graph)
print("distance\n", adj_matrix)

colors = ["r" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]

# draw_graph(tsp.graph, colors, pos)
# plt.show()

qp = tsp.to_quadratic_program()
print(qp.prettyprint())

qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
print(qubo.prettyprint())


def to_ising(quad_prog):
    num_vars = quad_prog.get_num_vars()
    pauli_list = []
    offset = 0.0
    zero = np.zeros(num_vars, dtype=bool)

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sense = quad_prog.objective.sense.value
    print(sense)

    # convert a constant part of the objective function into Hamiltonian.
    offset += quad_prog.objective.constant * sense
    print(offset)
    linear_part = quad_prog.objective.linear.to_dict().items()
    print(linear_part)
    # convert linear parts of the objective function into Hamiltonian.
    for idx, coef in linear_part:
        z_p = zero.copy()
        weight = coef * sense / 2
        z_p[idx] = True
        pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))
        offset += weight

    quadratic_part = quad_prog.objective.quadratic.to_dict().items()
    print(quadratic_part)
    # create Pauli terms
    for (i, j), coeff in quadratic_part:

        weight = coeff * sense / 4
        print((i, j), weight)
        if i == j:
            offset += weight
        else:
            z_p = zero.copy()
            z_p[i] = True
            z_p[j] = True
            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), weight))
            print(pauli_list[-1])

        z_p = zero.copy()
        z_p[i] = True
        pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))
        print(pauli_list[-1])

        z_p = zero.copy()
        z_p[j] = True
        pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))
        print(pauli_list[-1])

        offset += weight

    if pauli_list:
        # Remove paulis whose coefficients are zeros.
        qubit_op = sum(pauli_list).simplify(atol=0)
    else:
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        num_vars = max(1, num_vars)
        qubit_op = SparsePauliOp("I" * num_vars, 0)

    return qubit_op, offset


# qubitOp, offset = qubo.to_ising()
qubitOp, offset = to_ising(qubo)
# print(qubitOp, offset)
print(offset)
print(qubitOp.coeffs)
print(qubitOp)

exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())

eigen_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubitOp)
print(eigen_result)
result = exact.solve(qubo)
print(result.prettyprint())
