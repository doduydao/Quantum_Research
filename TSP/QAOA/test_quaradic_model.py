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

from qiskit_algorithms.optimizers import COBYLA

# Generating a graph of 3 nodes
n = 4
num_qubits = n**2
tsp = Tsp.create_random_instance(n, seed=123)

adj_matrix = nx.to_numpy_array(tsp.graph)
print("distance\n", adj_matrix)

colors = ["r" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
# draw_graph(tsp.graph, colors, pos)

qp = tsp.to_quadratic_program()
# print(qp.prettyprint())

from qiskit_optimization.converters import QuadraticProgramToQubo

qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
print(qubo)
qubitOp, offset = qubo.to_ising()
# print("Offset:", offset)
# print("Ising Hamiltonian:")
# print(len(qubitOp.coeffs))

from qiskit_algorithms import QAOA, NumPyMinimumEigensolver

qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
qaoa = MinimumEigenOptimizer(qaoa_mes)
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = qaoa.solve(qubo)
print(result.prettyprint())