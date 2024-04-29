from docplex.mp.model import Model
import math
import networkx as nx
import re
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np


class Graph_Coloring:
    def __init__(self,
                 edges,
                 K,
                 A: float | int,
                 node_size=500,
                 show_graph=False,
                 save_graph=True):

        self.edges = edges
        self.K = K
        self.A = A
        self.node_size = node_size
        self.show_graph = show_graph
        self.save_graph = save_graph
        self.graph = self.make_graph()
        self.nodes = self.get_nodes()
        self.model = Model(name="Graph Coloring")
        self.model.float_precision = len(str(A).replace('.', ''))
        self.Z = self.Z_gate
        self.Hamiltonian = self.create_hamiltonian()

    def get_nodes(self):
        nodes = set()
        for i, j in self.edges:
            nodes.add(i)
            nodes.add(j)
        return nodes

    @property
    def Z_gate(self) -> Model:
        no_nodes = len(self.nodes)
        return self.model.continuous_var_dict([(i, k) for k in range(self.K) for i in range(no_nodes)], name="Z")

    def create_hamiltonian(self) -> (Model, Model):
        hamiltonian = 0
        for k in range(self.K):
            hamiltonian += self.model.sum(
                (1 / 4) * (1 - self.Z[i, k]) * (1 - self.Z[j, k]) + (1 / 4) * (1 - self.Z[j, k]) * (1 - self.Z[i, k])
                for i, j in self.edges
            )
        penalty = 0
        for i in self.nodes:
            b = self.model.sum((1 - self.Z[i, k]) / 2 for k in range(self.K))
            penalty += (1 - b) ** 2

        hamiltonian = hamiltonian + self.A * penalty

        return hamiltonian

    def simply_Hamiltonian(self) -> Model:
        H_z_p, offset = self.get_pair_coeff_gate()

        Z = self.model.continuous_var_dict([(i) for i in range(len(self.edges) * self.K)], name="Z")
        linear_part = [i for i in H_z_p if len(i[1]) == 1]
        quadratic_part = [i for i in H_z_p if len(i[1]) == 2]

        H_linear_part = self.model.sum(
            i[0] * Z[i[1][0]]
            for i in linear_part
        )

        H_quadratic_part = self.model.sum(
            i[0] * Z[i[1][0]] * Z[i[1][1]]
            for i in quadratic_part
        )
        return H_linear_part + H_quadratic_part + offset

    def get_pair_coeff_gate(self, H=None) -> (list, float):
        if H is None:
            H = self.Hamiltonian

        str_cost = H.repr_str().replace("+-", "-").replace("-", "+-")
        elements = str_cost.split("+")

        while "" in elements:
            elements.remove("")

        for i in range(len(elements)):
            e = elements[i]
            if "^2" in e:
                match = re.search(r'^[^Z]+Z', e, )
                # print(match)
                if match:
                    i_cut = match.end() - 1
                    if e[:i_cut].replace('.', '', 1).replace('-', '').isdigit() or e[:i_cut] == "-":
                        elements[i] = e[:i_cut] + e[i_cut:-2] + "*" + e[i_cut:-2]
                    else:
                        elements[i] = e[:-2] + "*" + e[:-2]
                else:
                    elements[i] = e[:-2] + "*" + e[:-2]
            # print(e, "->", elements[i])

        H_z = float(elements[-1])
        H_z_p = []
        for e in elements[:-1]:
            coeff = 1.0
            op = ''
            # split coeff with gate
            match = re.search(r'^[^Z]*Z', e)

            if match:
                i_cut = match.end() - 1
                coeff = e[:i_cut]
                op = e[i_cut:]
                if coeff != "":
                    if '-' == coeff:
                        coeff = -1.0
                    else:
                        coeff = float(coeff)
                else:
                    coeff = 1

            else:
                coeff = 1

            ops = op.split("*")
            Z = []
            for o in ops:
                i_j = o.split("_")[1:]
                i = int(i_j[0])
                j = int(i_j[1])
                Z.append(i * self.K + j)

            if len(Z) == 2:
                if Z[0] == Z[1]:
                    H_z += coeff
                    continue
            H_z_p.append([coeff, Z])

        linear_part = [e for e in H_z_p if len(e[1]) == 1]
        linear_part = sorted(linear_part, key=lambda x: x[1][0])
        # print(linear_part)
        quadra_part = [e for e in H_z_p if len(e[1]) == 2]
        quadra_part = sorted(quadra_part, key=lambda x: (x[1][0], x[1][1]))
        # print(quadra_part)
        H_z_p = quadra_part + linear_part
        return H_z_p, H_z

    @property
    def to_ising(self) -> (SparsePauliOp, float):

        H_z_p, offset = self.get_pair_coeff_gate()
        num_vars = len(self.Z)
        pauli_list = []
        zero = np.zeros(num_vars, dtype=bool)

        for gate in H_z_p:
            weight = gate[0]
            z_p = zero.copy()
            if len(gate[1]) == 1:
                idx = gate[1]
                z_p[idx] = True
                pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), weight))
            else:
                i = gate[1][0]
                j = gate[1][1]
                z_p[i] = True
                z_p[j] = True
                pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), weight))

        if pauli_list:
            # Remove paulis whose coefficients are zeros.
            qubit_op = sum(pauli_list).simplify(atol=0)
        else:
            # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
            # If num_nodes=0, I^0 = 1 (int).
            num_vars = max(1, num_vars)
            qubit_op = SparsePauliOp("I" * num_vars, 0)

        return qubit_op, offset

    def make_graph(self) -> nx.Graph():
        edges = [(str(e[0]), str(e[1])) for e in self.edges]
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=self.node_size)
        nx.draw_networkx_edges(G, pos)
        # Add labels to nodes
        nx.draw_networkx_labels(G, pos)  # Uses node names by default

        if self.show_graph:
            plt.axis('off')
            plt.show()
        if self.save_graph:
            plt.savefig("Graph.png", format="PNG")

        return G


def generate_binary_string(length, current_string=""):
    """
    Generates all possible binary strings of a given length using recursion.

    Args:
        length: The desired length of the binary string.
        current_string: The current binary string being built (internal use).

    Returns:
        A list containing all possible binary strings of the given length.
    """
    if length == 0:
        return [current_string]
    else:
        # Append 0 and 1 to the current string and call recursively
        return generate_binary_string(length - 1, current_string + "0") + \
            generate_binary_string(length - 1, current_string + "1")


def prettyprint(cost_function: Model):
    str_cost = cost_function.repr_str().replace("+-", "-").replace("-", "+-")
    elements = str_cost.split("+")
    while "" in elements:
        elements.remove("")
    f = ""
    no_element_inline = 5
    for i in range(0, len(elements), no_element_inline):
        f += " + ".join(elements[i: i + no_element_inline]) + "\n"
    print(f)


if __name__ == '__main__':
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 10
    gcp = Graph_Coloring(edges, K, A)

    print("Hamiltonian:", gcp.Hamiltonian)
    # print()
    # prettyprint(gcp.Hamiltonian)

    # print("Hamiltonian simplification:")
    # prettyprint(gcp.simply_Hamiltonian())
    gates, offset = gcp.get_pair_coeff_gate()
    print(offset)
    for gate in gates:
        print(gate)
