
from docplex.mp.model import Model
import math
import networkx as nx
import re
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np


class Graph_Coloring_Model:
    def __init__(self, edges: list[tuple[int, int]],
                 K: int,
                 A: float | int,
                 model_name: str = "TSP_model"):
        self.model = Model(name=model_name)
        self.model.float_precision = len(str(A).replace('.', ''))
        self.edges = edges
        self.nodes = self.get_nodes()
        self.K = K
        self.A = A
        self.x = self.x_variable
        self.Z = self.Z_gate
        self.cost_dist = self.cost_dist()
        self.cost_penalty = self.penalty()
        self.cost_function = self.get_cost_function()
        self.Hamiltonian = self.get_Hamiltonian()

    def get_nodes(self):
        nodes = set()
        for i, j in self.edges:
            nodes.add(i)
            nodes.add(j)
        return nodes

    @property
    def x_variable(self) -> Model:

        no_nodes = len(self.nodes)
        return self.model.continuous_var_dict([(i, k) for k in range(self.K) for i in range(no_nodes)], name="x")

    @property
    def Z_gate(self) -> Model:
        no_nodes = len(self.nodes)
        return self.model.continuous_var_dict([(i, k) for k in range(self.K) for i in range(no_nodes)], name="Z")

    def cost_dist(self) -> (Model, Model):
        cost_dist = 0
        hamiltonian_cost_dist = 0
        for k in range(self.K):
            cost_dist += self.model.sum(
                2 * self.x[i, k] * self.x[j, k]
                for i, j in self.edges

            )
            hamiltonian_cost_dist += self.model.sum(
                    (1 - self.Z[i, k]) * (1 - self.Z[j, k]) / 2
                    for i, j in self.edges
                )
        return cost_dist, hamiltonian_cost_dist

    def penalty(self) -> (Model, Model):
        penalty_cost = 0
        Hamiltonian_penalty_cost = 0

        for i in self.nodes:
            a = self.model.sum(self.x[i, k] for k in range(self.K))
            penalty_cost += (1-a)**2

            b = self.model.sum((1 - self.Z[i, k]) / 2 for k in range(self.K))
            Hamiltonian_penalty_cost += (1-b)**2

        return self.A * penalty_cost, self.A * Hamiltonian_penalty_cost

    def get_cost_function(self) -> Model:
        return self.cost_dist[0] + self.cost_penalty[0]

    def get_Hamiltonian(self) -> Model:
        return self.cost_dist[1] + self.cost_penalty[1]

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

    def get_pair_coeff_var(self) -> list:

        str_cost = self.cost_function.repr_str().replace("+-", "-").replace("-", "+-")
        elements = str_cost.split("+")
        while "" in elements:
            elements.remove("")
        for i in range(len(elements)):
            e = elements[i]
            if "^2" in e:
                match = re.search(r'^[^x]+x', e, )
                # print(match)
                if match:
                    i_cut = match.end() - 1
                    if e[:i_cut].replace('.', '', 1).replace('-', '').isdigit() or e[:i_cut] == "-":
                        elements[i] = e[:i_cut] + e[i_cut:-2] + "*" + e[i_cut:-2]
                    else:
                        elements[i] = e[:-2] + "*" + e[:-2]
                else:
                    elements[i] = e[:-2] + "*" + e[:-2]
        # print(elements)
        if 'x' not in elements[-1]:
            H_z = float(elements[-1])
        else:
            H_z = 0
        fx = []
        for e in elements[:-1]:
            w = 1.0
            Z = []
            match = re.search(r'^[^x]*x', e)
            if match:
                i_cut = match.end() - 1
                if e[:i_cut] != "":
                    w = e[:i_cut]
                    if '-' == w:
                        w = -1.0
                    else:
                        w = float(w)
                    e = e[i_cut:]
                Xs = e.split("*")
                for x_i_j in Xs:
                    i_j = x_i_j.split("_")[1:]
                    i = int(i_j[0])
                    j = int(i_j[1])
                    Z.append(i * self.K + j)
                fx.append([w, Z])
        fx.append(H_z)
        return fx

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
            for op in ops:
                i_j = op.split("_")[1:]
                i = int(i_j[0])
                j = int(i_j[1])
                Z.append(i * self.K + j)

            if len(Z) == 2:
                if Z[0] == Z[1]:
                    H_z += coeff
                else:
                    H_z_p.append([coeff, Z])
            else:
                H_z_p.append([coeff, Z])
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


class Graph_Coloring(Graph_Coloring_Model):
    def __init__(self,
                 edges,
                 K,
                 A: float | int,
                 node_size=500,
                 show_graph=False,
                 save_graph=True):
        super().__init__(edges, K, A)
        self.edges = edges
        self.K = K
        self.A = A
        self.node_size = node_size
        self.show_graph = show_graph
        self.save_graph = save_graph
        self.graph = self.make_graph()

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




def filter_solution(counts, total, threshold=1e-6):
    solutions = dict()
    for solution, counted in counts.items():
        if counted / total > threshold:
            solutions[solution] = counted
    # sorted_by_value = sorted(solutions.items(), key=lambda item: item[1], reverse=True)
    # solutions = {i[0]: i[1] for i in sorted_by_value}
    return solutions



def make_order(solutions):
    orders = []
    for solution in solutions:
        solution = solution[0]
        order = [0]
        n = int(math.sqrt(len(solution)))
        for i in range(0, len(solution), n):
            p = 0
            for j, char in enumerate(solution[i:i + n]):
                if char == '1':
                    p = j + 1
            order.append(p)
        orders.append(order)
    return orders


def prettyprint(cost_function: Model):
    str_cost = cost_function.repr_str().replace("+-", "-").replace("-", "+-")
    elements = str_cost.split("+")
    print(elements)
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
    tsp = Graph_Coloring(edges, K, A)
    # print(tsp.weight)

    # print("tsp.cost_dist[0]:")
    # print(tsp.cost_dist[0])
    # prettyprint(tsp.cost_dist[0])
    # print("tsp.cost_penalty[0]:")
    # print(tsp.cost_penalty[0])
    # prettyprint(tsp.cost_penalty[0])
    # print(tsp.cost_penalty_2[0])
    print("Cost function:")
    print(tsp.cost_function)
    # print(tsp.get_pair_coeff_var())

    # print()
    # print(tsp.cost_dist[1])
    # print(tsp.cost_start[1])
    # print(tsp.cost_end[1])
    # print(tsp.cost_penalty_1[1])
    # print(tsp.cost_penalty_2[1])
    print("Hamiltonian:")
    prettyprint(tsp.Hamiltonian)
    print("Hamiltonian simplification:")
    prettyprint(tsp.simply_Hamiltonian())
    # gates, offset = tsp.get_pair_coeff_gate()
    # print(offset)
    # for gate in gates:
    #     print(gate)

    qubit_op, offset = tsp.to_ising
    # print(qubit_op)
    print("Offset:", offset)
    print("Qubit operators:", qubit_op)
