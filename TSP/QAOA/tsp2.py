from docplex.mp.model import Model
import math
import networkx as nx
import re
import matplotlib.pyplot as plt


class TSP_Model:
    def __init__(self, weight: list[list[float]] | list[list[int]], A: float | int, B: float | int,
                 model_name: str = "TSP_model"):
        self.model = Model(name=model_name)
        self.weight = weight
        self.A = A
        self.B = B
        self.x = self.x_variable()
        self.Z = self.Z_gate()
        self.cost_dist = self.cost_dist()
        # self.cost_start = self.cost_start()
        # self.cost_end = self.cost_end()
        self.cost_penalty_1 = self.penalty_1()
        self.cost_penalty_2 = self.penalty_2()
        self.cost_function = self.get_cost_function()
        self.Hamiltonian = self.get_Hamiltonian()


    def x_variable(self):
        n = len(self.weight)
        return self.model.continuous_var_dict([(i, j) for i in range(n) for j in range(n)], name="x")

    def Z_gate(self):
        n = len(self.weight)
        return self.model.continuous_var_dict([(i, j) for i in range(n) for j in range(n)], name="Z")

    def cost_dist(self):
        n = len(self.weight)
        cost_dist = self.model.sum(
            self.weight[i][j] * self.x[i, p] * self.x[j,(p + 1) % n]
            for i in range(n)
            for j in range(n)
            for p in range(n)
            if i != j
        )
        hamiltonian_cost_dist = self.model.sum(
            self.weight[i][j] * (1 - self.Z[i, p]) * (1 - self.Z[j, (p + 1) % n]) / 4
            for i in range(n)
            for j in range(n)
            for p in range(n)
            if i != j
        )
        return cost_dist, hamiltonian_cost_dist
    def penalty_1(self):
        n = len(self.weight)
        penalty_1 = self.A * self.model.sum(
            (1 - self.model.sum(self.x[i, p] for i in range(n))) ** 2
            for p in range(n)
        )
        Hamiltonian_penalty_1 = self.A * self.model.sum(
            (1 - self.model.sum((1 - self.Z[i, p]) / 2 for i in range(0, n))) ** 2
            for p in range(0, n)
        )
        return penalty_1, Hamiltonian_penalty_1

    def penalty_2(self):
        n = len(self.weight)
        penalty_2 = self.B * self.model.sum(
            (1 - self.model.sum(self.x[i, p] for p in range(0, n))) ** 2
            for i in range(0, n)
        )
        Hamiltonian_penalty_2 = self.B * self.model.sum(
            (1 - self.model.sum((1 - self.Z[i, p]) / 2 for p in range(0, n))) ** 2
            for i in range(0, n)
        )
        return penalty_2, Hamiltonian_penalty_2

    def get_cost_function(self):
        return self.cost_dist[0] + self.cost_penalty_1[0] + self.cost_penalty_2[
            0]

    def get_Hamiltonian(self):
        return self.cost_dist[1] + self.cost_penalty_1[1] + self.cost_penalty_2[
            1]

    def get_pair_coeff_var(self):
        n = len(self.weight)
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
                    Z.append(i * n + j)
                fx.append([w, Z])
        fx.append(H_z)
        return fx

    def get_pair_coeff_gate(self):
        n = len(self.weight)
        # print(self.Hamiltonian)
        str_cost = self.Hamiltonian.repr_str().replace("+-", "-").replace("-", "+-")
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
        # print(H_z)
        H_z_p = []
        for e in elements[:-1]:
            w = 1.0
            Z = []
            match = re.search(r'^[^Z]*Z', e)
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
                    Z.append(i * n + j)
            if len(Z) == 2:
                if Z[0] == Z[1]:
                    H_z += w
                else:
                    H_z_p.append([w, Z])
            else:
                H_z_p.append([w, Z])

        # print(H_z_p)
        # print(H_z)

        # for i in range(len(H_z_p)):
        #     H_z_p[i][0] = H_z_p[i][0] / H_z

        return H_z_p, H_z


class TSP(TSP_Model):
    def __init__(self,
                 edge_with_weights,
                 A,
                 B,
                 node_size=500,
                 show_graph=True,
                 save_graph=True):
        self.edge_weights = edge_with_weights
        self.A = A
        self.B = B
        self.node_size = node_size
        self.show_graph = show_graph
        self.save_graph = save_graph
        self.TSP_graph = self.make_graph()
        self.weights = self.make_weights_matrix()
        super().__init__(self.weights, A, B)

    def make_graph(self):
        edges = []
        for weight in self.edge_weights:
            edges.append((str(weight[0]), str(weight[1]), weight[2]))

        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        pos = nx.spring_layout(G)
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=self.node_size)
        nx.draw_networkx_edges(G, pos)

        # Get edge weights as a dictionary
        edge_labels = nx.get_edge_attributes(G, "weight")

        # Draw edge labels with positions
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        # Add labels to nodes
        nx.draw_networkx_labels(G, pos)  # Uses node names by default

        if self.show_graph:
            plt.axis('off')
            plt.show()

        if self.save_graph:
            plt.savefig("Graph.png", format="PNG")
        return G

    def make_weights_matrix(self):
        n = len(self.TSP_graph.nodes)
        weights = [[0 for _ in range(n)] for _ in range(n)]
        for edge in self.edge_weights:
            i = edge[0]
            j = edge[1]
            w = edge[2]
            weights[i][j] = w
            weights[j][i] = w
        return weights


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
def filter_solution(counts, total, threshold=0.001):
    solutions = dict()
    for solution, counted in counts.items():
        if counted / total > threshold:
            solutions[solution] = counted
    sorted_by_value = sorted(solutions.items(), key=lambda item: item[1], reverse=True)
    solutions = {i[0]: i[1] for i in sorted_by_value}
    return solutions

def find_best_solution(solutions, fx):
    best_solution=[]
    max_value = max(solutions.values())
    for k, v in solutions.items():
        if v == max_value:
            cost = calculate_cost(fx, k)
            best_solution.append([k, v, cost])
    return best_solution

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


if __name__ == '__main__':
    # edge_with_weights = [(0, 1, 48), (0, 2, 91), (1,2,63)]
    # edge_with_weights = [(0, 1, 1), (0, 2, 2), (0, 3, 1), (1, 2, 2), (1, 3, 3), (2, 3, 3)]
    edge_with_weights = [(0, 1, 1), (0, 2, 1.41), (0, 3, 2.23), (1, 2, 1), (1, 3, 1.41), (2, 3, 1)]
    # A = sum([i[2]*2 for i in edge_with_weights])
    # B = A
    # print(A, B)
    A = 1213
    B = 1213
    tsp = TSP(edge_with_weights, A, B, show_graph=False, save_graph=True)
    print(tsp.cost_dist[0])
    print(tsp.cost_penalty_1[0])
    print(tsp.cost_penalty_2[0])
    print(tsp.get_cost_function())
    print()
    print(tsp.cost_dist[1])
    print(tsp.cost_penalty_1[1])
    print(tsp.cost_penalty_2[1])
    print(tsp.Hamiltonian)
    H_z_p, H_z = tsp.get_pair_coeff_gate()
    print(H_z)
    for gate in H_z_p:
        print(gate)
    # fx = tsp.get_pair_coeff_var()

    # solutions = generate_binary_string(length=9)
    # print(len(solutions))
    # for solution in solutions:
    #     print(solution, calculate_cost(fx, solution))
    # solution = "010001100"
    # order = make_order(solution)
    # print(order)
    # tsp.draw_tsp_solution(order)


