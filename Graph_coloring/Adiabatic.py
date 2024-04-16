from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
from hamiltonian import *
import pandas as pd

backend = Aer.get_backend('qasm_simulator')


def H(gcp, T, no_shots, show_iter=False):
    """
    Create quantum circuit of Hamiltonian by H_P, which is created in gcp object.

    :param gcp: cp object
    :type gcp: gcp Object
    :param T: number of iterations
    :type T: int
    :return: quantum circuit
    :rtype:
    """
    H_z_p, H_z = gcp.get_pair_coeff_gate()

    no_qubits = len(gcp.nodes) * gcp.K
    bit_strings = generate_binary_string(no_qubits)

    delta_t = 1 / T
    num_str = str(delta_t)
    split_num = num_str.split(".")
    num_decimals = len(split_num[1])

    states = []

    if show_iter:
        start = 1
        end = T+1
        step = T-1
    else:
        start = T
        end = T+1
        step = 1

    for iter in range(start, end+step, step):
        print('iter:', iter, 'delta_t:', delta_t)
        t = 1
        qreg_q = None
        creg_c = None
        qc = None
        qreg_q = QuantumRegister(no_qubits, 'q')
        creg_c = ClassicalRegister(no_qubits, 'c')
        qc = QuantumCircuit(qreg_q, creg_c)
        qc.reset(qreg_q)
        qc.h(qreg_q)  # Apply Hadamard gate
        qc.barrier()

        for _ in range(iter):
            qc.rx(2 * t, qreg_q)
            for h in H_z_p:
                w = h[0] / H_z
                idx = h[1]
                if len(idx) == 1:
                    qc.rz(2 * (1-t) * w, qreg_q[idx[0]])
                else:
                    i = idx[0]
                    j = idx[1]
                    qc.cx(qreg_q[i], qreg_q[j])
                    qc.rz(2 * (1-t) * w, qreg_q[j])
                    qc.cx(qreg_q[i], qreg_q[j])
            t = round(t - delta_t, num_decimals)
        qc.measure(qreg_q, creg_c)
        solutions = find_solution(qc, no_shots, bit_strings)
        states.append([iter, solutions])
        # print(qc)
    return states


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
    # for state in bit_strings:
    #     if state not in solutions:
    #         solutions[state] = 0
    return solutions



def run(gcp, T, no_shots, show_iter, P, C):
    states = H(gcp, T, no_shots, show_iter)
    fx = gcp.get_pair_coeff_var()

    results = compare_cost_by_iter(states, fx, len(gcp.nodes), gcp.edges, P, C)
    solutions = states[-1][-1]
    redundants = []

    for k, v in solutions.items():
        if v == 0:
            redundants.append(k)
    for k in redundants:
        del solutions[k]

    fig_counted = plot_histogram(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('Adiabatic/histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('Adiabatic/distribution_optimal.png', bbox_inches='tight')


    # tmp = {i[0]:i[1] for i in sorted(states[-1][-1].items(), key = lambda x: x[1], reverse=True)}
    # len(gcp.nodes), gcp.A, gcp.edges
    distribution = dict()
    for state, shot in solutions.items():
        no_conflict = calculate_no_conflict(state, len(gcp.nodes), gcp.edges)
        no_colors = count_color(state, len(gcp.nodes))
        cost_by_state = P * penalty_part(state, len(gcp.nodes)) + C * no_conflict + no_colors
        prob = shot / no_shots * 100
        distribution[state] = [cost_by_state, prob]
        # print('state:',state, 'cost:', cost_by_state, 'prob', prob)

    distribution = {i[0]:i[1] for i in sorted(distribution.items(), key = lambda x: x[1][0], reverse=True)}
    # solutions = solutions
    print("-"*20)
    no_states = len(distribution.keys())
    print(f"Have {no_states:d} states")
    prob_is_solutions = 0
    prob_has_K_colors = 0
    no_state_has_K_colors = 0
    no_state_is_solution = 0
    for state, v in distribution.items():
        cost = v[0]
        prob_by_simulator = v[1]
        if has_K_color(state, K, edges, len(gcp.nodes)):
            no_state_has_K_colors+=1
            prob_has_K_colors+=prob_by_simulator
            if is_solution(state, K, len(gcp.nodes)):
                no_state_is_solution+=1
                prob_is_solutions += prob_by_simulator
                print(state, cost, prob_by_simulator)
    # print()
    # print("no_state_has_K_colors is", no_state_has_K_colors)
    # print(f"prob_state_has_K_colors is {prob_has_K_colors:.2f}% in all states (captured)")
    # print(f"prob_state_has_K_colors is {no_state_has_K_colors/len(distribution.keys()):.4f}% (all states have same probability)")
    #
    # print()
    # print("no_state_is_solution is", no_state_is_solution)
    # print(f"prob_is_solutions is {prob_is_solutions:.2f}% in all states (captured)")
    # print(f"prob_is_solutions is {no_state_is_solution/len(distribution.keys()):.4f}% (all states have same probability)")
    #
    # print()
    # print(
    #     f"Probability of 1 solution is {prob_is_solutions / prob_has_K_colors:.4f}% in all states have 3 colors (captured)")
    # print(
    #     f"Probability of 1 solution is {no_state_is_solution / no_state_has_K_colors * 100:.2f}% in all states have 3 colors (all states have same probability)")

    # print(prob_is_solutions)

    return results


def create_chart(results, is_export_data=True):
    # results = [[iter, energy, distribution_no_colors, distribution_cost],..]
    data_distribution_no_color = dict()
    data_distribution_cost = dict()
    data_average_cost = dict()
    average_costs = []
    iters = []
    for result_iter in results:
        iter = result_iter[0]
        iters.append(iter)
        average_cost = result_iter[1]
        average_costs.append(average_cost)
        distribution_no_colors = result_iter[2]
        data_distribution_no_color['no_colors'] = list(distribution_no_colors.keys())
        data_distribution_no_color["iter "+str(iter)] = list(distribution_no_colors.values())

        distribution_cost = result_iter[3]
        data_distribution_cost['cost'] = list(distribution_cost.keys())
        data_distribution_cost["iter "+str(iter)] = list(distribution_cost.values())

    data_average_cost['iters'] = iters
    data_average_cost['average_costs'] = average_costs

    cumulative_distribution_no_color = calculate_cumulative_prob(data_distribution_no_color)
    cumulative_distribution_cost = calculate_cumulative_prob(data_distribution_cost)

    if is_export_data:
        df = pd.DataFrame.from_dict(data_distribution_no_color)
        df.to_csv('Adiabatic/result_distribution_no_color.csv', index=False, sep='\t')
        print("Adiabatic/result_distribution_no_color.csv file created successfully!")

        df = pd.DataFrame.from_dict(data_distribution_cost)
        df.to_csv('Adiabatic/result_distribution_cost.csv', index=False, sep='\t')
        print("Adiabatic/result_distribution_cost.csv file created successfully!")

        df = pd.DataFrame.from_dict(cumulative_distribution_no_color)
        df.to_csv('Adiabatic/result_cumulative_distribution_no_color.csv', index=False, sep='\t')
        print("Adiabatic/result_cumulative_distribution_no_color.csv file created successfully!")

        df = pd.DataFrame.from_dict(cumulative_distribution_cost)
        df.to_csv('Adiabatic/result_cumulative_distribution_cost.csv', index=False, sep='\t')
        print("Adiabatic/result_cumulative_distribution_cost.csv file created successfully!")

        df = pd.DataFrame.from_dict(data_average_cost)
        df.to_csv('Adiabatic/result_average_cost.csv', index=False, sep='\t')
        print("Adiabatic/result_average_cost.csv file created successfully!")

    plt.clf()
    plt.figure(figsize=(8, 6))# Set chart size
    keys = list(data_distribution_no_color.keys())[1:]
    df = pd.DataFrame(data_distribution_no_color)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['no_colors'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['no_colors'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Probability of colors')
    plt.xlabel('Colors')
    plt.ylabel('Probability')
    plt.xticks(df['no_colors'],rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('Adiabatic/Probability of colors.PNG')

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
    plt.savefig('Adiabatic/Probability of cost.PNG')


    # cummulative proba
    plt.clf()
    plt.figure(figsize=(8, 6))  # Set chart size
    keys = list(cumulative_distribution_no_color.keys())[1:]
    df = pd.DataFrame(cumulative_distribution_no_color)
    begin = keys[0]
    final = keys[1]
    plt.plot(df['no_colors'], df[begin], label='Begin', marker='o', linestyle='-')
    plt.plot(df['no_colors'], df[final], label='Final', marker='s', linestyle='--')
    plt.title('Cumulative probability of colors')
    plt.xlabel('Colors')
    plt.ylabel('Cumulative probability')
    plt.xticks(df['no_colors'],rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust spacing for labels
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('Adiabatic/Cumulative probability of colors.PNG')

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
    plt.savefig('Adiabatic/Cumulative probability of cost.PNG')



if __name__ == '__main__':
    # make a graph
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 10000
    P = A
    C = 100
    no_shots = 2048
    T = 100



    gcp = Graph_Coloring(edges,K=K, A=A, node_size=500, show_graph=False, save_graph=True)
    results = run(gcp, T, no_shots, show_iter=False, P=P, C=C)
    # create_chart(results, is_export_data=True)