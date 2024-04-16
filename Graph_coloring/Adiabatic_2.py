from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
from hamiltonian import *
import pandas as pd

backend = Aer.get_backend('qasm_simulator')

def H(gcp, T, no_shots):
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
    for iter in [1, T]:
        print('iter:', iter, 'delta_t:', delta_t)
        t = 1
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
        solutions = find_solution(qc, no_shots)
        states.append([iter, solutions])
        # print(qc)
    return states

def find_solution(circuit, no_shots):
    job = execute(circuit, backend, seed_simulator=10, shots=no_shots)  # NUM_SHOTS
    result = job.result()
    solutions = result.get_counts(circuit)
    solutions = inversion_affichage(solutions)  # Reverse qubit order
    return solutions


def run(gcp, T, no_shots, P, C):
    states = H(gcp, T, no_shots)
    fx = gcp.get_pair_coeff_var()

    results = compare_cost_by_iter(states, fx, len(gcp.nodes), gcp.edges, P, C)
    solutions = states[-1][-1]

    fig_counted = plot_histogram(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_proba = plot_distribution(solutions, title="Qasm Distribution", figsize=(10, 5))
    fig_counted.savefig('Adiabatic/histogram_optimal.png', bbox_inches='tight')
    fig_proba.savefig('Adiabatic/distribution_optimal.png', bbox_inches='tight')

    return results


def create_chart(results, is_export_data=True, save_fig=False):
    # results = [[iter, distribution_no_colors, distribution_cost],..]
    data_distribution_no_color = dict()
    data_distribution_cost = dict()
    data_average_cost = dict()
    average_costs = []
    iters = []
    for result_iter in results:
        print("result_iter:", result_iter)
        iter = result_iter[0]
        iters.append(iter)
        distribution_no_colors = result_iter[1]
        data_distribution_no_color['no_colors'] = list(distribution_no_colors.keys())
        data_distribution_no_color["iter "+str(iter)] = list(distribution_no_colors.values())

        distribution_cost = result_iter[2]
        data_distribution_cost['cost'] = list(distribution_cost.keys())
        data_distribution_cost["iter "+str(iter)] = list(distribution_cost.values())

    data_average_cost['iters'] = iters
    # data_average_cost['average_costs'] = average_costs

    # cumulative_distribution_no_color = calculate_cumulative_prob(data_distribution_no_color)
    # cumulative_distribution_cost = calculate_cumulative_prob(data_distribution_cost)

    if is_export_data:
        df = pd.DataFrame.from_dict(data_distribution_no_color)
        df.to_csv('Adiabatic/result_distribution_no_color.csv', index=False, sep='\t')
        print("Adiabatic/result_distribution_no_color.csv file created successfully!")

        df = pd.DataFrame.from_dict(data_distribution_cost)
        df.to_csv('Adiabatic/result_distribution_cost.csv', index=False, sep='\t')
        print("Adiabatic/result_distribution_cost.csv file created successfully!")

        # df = pd.DataFrame.from_dict(cumulative_distribution_no_color)
        # df.to_csv('Adiabatic/result_cumulative_distribution_no_color.csv', index=False, sep='\t')
        # print("Adiabatic/result_cumulative_distribution_no_color.csv file created successfully!")
        #
        # df = pd.DataFrame.from_dict(cumulative_distribution_cost)
        # df.to_csv('Adiabatic/result_cumulative_distribution_cost.csv', index=False, sep='\t')
        # print("Adiabatic/result_cumulative_distribution_cost.csv file created successfully!")

        # df = pd.DataFrame.from_dict(data_average_cost)
        # df.to_csv('Adiabatic/result_average_cost.csv', index=False, sep='\t')
        # print("Adiabatic/result_average_cost.csv file created successfully!")

    if save_fig:
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


    # # cummulative proba
    # plt.clf()
    # plt.figure(figsize=(8, 6))  # Set chart size
    # keys = list(cumulative_distribution_no_color.keys())[1:]
    # df = pd.DataFrame(cumulative_distribution_no_color)
    # begin = keys[0]
    # final = keys[1]
    # plt.plot(df['no_colors'], df[begin], label='Begin', marker='o', linestyle='-')
    # plt.plot(df['no_colors'], df[final], label='Final', marker='s', linestyle='--')
    # plt.title('Cumulative probability of colors')
    # plt.xlabel('Colors')
    # plt.ylabel('Cumulative probability')
    # plt.xticks(df['no_colors'],rotation=45)  # Rotate x-axis labels for better readability
    # plt.tight_layout()  # Adjust spacing for labels
    # plt.legend()
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('Adiabatic/Cumulative probability of colors.PNG')
    #
    # plt.clf()
    # plt.figure(figsize=(8, 6))  # Set chart size
    # keys = list(cumulative_distribution_cost.keys())[1:]
    # df = pd.DataFrame(cumulative_distribution_cost)
    # begin = keys[0]
    # final = keys[1]
    # plt.plot(df['cost'], df[begin], label='Begin', marker='o', linestyle='-')
    # plt.plot(df['cost'], df[final], label='Final', marker='s', linestyle='--')
    # plt.title('Cumulative probability of cost')
    # plt.xlabel('Cost')
    # plt.ylabel('Cumulative probability')
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.tight_layout()  # Adjust spacing for labels
    # plt.legend()
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('Adiabatic/Cumulative probability of cost.PNG')



if __name__ == '__main__':
    # make a graph
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    K = 3
    A = 10000
    P = A
    C = 100
    no_shots = 2048
    T = 10

    gcp = Graph_Coloring(edges,K=K, A=A, node_size=500, show_graph=False, save_graph=True)
    results = run(gcp, T, no_shots, P=P, C=C)
    create_chart(results, is_export_data=True, save_fig=False)