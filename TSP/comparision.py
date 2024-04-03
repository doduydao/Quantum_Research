import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import seaborn.objects as so
def read_data(file_name):
    costs = []
    initial_prob = []
    final_prob = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split("\t")
            line = [i.replace(",", ".") for i in line]
            costs.append(float(line[0]))
            initial_prob.append(float(line[1]))
            final_prob.append(float(line[2]))

    return costs, initial_prob, final_prob

def line_plot(costs, initial_prob, final_prob):

    data = {'x': costs, 'Series 1': initial_prob, 'Series 2': final_prob}
    # plt.plot(costs, initial_prob, label="Initial probability result")
    # plt.plot(costs, final_prob, label="Final probability result")
    # sns.kdeplot(data=data, x="Series 1", multiple="stack")
    # sns.kdeplot(data=data, x='Series 2', multiple="stack")
    # plt.scatter(costs, initial_prob, label="Initial probability result")
    # plt.scatter(costs, final_prob, label="Final probability result")

    plt.stackplot(costs,
                  initial_prob, final_prob,
                  labels=["Initial probability result", "Final probability result"],
                  colors=["#4472c4", "#ed7d31"],
                  alpha=0.8)

    # plt.xticks([i for i in range(int(min(costs)), int(max(costs)), 100)], costs)
    plt.xlabel("Cost")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("line_plot.png")
    plt.show()



def box_plot(costs, initial_prob, final_prob):
    plt.boxplot(initial_prob, positions=[1], labels=["Initial probability result"])
    plt.boxplot(final_prob, positions=[2], labels=["Final probability result"])
    plt.xlabel("Cost")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("box_plot.png")
    plt.show()



def desity_plot(x, y, z):
    # sns.kdeplot(data=[y], x="total_bill", hue="time", multiple="stack")
    sns.kdeplot(z, label="Initial probability result", multiple="stack")
    sns.kdeplot(y, label="Final probability result", multiple="stack")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("desity_plot.png")
    plt.show()


def calculate_cumulative(probs):
    cumulative = [probs[0]]
    for i in range(1, len(probs)):
        accumlate_proba = cumulative[i - 1] + probs[i]
        cumulative.append(accumlate_proba)
    return cumulative


def cumulative_distribution(x, y ,z):
    cdf_y = calculate_cumulative(y)
    cdf_z = calculate_cumulative(z)

    # plt.plot(x, cdf_y, label="Initial probability result")
    # plt.plot(x, cdf_z, label="Final probability result")

    plt.scatter(x, cdf_y, label="Initial probability result")
    plt.scatter(x, cdf_z, label="Final probability result")
    plt.xlabel("Cost")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("cumulative_distribution.png")
    plt.show()


def violin_plot(x, y, z):
    data = {'cost': costs, 'Series 1': initial_prob, 'Series 2': final_prob}
    # df = sns.load_dataset("titanic")
    # print(df)
    sns.violinplot(data=data, x="Series 1")
    # sns.violinplot(data=data, x="Series 2")
    plt.xlabel("Cost")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("violin_plot.png")
    plt.show()



def calculation_expectation(costs, probs):
    expectation = []

    for i in range(len(costs)):
        expectation.append(costs[i]*probs[i])
    return expectation


def expectation_plot(costs, initial_prob, final_prob):
    ex_initial = calculation_expectation(costs, initial_prob)
    ex_final = calculation_expectation(costs, final_prob)
    plt.plot(costs, ex_initial, label="Initial probability result")
    plt.plot(costs, ex_final, label="Final probability result")

    plt.xlabel("Cost")
    plt.ylabel("Expectation")
    plt.legend()
    plt.savefig("expectation_plot.png")
    plt.show()


if __name__ == '__main__':
    file_name = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/TSP/comparison.txt"
    costs, initial_prob, final_prob = read_data(file_name)
    line_plot(costs, initial_prob, final_prob)
    # box_plot(costs, initial_prob, final_prob)
    # desity_plot(costs, initial_prob, final_prob)
    cumulative_distribution(costs, initial_prob, final_prob)
    # violin_plot(costs, initial_prob, final_prob)
    expectation_plot(costs, initial_prob, final_prob)

