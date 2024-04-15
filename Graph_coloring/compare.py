import pandas as pd
import matplotlib.pyplot as plt

def make_prob_char(data, cum=False):
    if cum != True:
        keys = list(data.columns)
        plt.clf()
        plt.figure(figsize=(8, 6))  # Set chart size

        plt.plot(data[keys[0]], data[keys[1]], label=keys[1], marker='o', linestyle='-')
        plt.plot(data[keys[0]], data[keys[2]], label=keys[2], marker='s', linestyle='--')
        plt.title('Probability of ' + keys[0])
        plt.xlabel(keys[0])
        plt.ylabel('Probability')
        if keys[0] == 'no_colors':
            plt.xticks(data[keys[0]], rotation=45)  # Rotate x-axis labels for better readability
        else:
            plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust spacing for labels
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('Compare/Probability of '+ keys[0]+'.PNG')
    else:
        keys = list(data.columns)
        plt.clf()
        plt.figure(figsize=(8, 6))  # Set chart size

        plt.plot(data[keys[0]], data[keys[1]], label=keys[1], marker='o', linestyle='-')
        plt.plot(data[keys[0]], data[keys[2]], label=keys[2], marker='s', linestyle='--')
        plt.title('Cumulative probability of ' + keys[0])
        plt.xlabel(keys[0])
        plt.ylabel('Probability')
        if keys[0] == 'no_colors':
            plt.xticks(data[keys[0]], rotation=45)  # Rotate x-axis labels for better readability
        else:
            plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust spacing for labels
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig('Compare/Cumulative_probability' + keys[0] + '.PNG')


def make_data_to_compare(d1, d2):
    keys = list(d1.columns)
    key = keys[0]
    iter_keys = keys[1:]
    d1 = d1.drop(iter_keys[0], axis=1)
    d2 = d2.drop(list(d2.columns)[1], axis=1)
    d1.rename(columns={iter_keys[1]: 'AQC'}, inplace=True)
    d2.rename(columns={list(d2.columns)[1]: 'QAOA'}, inplace=True)

    merged_df = d1.merge(d2, on=key, how='left')  # 'left' join by default
    # print(merged_df)
    return merged_df

if __name__ == '__main__':

    QAOA_cost_path = 'QAOA/result_distribution_cost.csv'
    QAOA_color_path = 'QAOA/result_distribution_no_color.csv'
    QAOA_cum_cost_path = 'QAOA/result_cumulative_distribution_cost.csv'
    QAOA_cum_color_path = 'QAOA/result_cumulative_distribution_no_color.csv'
    QAOA_cost = pd.read_csv(QAOA_cost_path, delimiter='\t')
    QAOA_color = pd.read_csv(QAOA_color_path, delimiter='\t')
    QAOA_cum_cost = pd.read_csv(QAOA_cum_cost_path, delimiter='\t')
    QAOA_cum_color = pd.read_csv(QAOA_cum_color_path, delimiter='\t')

    Adiabatic_cost_path = 'Adiabatic/result_distribution_cost.csv'
    Adiabatic_color_path = 'Adiabatic/result_distribution_no_color.csv'
    Adiabatic_cum_cost_path = 'Adiabatic/result_cumulative_distribution_cost.csv'
    Adiabatic_cum_color_path = 'Adiabatic/result_cumulative_distribution_no_color.csv'
    Adiabatic_cost = pd.read_csv(Adiabatic_cost_path, delimiter='\t')
    Adiabatic_color = pd.read_csv(Adiabatic_color_path, delimiter='\t')
    Adiabatic_cum_cost = pd.read_csv(Adiabatic_cum_cost_path, delimiter='\t')
    Adiabatic_cum_color = pd.read_csv(Adiabatic_cum_color_path, delimiter='\t')

    data_cost = make_data_to_compare(Adiabatic_cost, QAOA_cost)
    make_prob_char(data_cost)
    data_color = make_data_to_compare(Adiabatic_color, QAOA_color)
    make_prob_char(data_color)

    data_cost = make_data_to_compare(Adiabatic_cum_cost, QAOA_cum_cost)
    make_prob_char(data_cost, cum=True)

    data_color = make_data_to_compare(Adiabatic_cum_color, QAOA_cum_color)
    make_prob_char(data_color, cum=True)
