from data import *
from read_data import *
from docplex.mp.model import Model
import json
import matplotlib.pyplot as plt


def create_variables(model, hits):
    layers = sorted(list(hits.keys()))
    K = len(layers) + 1
    layers = [0] + layers + [0]
    v = []
    for p_1 in range(1, K - 1):
        n_p_1 = len(hits[layers[p_1]]) + 1
        for p_2 in range(2, K):
            n_p_2 = len(hits[layers[p_2]]) + 1
            for i in range(1, n_p_1):
                for j in range(1, n_p_2):
                    v.append((p_1, p_2, i, j))

    phi = model.binary_var_dict(v, name="phi")

    v = []
    for p_1 in range(1, K - 2):
        n_p_1 = len(hits[layers[p_1]]) + 1
        for i in range(1, n_p_1):
            v.append((p_1, i))
    c = model.continuous_var_dict(v, name="c", lb=0)
    ob = model.continuous_var(name="ob")
    return ob, phi, c


def distance(h1, h2):
    distance = math.sqrt((h2.x - h1.x) ** 2 + (h2.y - h1.y) ** 2 + (h2.z - h1.z) ** 2)
    return distance


def run(hits, M, model_path_out, solution_path_out, figure_path_out):
    model = Model(name="Track")
    layers = sorted(list(hits.keys()))
    K = len(layers) + 1
    layers = [0] + layers + [0]
    print("layers:", layers)

    print("K=", K)
    # create_variables
    ob, phi, c = create_variables(model, hits)

    nts = [len(hits[l]) for l in layers[1:-1]]
    min_nt = min(nts)

    # First constraints:
    print("---First constraints---")
    count_constraint_0 = 0
    for p_1 in range(2, K):
        n_p_1 = len(hits[layers[p_1]]) + 1
        for i in range(1, n_p_1):
            tmp = 0
            for p_2 in range(1, p_1):
                n_p_2 = len(hits[layers[p_2]]) + 1
                for j in range(1, n_p_2):
                    tmp += phi[p_2, p_1, j, i]
            count_constraint_0 += 1
            c0 = "ZC_" + str(count_constraint_0)
            model.add_constraint(tmp <= 1, ctname=c0)

    # First constraints:
    print("---First constraints---")
    count_constraint_1 = 0
    for p_1 in range(1, K - 1):
        n_p_1 = len(hits[layers[p_1]]) + 1
        for i in range(1, n_p_1):
            tmp = 0
            for p_2 in range(p_1 + 1, K):
                n_p_2 = len(hits[layers[p_2]]) + 1
                for j in range(1, n_p_2):
                    tmp += phi[p_1, p_2, i, j]
            count_constraint_1 += 1
            c1 = "FC_" + str(count_constraint_1)
            model.add_constraint(tmp <= 1, ctname=c1)

    print("---Second, Third and Fourth constraints---")
    count_constraint_2 = 0
    count_constraint_3 = 0
    count_constraint_4 = 0
    for p_1 in range(2, K - 1):
        n_p_1 = len(hits[layers[p_1]]) + 1
        tmp_1 = 0
        tmp_2 = 0
        for i in range(1, n_p_1):
            t_1 = 0
            for p_2 in range(1, p_1):
                n_p_2 = len(hits[layers[p_2]]) + 1
                for j in range(1, n_p_2):
                    tmp_1 += phi[p_2, p_1, j, i]
                    t_1 += phi[p_2, p_1, j, i]
            t_2 = 0
            for p_2 in range(p_1 + 1, K):
                n_p_2 = len(hits[layers[p_2]]) + 1
                for j in range(1, n_p_2):
                    tmp_2 += phi[p_1, p_2, i, j]
                    t_2 += phi[p_1, p_2, i, j]
            count_constraint_2 += 1
            c2 = "SeC_" + str(count_constraint_2)
            model.add_constraint(t_1 == t_2, ctname=c2)

        count_constraint_3 += 1

        c3 = "ThC_" + str(count_constraint_3)
        model.add_constraint(tmp_1 >= min_nt, ctname=c3)
        count_constraint_3 += 1
        c3 = "ThC_" + str(count_constraint_3)
        model.add_constraint(tmp_1 <= n_p_1 - 1, ctname=c3)

        count_constraint_4 += 1
        c4 = "FoC_" + str(count_constraint_4)
        model.add_constraint(tmp_2 >= min_nt, ctname=c4)
        count_constraint_4 += 1
        c4 = "FoC_" + str(count_constraint_4)
        model.add_constraint(tmp_2 <= n_p_1 - 1, ctname=c4)


    print("--- Fifth constraints---")
    count_constraint_5 = 0

    min_cost = 0
    for p_1 in range(1, K - 2):
        n_p_1 = len(hits[layers[p_1]]) + 1
        for i in range(1, n_p_1):
            min_beta = 1000000000

            for p_2 in range(p_1 + 1, K - 1):
                n_p_2 = len(hits[layers[p_2]]) + 1
                for j in range(1, n_p_2):
                    for p_3 in range(p_2 + 1, K):
                        n_p_3 = len(hits[layers[p_3]]) + 1
                        for k in range(1, n_p_3):
                            h_i = hits[layers[p_1]][i - 1]
                            h_j = hits[layers[p_2]][j - 1]
                            h_k = hits[layers[p_3]][k - 1]
                            seg_1 = Segment(h_j, h_i)
                            seg_2 = Segment(h_j, h_k)
                            beta = Angle(seg_1=seg_1, seg_2=seg_2).angle

                            if beta < min_beta:
                                min_beta = beta

                            count_constraint_5 += 1
                            c5 = "FiC_" + str(count_constraint_5)
                            model.add_constraint(
                                beta <= c[p_1, i] + (2 - phi[p_1, p_2, i, j] - phi[p_2, p_3, j, k]) * M, ctname=c5)

            min_cost += min_beta

    objective = 0
    for p_1 in range(1, K - 2):
        n_p_1 = len(hits[layers[p_1]]) + 1
        for i in range(1, n_p_1):
            objective += c[p_1, i]
    model.add_constraint(objective >= min_cost, ctname="LB of objective value")
    # model.add_constraint(objective <= 0.6618)
    model.add_constraint(ob >= objective)
    model.set_objective('min', ob)

    model.print_information()
    model.export_as_lp(model_path_out)
    model.solve(log_output=True)
    print(model.solve_status)
    model.solution.export(solution_path_out)

    f = open(solution_path_out)
    result = json.load(f)
    f.close()

    result = result['CPLEXSolution']['variables']

    segments = []
    for var in result:
        var_name = var['name']
        var_value = round(float(var['value']))
        phi_p_p_i_j = var_name.split('_')
        print("var:", var_name, "-- value:", var_value)
        if 'c' in phi_p_p_i_j[0] or var_value != 1.0 or 's' in phi_p_p_i_j[0] or 'q' in phi_p_p_i_j[0] or 'ob' in \
                phi_p_p_i_j[0]:
            continue

        p_1 = int(phi_p_p_i_j[1])
        p_2 = int(phi_p_p_i_j[2])
        i = int(phi_p_p_i_j[3])
        j = int(phi_p_p_i_j[4])
        h_1 = hits[layers[p_1]][i - 1]
        h_2 = hits[layers[p_2]][j - 1]
        segments.append([h_1, h_2])
    display(hits, segments, figure_path_out)
    return


def display(hits, segments, out=""):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    layers = list(hits.keys())
    xs = []
    ys = []
    zs = []

    for p in layers:
        h_p = hits[p]
        for h in h_p:
            xs.append(h.x)
            ys.append(h.y)
            zs.append(h.z)
    ax.scatter(xs, ys, zs, marker='o', color='red')

    for segment in segments:
        h1 = segment[0]
        h2 = segment[1]
        ax.plot(xs=[h1.x, h2.x], ys=[h1.y, h2.y], zs=[h1.z, h2.z], color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(out)
    plt.show()


if __name__ == '__main__':
    src_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/data_selected'
    data_path = src_path + '/15hits/know_track/hits.csv'

    hits_volume = read_hits(data_path)
    hits = hits_volume[9]

    model_path_out = "results/15hits/know_track/model_docplex_LB.lp"
    solution_path_out = "results/15hits/know_track/solution_LB.json"
    figure_path_out = "results/15hits/know_track/result_LB.PNG"

    M = 10000
    result = run(hits, M, model_path_out, solution_path_out, figure_path_out)