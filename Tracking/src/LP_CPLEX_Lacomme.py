from data import *
from read_data import *
# import pulp
import cplex
from docplex.mp.model import Model
import json
import random
import matplotlib.pyplot as plt


def create_variables(model, hits):
    layers = list(hits.keys())
    # create phi_p_p'_i_j variables
    v = []
    for p_1 in range(0, len(layers) - 1):
        for p_2 in range(p_1 + 1, len(layers)):
            for i in range(1, len(hits[p_1]) + 1):
                for j in range(1, len(hits[p_2]) + 1):
                    v.append((p_1, p_2, i, j))
    phi = model.binary_var_dict(v, name="phi")
    print("No_phi_variables: ", len(phi))

    # create b_p_i variables
    v = []
    for p in range(0, len(layers)):
        for i in range(0, len(hits[p])):
            v.append((p, i))
    b = model.binary_var_dict(v, name="b")
    print("No_b_variables: ", len(b))

    return phi, b


def run(hits, nt, M, model_path_out, solution_path_out):
    model = Model(name="Track")

    layers = list(hits.keys())
    print("layers:", layers)
    no_layer = len(layers)
    no_hits = len(list(hits.values())[0])
    print(no_layer, no_hits)

    # create_variables
    phi, b = create_variables(model, hits)

    # add constraints
    # first constraints:
    print("---First constraints---")
    count_constraint = 0
    for p_2 in range(1, len(layers) - 1):
        tmp = 0
        for j in range(1, len(hits[p_2])):
            tmp += phi[0, p_2, 0, j]
        c1 = "FC_" + str(count_constraint)
        count_constraint += 1
        model.add_constraint(tmp == nt, ctname=c1)
    print("Number of first constraints:", count_constraint)

    # Second constraints:
    print("---Second constraints---")
    count_constraint = 0
    for p in range(1, len(layers) - 1):
        tmp = 0
        for i in range(1, len(hits[p])):
            tmp += phi[p, len(layers) - 1, i, 0]
            c2 = "SC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp == 1, ctname=c2)
    print("Number of second constraints:", count_constraint)

    # Third constraints:
    print("---Third constraints---")
    count_constraint = 0
    for p_1 in range(1, len(layers) - 1):
        for i in range(1, len(hits[p_1])):
            tmp_1 = 0
            for p_2 in range(0, p_1):
                for j in range(1, len(hits[p_2])):
                    tmp_1 += phi[p_2, p_1, i, j]

            tmp_2 = 0
            for p_2 in range(p_1, len(layers)):
                for j in range(1, len(hits[p_2])):
                    tmp_2 += phi[p_1, p_2, i, j]

            c3 = "TC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp_1 == tmp_2, ctname=c3)

    print("Number of Third constraints:", count_constraint)

    print("---Fourth constraints---")
    count_constraint = 1
    for p_1 in range(1, len(layers) - 1):
        for i in range(1, len(hits[p_1])):
            tmp = 0
            for p_2 in range(0, p_1):
                for j in range(1, len(hits[p_2])):
                    tmp += phi[p_2, p_1, i, j]
            c4 = "FoC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp <= 1, ctname=c4)
    print("Number of Fourth constraints:", count_constraint)

    print("---Fiveth constraints---")
    count_constraint = 1
    for p_1 in range(1, len(layers) - 1):
        for i in range(1, len(hits[p_1])):
            tmp = 0
            for p_2 in range(0, p_1):
                for j in range(1, len(hits[p_2])):
                    tmp += phi[p_2, p_1, i, j]
            c5 = "FiC_1_" + str(count_constraint)
            c6 = "FiC_2_" + str(count_constraint)
            count_constraint += 2
            model.add_constraint(tmp <= b[p_1, i], ctname=c5)
            model.add_constraint(tmp >= b[p_1, i], ctname=c6)
    print("Number of Fiveth constraints:", count_constraint)

    print("---Sixth constraints---")
    count_constraint = 1
    for p_1 in range(1, len(layers) - 2):
        for i in range(0, len(hits[p_1])):
            for p_2 in range(p_1 + 1, len(layers) - 1):
                for j in range(0, len(hits[p_2])):
                    for p_3 in range(p_2, len(layers)):
                        for k in range(0, len(hits[p_3])):
                            h_i = hits[layers[p_1]][i]
                            h_j = hits[layers[p_2]][j]
                            h_k = hits[layers[p_3]][k]
                            seg_1 = Segment(h_j, h_i)
                            seg_2 = Segment(h_j, h_k)
                            beta = Angle(seg_1=seg_1, seg_2=seg_2).angle
                            tmp = (5 - b[p_1, i] - b[p_2,j] - b[p_3, k] - phi[p_1, p_2, i, j] - phi[p_2, p_3, j, k]) * M + ?
                            c7 = "SiC_" + str(count_constraint)
                            count_constraint += 1
                            model.add_constraint(beta <= tmp, ctname=c7)
    print("Number of Sixth constraints:", count_constraint)

    objective = 0
    for p in range(1, no_layer - 1):
        for i in range(1, no_hits + 1):
            for j in range(1, no_hits + 1):
                for k in range(1, no_hits + 1):
                    h_i = hits[layers[p - 1]][i - 1]
                    h_j = hits[layers[p]][j - 1]
                    h_k = hits[layers[p + 1]][k - 1]
                    seg_1 = Segment(h_j, h_i)
                    seg_2 = Segment(h_j, h_k)
                    objective += z[p, i, j, k] * Angle(seg_1=seg_1, seg_2=seg_2).angle

    model.set_objective('min', objective)

    model.print_information()
    model.solve(log_output=True)

    model.export_as_lp(model_path_out)
    model.solution.export(solution_path_out)
    f = open(solution_path_out)
    result = json.load(f)
    f.close()

    return result['CPLEXSolution']['variables']


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
    hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/sel/event000001000-hits-sel-01.csv'
    hits_volume = read_hits(hits_path)
    hits = dict()
    for k, v in hits_volume.items():
        # print(k, v)
        print("Volume id:", k)
        print("No_layers:", len(v))
        hits = v

    # hits_volume
    layers = list(hits.keys())

    model_path_out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_f2_Lacomme/model_docplex.lp"
    solution_path_out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_f2_Lacomme/solution.json"
    result = run(hits, model_path_out, solution_path_out)

    solution = dict()
    segments = []
    for var in result:
        f_p_i_j = var['name'].split('_')
        if f_p_i_j[0] == 'z':
            continue
        print(var)
        p = int(f_p_i_j[1])
        i = int(f_p_i_j[2])
        j = int(f_p_i_j[3])

        h_1 = hits[layers[p - 1]][i - 1]
        h_2 = hits[layers[p]][j - 1]
        segments.append([h_1, h_2])
    out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_f2_Lacomme/result.PNG"
    display(hits, segments, out)
