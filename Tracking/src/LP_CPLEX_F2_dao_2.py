from data import *
from read_data import *
# import pulp
import cplex
from docplex.mp.model import Model
import json
import random
import matplotlib.pyplot as plt


def run(hits, model_path_out, solution_path_out, M):
    model = Model(name="Track")

    layers = list(hits.keys())
    print("layers:", layers)
    no_layer = len(layers)
    no_hits = len(list(hits.values())[0])
    print(no_layer, no_hits)

    f = model.binary_var_dict(
        [(p, i, j) for p in range(1, no_layer) for i in range(1, no_hits + 1) for j in range(1, no_hits + 1)],
        name="f")
    # print(f)
    z = model.continuous_var_dict(
        [(p, i, j, k) for p in range(1, no_layer - 1) for i in range(1, no_hits + 1) for j in range(1, no_hits + 1) for
         k in
         range(1, no_hits + 1)], name="z", lb=0, ub=1)
    c = model.continuous_var_dict(
        [(p, i) for p in range(1, no_layer - 1) for i in range(1, no_hits + 1)], name="c", lb=0)

    min_cost = 0
    for p in range(1, no_layer - 1):
        for i in range(1, no_hits + 1):
            min_angle = 10
            for j in range(1, no_hits + 1):
                for k in range(1, no_hits + 1):
                    h_i = hits[layers[p - 1]][i - 1]
                    h_j = hits[layers[p]][j - 1]
                    h_k = hits[layers[p + 1]][k - 1]
                    seg_1 = Segment(h_j, h_i)
                    seg_2 = Segment(h_j, h_k)
                    beta = Angle(seg_1=seg_1, seg_2=seg_2).angle
                    if beta < min_angle:
                        min_angle = beta
                    model.add_constraint(beta <= (1 - z[p, i, j, k]) * M + c[p, i])
            model.add_constraint(c[p, i] >= min_angle)
            min_cost += min_angle

    objective = 0
    for p in range(1, no_layer - 1):
        for i in range(1, no_hits + 1):
            objective += c[p, i]
    model.add_constraint(objective >= min_cost)
    model.set_objective('min', objective)
    # model.add_constraint(objective >= LB, ctname="LB of objective value")
    # Constraints
    # first constraints:
    print("---First constraints---")
    count_constraint = 0

    for j in range(1, no_hits + 1):
        for p in range(1, no_layer):
            tmp = 0
            for i in range(1, no_hits + 1):
                tmp += f[p, i, j]
            constraint_name = "FC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp == 1, ctname=constraint_name)
    print("Number of first constraints:", count_constraint)
    # print()
    # Second constraints:
    print("---Second constraints---")
    count_constraint = 0
    for i in range(1, no_hits + 1):
        for p in range(1, no_layer):
            tmp = 0
            for j in range(1, no_hits + 1):
                tmp += f[p, i, j]
            constraint_name = "SC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp == 1, ctname=constraint_name)
    print("Number of second constraints:", count_constraint)
    # Addition constraints:
    print("---Addition constraints---")
    count_constraint = 0
    for p in range(1, no_layer - 1):
        for i in range(1, no_hits + 1):
            for j in range(1, no_hits + 1):
                for k in range(1, no_hits + 1):
                    c1 = f[p, i, j] + f[p + 1, j, k] - z[p, i, j, k] <= 1
                    c2 = z[p, i, j, k] <= f[p, i, j]
                    c3 = z[p, i, j, k] <= f[p + 1, j, k]
                    c4 = z[p, i, j, k] >= 0
                    constraint_1_name = "AC_" + str(count_constraint) + "_1"
                    constraint_2_name = "AC_" + str(count_constraint) + "_2"
                    constraint_3_name = "AC_" + str(count_constraint) + "_3"
                    constraint_4_name = "AC_" + str(count_constraint) + "_4"
                    count_constraint += 4

                    model.add_constraint(c1, ctname=constraint_1_name)
                    model.add_constraint(c2, ctname=constraint_2_name)
                    model.add_constraint(c3, ctname=constraint_3_name)
                    model.add_constraint(c4, ctname=constraint_4_name)
    print("Number of addition constraints:", count_constraint)

    model.print_information()
    model.solve(log_output=True)

    model.export_as_lp(model_path_out)
    model.solution.export(solution_path_out)
    f = open(solution_path_out)
    result = json.load(f)
    f.close()

    return result['CPLEXSolution']['variables']


def pick_random_hits(no_track, hits):
    layers = list(hits.keys())
    new_hits = dict()
    for p, hp in hits.items:
        idx = []
        while len(idx) < no_track:
            id = random.randint(0, len(hp))
            if id not in idx:
                idx.append(id)
        new_hp = [hp[i] for i in idx]
        new_hits[p] = new_hp
    return new_hits


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
    hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/volume_id_9/hits-vol_9_20_track.csv'
    hits_volume = read_hits(hits_path)
    hits = dict()
    for k, v in hits_volume.items():
        # print(k, v)
        print("Volume id:", k)
        print("No_layers:", len(v))
        hits = v

    # hits_volume

    out_angles_path = "result_f2_20_hits/min_angles.json"


    model_path_out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_f2_20_hits/model_docplex.lp"
    solution_path_out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_f2_20_hits/solution.json"

    M = 1000
    result = run(hits, model_path_out, solution_path_out, M)
    with open(solution_path_out, 'r', encoding='utf-8') as f:
        result = json.load(f)['CPLEXSolution']['variables']

    layers = list(hits.keys())
    segments = []
    for var in result:
        f_p_i_j = var['name'].split('_')
        print(var)
        if f_p_i_j[0] == 'z' or f_p_i_j[0] == 'c':
            continue
        p = int(f_p_i_j[1])
        i = int(f_p_i_j[2])
        j = int(f_p_i_j[3])

        h_1 = hits[layers[p - 1]][i - 1]
        h_2 = hits[layers[p]][j - 1]
        segments.append([h_1, h_2])
    out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_f2_20_hits/result.PNG"
    display(hits, segments, out)
