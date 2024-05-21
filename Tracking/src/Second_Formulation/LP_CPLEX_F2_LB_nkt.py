from data import *
from read_data import *
# import pulp
import cplex
from docplex.mp.model import Model
import json
import random
import matplotlib.pyplot as plt


def run(hits, model_path_out, solution_path_out, figure_path_out):
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

    objective = 0
    LB = 0
    for p in range(1, no_layer - 1):
        n_p = len(hits[layers[p - 1]]) + 1
        for i in range(1, n_p):
            min_beta = 10000
            n_p_1 = len(hits[layers[p]]) + 1
            for j in range(1,n_p_1):
                n_p_2 = len(hits[layers[p+1]]) + 1
                for k in range(1, n_p_2):
                    h_i = hits[layers[p - 1]][i - 1]
                    h_j = hits[layers[p]][j - 1]
                    h_k = hits[layers[p + 1]][k - 1]
                    seg_1 = Segment(h_j, h_i)
                    seg_2 = Segment(h_j, h_k)
                    beta = Angle(seg_1=seg_1, seg_2=seg_2).angle
                    if min_beta > beta:
                        min_beta = min_beta
                    objective += z[p, i, j, k] * beta
            LB += min_beta

    model.set_objective('min', objective)
    model.add_constraint(objective >= LB, ctname="LB of objective value")
    # Constraints
    # first constraints:
    print("---First constraints---")
    count_constraint = 0

    for j in range(1, no_hits + 1):
        for p in range(1, no_layer):
            tmp = 0
            n_p = len(hits[layers[p - 1]]) + 1
            for i in range(1, n_p):
                tmp += f[p, i, j]
            constraint_name = "FC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp == 1, ctname=constraint_name)
    print("Number of first constraints:", count_constraint)
    # print()
    # Second constraints:
    print("---Second constraints---")
    count_constraint = 0
    for p in range(1, no_layer):
        n_p = len(hits[layers[p - 1]]) + 1
        for i in range(1, n_p):
            tmp = 0
            n_p_1 = len(hits[layers[p]]) + 1
            for j in range(1, n_p_1):
                tmp += f[p, i, j]
            constraint_name = "SC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp == 1, ctname=constraint_name)
    print("Number of second constraints:", count_constraint)
    # Addition constraints:
    print("---Addition constraints---")
    count_constraint = 0
    for p in range(1, no_layer - 1):
        n_p = len(hits[layers[p - 1]]) + 1
        for i in range(1, n_p):
            n_p_1 = len(hits[layers[p]]) + 1
            for j in range(1, n_p_1):
                n_p_2 = len(hits[layers[p + 1]]) + 1
                for k in range(1, n_p_2):
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

    result = result['CPLEXSolution']['variables']

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

    display(hits, segments, figure_path_out)


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

def load_hits(df):
    list_df = [row.tolist() for index, row in df.iterrows()]
    volumes = dict()

    for i in list_df:
        hit = Hit(
            hit_id=i[0],
            x=i[1],
            y=i[2],
            z=i[3],
            volume_id=i[4],
            layer_id=i[5],
            module_id=i[6],
            particle_id=i[7]
        )
        volume_id = int(hit.volume_id)
        if volume_id not in volumes:
            volumes[volume_id] = [hit]
        else:
            volumes[volume_id] += [hit]
    for id, hits in volumes.items():
        layers = dict()
        for hit in hits:
            layer_id = int(hit.layer_id)
            if layer_id not in layers:
                layers[layer_id] = [hit]
            else:
                layers[layer_id] += [hit]
        volumes[id] = layers
    return volumes


if __name__ == '__main__':
    src_path = '../data_selected'
    data_path = src_path + '/20hits/unknow_track/hits.csv'
    hits_volume = read_hits(data_path)
    hits = hits_volume[9]

    model_path_out = "results/20hits/unknow_track/model_docplex_LB.lp"
    solution_path_out = "results/20hits/unknow_track/solution.json"
    figure_path_out = "results/20hits/unknow_track/result.PNG"

    result = run(hits, model_path_out, solution_path_out, figure_path_out)

