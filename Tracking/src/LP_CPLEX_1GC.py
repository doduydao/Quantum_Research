from data import *
from read_data import *
from docplex.mp.model import Model
import json
import random
import matplotlib.pyplot as plt


def run(hits, model_path_out, solution_path_out):
    model = Model(name="Track")

    layers = list(hits.keys())
    all_hits = list(hits.values())

    print("Original:")
    for p in range(len(layers)):
        print("Layer", layers[p], "-", len(all_hits[p]), "hits")

    new_hits = []
    for p in range(len(layers)):
        k = random.randint(1, 10)
        idx = random.sample(range(len(all_hits[p])), k)
        new_hits.append([all_hits[p][i] for i in idx])

    print("Reduced:")
    for p in range(len(layers)):
        print("Layer", layers[p], "-", len(new_hits[p]), "hits")

    # create f_p_i_j variables
    v = []
    for p in range(1, len(layers)):
        for i in range(1, len(new_hits[p - 1]) + 1):
            for j in range(1, len(new_hits[p]) + 1):
                v.append((p, i, j))
    f = model.binary_var_dict(v, name="f")
    print("No_f_variables: ", len(f))

    # create z_p_i_j_k variables
    v = []
    for p in range(1, len(layers) - 1):
        for i in range(1, len(new_hits[p - 1]) + 1):
            for j in range(1, len(new_hits[p]) + 1):
                for k in range(1, len(new_hits[p + 1]) + 1):
                    v.append((p, i, j, k))
    z = model.binary_var_dict(v, name="z")
    print("No_z_variables: ", len(z))

    objective = 0
    for p in range(1, len(layers) - 1):
        for i in range(1, len(new_hits[p - 1]) + 1):
            for j in range(1, len(new_hits[p]) + 1):
                for k in range(1, len(new_hits[p + 1]) + 1):
                    h_i = new_hits[p - 1][i - 1]
                    h_j = new_hits[p][j - 1]
                    h_k = new_hits[p + 1][k - 1]
                    seg_1 = Segment(h_i, h_j)
                    seg_2 = Segment(h_j, h_k)
                    objective += z[p, i, j, k] * Angle(seg_1=seg_1, seg_2=seg_2).angle

    model.set_objective('min', objective)

    # Constraints
    # first constraints:
    print("---First constraints---")
    count_constraint = 1

    for p in range(1, len(layers)):
        for j in range(1, len(new_hits[p]) + 1):
            tmp = 0
            for i in range(1, len(new_hits[p - 1]) + 1):
                tmp += f[p, i, j]
            constraint_name = "FC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp <= 1, ctname=constraint_name)
    print("Number of first constraints:", count_constraint - 1)
    # print()
    # Second constraints:
    print("---Second constraints---")
    count_constraint = 1
    for p in range(1, len(layers)):
        for i in range(1, len(new_hits[p-1]) + 1):
            tmp = 0
            for j in range(1, len(new_hits[p]) + 1):
                tmp += f[p, i, j]
            constraint_name = "SC_" + str(count_constraint)
            count_constraint += 1
            model.add_constraint(tmp == 1, ctname=constraint_name)
    print("Number of second constraints:", count_constraint - 1)

    # Addition constraints:
    print("---Addition constraints---")
    count_constraint = 1
    for p in range(1, len(layers) - 1):
        for i in range(1, len(new_hits[p - 1]) + 1):
            for j in range(1, len(new_hits[p]) + 1):
                for k in range(1, len(new_hits[p + 1]) + 1):
                    c1 = z[p, i, j, k] <= f[p, i, j]
                    c2 = z[p, i, j, k] <= f[p + 1, j, k]
                    c3 = f[p, i, j] + f[p + 1, j, k] - z[p, i, j, k] <= 1
                    constraint_1_name = "AC_" + str(count_constraint) + "_1"
                    constraint_2_name = "AC_" + str(count_constraint) + "_2"
                    constraint_3_name = "AC_" + str(count_constraint) + "_3"
                    count_constraint += 1

                    model.add_constraint(c1, ctname=constraint_1_name)
                    model.add_constraint(c2, ctname=constraint_2_name)
                    model.add_constraint(c3, ctname=constraint_3_name)
    print("Number of addition constraints:", count_constraint - 1)

    model.print_information()
    model.solve(log_output=True)

    model.export_as_lp(model_path_out)
    model.solution.export(solution_path_out)
    f = open(solution_path_out)
    result = json.load(f)
    f.close()

    return result['CPLEXSolution']['variables']
    # return 0


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
    hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/volume_id_9/event000001000-volume_id_9.csv'
    hits_volume = read_hits(hits_path)
    hits = dict()
    for k, v in hits_volume.items():
        # print(k, v)
        print("Volume id:", k)
        print("No_layers:", len(v))
        hits = v

    # hits_volume
    layers = list(hits.keys())

    model_path_out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_1gc/model_docplex.lp"
    solution_path_out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_1gc/solution.json"
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
    out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/src/result_1gc/result.PNG"
    display(hits, segments, out)
