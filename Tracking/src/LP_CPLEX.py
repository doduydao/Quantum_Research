from data import *
from read_data import *
# import pulp
import cplex
from docplex.mp.model import Model
import json
from show_3D import *
import random

def run(hits):
    model = Model(name="Track")

    layers = list(hits.keys())
    print(layers)
    no_layer = len(layers)
    no_hits = len(list(hits.values())[0])
    no_track = no_hits
    print(no_layer, no_hits)

    x = model.binary_var_dict(
        [(p, t, i) for t in range(1, no_track + 1) for p in range(1, no_layer + 1) for i in range(1, no_hits + 1)],
        name="x")
    y = model.binary_var_dict(
        [(p, t, i, j, k) for t in range(1, no_track + 1) for p in range(1, no_layer - 1) for i in range(1, no_hits + 1)
         for j in range(1, no_hits + 1) for k in range(1, no_hits + 1)], name="y")

    objective = 0
    for t in range(1, no_track + 1):
        for i in range(1, no_hits + 1):
            for j in range(1, no_hits + 1):
                for k in range(1, no_hits + 1):
                    for p in range(1, no_layer - 1):
                        h_i = hits[layers[p - 1]][i - 1]
                        h_j = hits[layers[p]][j - 1]
                        h_k = hits[layers[p + 1]][k - 1]
                        seg_1 = Segment(h_i, h_j)
                        seg_2 = Segment(h_j, h_k)
                        objective += y[p, t, i, j, k] * Angle(seg_1=seg_1, seg_2=seg_2).angle

    model.set_objective('min', objective)

    # Constraints
    # first constraints:
    print("First constraints:")
    for i in range(1, no_hits + 1):
        for p in range(1, no_layer + 1):
            tmp = 0
            for t in range(1, no_track + 1):
                tmp += x[p, t, i]
            model.add_constraint(tmp == 1)

    # print()
    # Second constraints:
    print("Second constraints:")
    for t in range(1, no_track + 1):
        for p in range(1, no_layer + 1):
            tmp = 0
            for i in range(1, no_hits + 1):
                tmp += x[p, t, i]
            model.add_constraint(tmp == 1)

    # Addition constraints:
    # print("Addition constraints")
    for t in range(1, no_track + 1):
        for p in range(1, no_layer - 1):
            for i in range(1, no_hits + 1):
                for j in range(1, no_hits + 1):
                    for k in range(1, no_hits + 1):
                        c1 = x[p, t, i] + x[p + 1, t, j] + x[p + 2, t, k] - y[p, t, i, j, k] <= 2
                        c2 = -(x[p, t, i] + x[p + 1, t, j] + x[p + 2, t, k]) - 3 * y[p, t, i, j, k] <= 0
                        model.add_constraint(c1)
                        model.add_constraint(c2)
    model.print_information()
    model.solve()
    model.print_solution()
    model.export_as_lp("model_docplex")
    model.solution.export("solution.json")
    f = open('solution.json')
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

if __name__ == '__main__':
    # hits_path = 'C:\\Users\dddo\PycharmProjects\Quantum_Research\Tracking\event000001000\event000001000-hits.csv'
    # hits_path = 'C:\\Users\dddo\PycharmProjects\Quantum_Research\Tracking\event000001000\sel\event000001000-hits-sel-01.csv'
    hits_path = 'C:\\Users\dddo\PycharmProjects\Quantum_Research\Tracking\event000001000\sublayer_2\event000001000-hits_random.csv'
    hits_volume = read_hits(hits_path)
    hits = dict()




    # hits_volume
    # hits_test = dict()
    layers = list(hits.keys())

    # no_track = 3
    # no_layer = 7
    # for l in layers[:no_layer]:
    #     hs = hits[l][:no_track]
    #     hits_test[l] = hs
        # for h in hs:
        #     print(l, h.id)
    # run(hits_test)
    # hits_test = pick_random_hits(no_track, hits)


    result = run(hits)
    solution = dict()
    for var in result:
        x_p_t_i = var['name'].split('_')
        if x_p_t_i[0] =='y':
            continue
        p = int(x_p_t_i[1])
        t = int(x_p_t_i[2])
        i = int(x_p_t_i[3])
        print(var['name'])

        if t not in solution:
            solution[t] = [hits[layers[p - 1]][i-1]]
        else:
            solution[t] += [hits[layers[p - 1]][i-1]]

    display(hits, solution)
