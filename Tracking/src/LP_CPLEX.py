from data import *
from read_data import *
# import pulp
import cplex
from docplex.mp.model import Model


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
    # model.print_information()
    # model.solve()
    # model.print_solution()
    model.export_as_lp("model_docplex")

    return model


if __name__ == '__main__':
    # hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/event000001000-hits.csv'
    hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/sel/event000001000-hits-sel-01.csv'
    hits = read_hits(hits_path)
    layers = list(hits.keys())
    hits_test = dict()

    # no_track = 2
    no_layer = 3
    for l in layers[:no_layer]:
        hs = hits[l]
        hits_test[l] = hs
        # for h in hs:
        #     print(l, h.id)
    # run(hits_test)
    model = run(hits_test)
