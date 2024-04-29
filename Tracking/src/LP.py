from data import *
from read_data import *
import pulp
import cplex
import json
from show_3D import *
def run(hits):
    # define model
    model = pulp.LpProblem('Track_Finding', pulp.LpMinimize)

    # define variable
    layers = list(hits.keys())
    print(layers)
    no_layer = len(layers)
    no_hits = len(list(hits.values())[0])
    no_track = no_hits
    print(no_layer, no_hits)

    x = dict()
    for t in range(1, no_track + 1):
        for p in range(1, no_layer + 1):
            for i in range(1, no_hits + 1):
                x_p_t_i = pulp.LpVariable(name=f'x_{p}_{t}_{i}', cat='Binary')
                key = str(p) + str(t) + str(i)
                x[key] = x_p_t_i
    # print(x)
    y = dict()
    for t in range(1, no_track + 1):
        for p in range(1, no_layer + 1):
            for i in range(1, no_hits + 1):
                for j in range(1, no_hits + 1):
                    for k in range(1, no_hits + 1):
                        y_p_t_i_j_k = pulp.LpVariable(name=f'y_{p}_{t}_{i}_{j}_{k}', cat='Binary')
                        key = str(p) + str(t) + str(i) + str(j) + str(k)
                        y[key] = y_p_t_i_j_k
    # print(y)
    # Objective function
    objective = None
    for t in range(1, no_track + 1):
        for i in range(1, no_hits + 1):
            for j in range(1, no_hits + 1):
                for k in range(1, no_hits + 1):
                    for p in range(0, no_layer - 2):
                        h_i = hits[layers[p]][i - 1]
                        h_j = hits[layers[p + 1]][j - 1]
                        h_k = hits[layers[p + 2]][k - 1]
                        seg_1 = Segment(h_i, h_j)
                        seg_2 = Segment(h_j, h_k)
                        key = str(p+1) + str(t) + str(i) + str(j) + str(k)
                        objective += y[key] * Angle(seg_1=seg_1, seg_2=seg_2).angle
    model.setObjective(objective)

    # Constraints

    # first constraints:
    # print("First constraints:")
    for i in range(1, no_hits + 1):
        for p in range(1, no_layer + 1):
            tmp = 0
            for t in range(1, no_track + 1):
                key = str(p) + str(t) + str(i)
                tmp += x[key]
            constraint = tmp == 1
            # print(constraint)
            model.addConstraint(constraint)

    # print()
    # Second constraints:
    # print("Second constraints:")
    for t in range(1, no_track + 1):
        for p in range(1, no_layer + 1):
            tmp = 0
            for i in range(1, no_hits + 1):
                key = str(p) + str(t) + str(i)
                tmp += x[key]
            constraint = tmp == 1
            # print(constraint)
            model.addConstraint(constraint)
    # print()

    # Addition constraints:
    # print(y_p_t_i_j_k)

    print("Addition constraints")
    for t in range(1, no_track + 1):
        for p in range(1, no_layer-1):
            for i in range(1, no_hits + 1):
                for j in range(1, no_hits + 1):
                    for k in range(1, no_hits + 1):
                        key_1 = str(p) + str(t) + str(i)
                        key_2 = str(p+1) + str(t) + str(j)
                        key_3 = str(p+2) + str(t) + str(k)
                        key_y = str(p) + str(t) + str(i) + str(j) + str(k)
                        constraint_1 = x[key_1] + x[key_2] + x[key_3] - y[key_y] <= 2
                        constraint_2 = -(x[key_1] + x[key_2] + x[key_3]) - 3 * y[key_y] <= 0
                        model.addConstraint(constraint_1)
                        model.addConstraint(constraint_2)
                        # print(constraint_1)
                        # print(constraint_2)


    print(model, file=open('model.txt', 'w'))
    # Solving
    # solver = pulp.getSolver('CPLEX_CMD')
    model.solve()
    print(model.sol_status)
    # model.writeLP("just_to_be_sure.txt")

    vars = dict()

    for var in model.variables():
        vars[var.name] = var.varValue
        print(var.name, var.varValue)

    with open("result_pulp.json", "w") as outfile:
        json.dump(vars, outfile)

    return vars

if __name__ == '__main__':
    # hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/event000001000-hits.csv'
    hits_path = 'C:\\Users\dddo\PycharmProjects\Quantum_Research\Tracking\event000001000\sel\event000001000-hits-sel-01.csv'
    hits = read_hits(hits_path)
    layers = list(hits.keys())
    hits_test = dict()

    # no_track = 2
    no_layer = 5
    for l in layers[:no_layer]:
        hs = hits[l]
        hits_test[l] = hs
        # for h in hs:
        #     print(l, h.id)
    result = run(hits_test)

    # f = open('result_pulp.json')
    # result = json.load(f)
    # f.close()

    solution = dict()
    for var_name, var_value in result.items():
        x_p_t_i = var_name.split('_')
        if x_p_t_i[0] == 'y' or var_value == 0:
            continue
        p = int(x_p_t_i[1])
        t = int(x_p_t_i[2])
        i = int(x_p_t_i[3])
        print(var_name, var_value)
        if t not in solution:
            solution[t] = [hits[layers[p - 1]][i - 1]]
        else:
            solution[t] += [hits[layers[p - 1]][i - 1]]

    display(hits_test, solution)
