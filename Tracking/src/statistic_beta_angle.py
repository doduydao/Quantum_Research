from data import *
from read_data import *
from docplex.mp.model import Model
import json
import matplotlib.pyplot as plt
import math

def calculate_all_angles(hits, out):
    # tính tất cả các góc nhỏ nhất theo layer
    angles = dict()
    layers = list(hits.keys())
    L = len(layers) + 1
    layers = [0] + layers + [0]

    betas = []

    for p in range(1, L - 2):
        min_angle = 100000000
        n_p = len(hits[layers[p]]) + 1
        n_p_1 = len(hits[layers[p + 1]]) + 1
        n_p_2 = len(hits[layers[p + 2]]) + 1

        for i in range(1, n_p):
            for j in range(1, n_p_1):
                for k in range(1, n_p_2):
                    h_i = hits[layers[p]][i - 1]
                    h_j = hits[layers[p + 1]][j - 1]
                    h_k = hits[layers[p + 2]][k - 1]
                    seg_1 = Segment(h_j, h_i)
                    seg_2 = Segment(h_j, h_k)
                    # angle = Angle(seg_1=seg_1, seg_2=seg_2).angle * (distance(h_i, h_j) + distance(h_j, h_k))
                    angle = Angle(seg_1=seg_1, seg_2=seg_2).angle
                    betas.append(angle)
                    if min_angle > angle:
                        min_angle = angle
        angles[layers[p]] = min_angle * len(hits[layers[p]])
    print(len(betas))
    print("Max beta:", max(betas))
    print("Min beta:", min(betas))
    print("Mean beta:", sum(betas) / len(betas))

    dict_betas = dict()
    interval = 0.1
    for i in range(math.ceil(math.pi / interval)):
        dict_betas[i * interval] = 0

    for beta in betas:
        key = math.floor(beta / interval) * interval

        if key not in dict_betas:
            dict_betas[key] = 1
        else:
            dict_betas[key] += 1

    fre_beta_sorted = sorted(dict_betas.items(), key=lambda x: x[0])
    for i in fre_beta_sorted:
        print(i[0], i[1])

    # with open(out, 'w') as file:
    #     json.dump(angles, file)
    # return angles


def distance(h1, h2):
    distance = math.sqrt((h2.x - h1.x) ** 2 + (h2.y - h1.y) ** 2 + (h2.z - h1.z) ** 2)
    return distance


if __name__ == '__main__':
    src_path = 'data_selected'
    data_path = src_path + '/6hits/known_track/hits.csv'
    hits_volume = read_hits(data_path)
    hits = hits_volume[9]

    out_angles_path = "angles.json"
    angles = calculate_all_angles(hits, out_angles_path)
