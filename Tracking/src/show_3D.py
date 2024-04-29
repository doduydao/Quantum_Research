import matplotlib.pyplot as plt
import numpy as np
from read_data import *
# Fixing random state for reproducibility



def display(hits, solution):
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
    ax.scatter(xs, ys, zs, marker='o',  color='red')

    for t, h in solution.items():
        for i in range(len(h)-1):
            # ax.plot_lines(xdata=[], ydata=[ ], zdata=[], color='blue')  # Adjust color as desired
            ax.plot(xs=[h[i].x , h[i+1].x ], ys=[h[i].y , h[i+1].y], zs=[h[i].z , h[i+1].z], color='blue')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == '__main__':
    # hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/event000001000-hits.csv'
    hits_path = 'C:/Users/dddo/PycharmProjects/Quantum_Research/Tracking/event000001000/sel/event000001000-hits-sel-01.csv'
    hits = read_hits(hits_path)
    layers = list(hits.keys())
    tmp = []
    for p in layers:
        tmp.append(hits[p][0])

    solution = {'1':tmp}
    display(hits, solution)

