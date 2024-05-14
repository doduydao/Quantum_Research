from data import *
from read_data import *
import matplotlib.pyplot as plt


def read_truth_data(path):
    df = pd.read_csv(path)

    # print(df)
    return df

def display(hits, solution, out=""):
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

    for t, h in solution.items():
        for i in range(len(h) - 1):
            # ax.plot_lines(xdata=[], ydata=[ ], zdata=[], color='blue')  # Adjust color as desired
            ax.plot(xs=[h[i].x, h[i + 1].x], ys=[h[i].y, h[i + 1].y], zs=[h[i].z, h[i + 1].z], color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(out)
    plt.show()



if __name__ == '__main__':
    hits_path = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/volume_id_9/hits-vol_9_20_track.csv"

    hits_volume = read_hits(hits_path)
    hits = dict()
    for k, v in hits_volume.items():
        # print(k, v)
        print("Volume id:", k)
        print("No_layers:", len(v))
        hits = v
    layers = list(hits.keys())

    truth_data = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/event000001000-truth-wo-noise.csv"
    df = read_truth_data(truth_data)

    track = dict()
    for l, hp in hits.items():
        for i in range(len(hp)):
            id = hp[i].hit_id

            particle_id = df[df['hit_id'] == id]['particle_id'].values[0]
            # print(particle_id.values[0])
            hp[i].particle_id = particle_id
            if particle_id not in track:
                track[particle_id] = [hp[i]]
            else:
                track[particle_id] += [hp[i]]
    for t, h in track.items():
        h = sorted(h, key=lambda obj: obj.z)
        track[t] = h


    # out = "data.PNG"
    # solution = dict()
    # out = ""
    out = "/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/volume_id_9/proposal_result.PNG"
    display(hits, track, out)
