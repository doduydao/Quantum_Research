import pandas as pd
from data import Hit
import random

def read_hits(path):
    df = pd.read_csv(path)
    # print(df)
    list_df = [row.tolist() for index, row in df.iterrows()]
    layers = dict()
    for i in list_df:
        hit = Hit(id=i[0],
                  x=i[1],
                  y=i[2],
                  z=i[3],
                  layer=i[5]
                  )
        layer = int(hit.layer)
        if layer not in layers:
            layers[layer] = [hit]
        else:
            layers[layer] += [hit]

    return layers

def get_data_from_sublayer(hits, sublayer):
    new_hits = dict()
    for p, hp in hits.items():

        sub = dict()
        for h in hp: # Hp : tập các hit có trên cùng 1 layer
            if h.z not in sub:
                sub[h.z] = [h]
            else:
                sub[h.z] += [h]
        sub = list(sorted(sub.items(), key=lambda x:x[0]))
        for sublayer in sub:
            print(sublayer)
        # print(subs)
        print()
        break
        new_hits[p] = new_hp

    return hits


if __name__ == '__main__':
    hits_path = 'C:\\Users\dddo\PycharmProjects\Quantum_Research\Tracking\event000001000\event000001000-hits.csv'
    hits = read_hits(hits_path)
    # for k, v in hits.items():
    #     print(k, v)
    get_data_from_sublayer(hits, 2)
    # layers = list(hits.keys())
    # for l in layers[:4]:
    #     hs = hits[l][:6]
    #     for h in hs:
    #         print(l, h.id)
    #     # print()



    # print(len(layers))
    # layer_name = layers[0]
    # print(len(hits[layer_name]))


