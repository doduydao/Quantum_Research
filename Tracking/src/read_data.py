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

if __name__ == '__main__':
    hits_path = '/Users/doduydao/daodd/PycharmProjects/Quantum_Research/Tracking/event000001000/sel/event000001000-hits-sel-01.csv'
    hits = read_hits(hits_path)
    for k, v in hits.items():
        print(k, v)

    # layers = list(hits.keys())
    # for l in layers[:4]:
    #     hs = hits[l][:6]
    #     for h in hs:
    #         print(l, h.id)
    #     # print()



    # print(len(layers))
    # layer_name = layers[0]
    # print(len(hits[layer_name]))


