import numpy as np


class Hit:
    def __init__(self, id, x, y, z, layer):
        self.id = id
        self.x = x
        self.y = y
        self.z = z # z sẽ có 4 giá trị khác nhau (lấy cái số 2)
        self.layer = layer

class Segment:
    def __init__(self, hit_1, hit_2):
        self.x = hit_2.x - hit_1.x
        self.y = hit_2.y - hit_1.y
        self.z = hit_2.z - hit_1.z

class Angle:
    def __init__(self, seg_1, seg_2):
        self.angle = self.calculate_angle(seg_1, seg_2)

    def calculate_angle(self, seg_1, seg_2):
        v1 = np.array([seg_1.x, seg_1.y, seg_1.z])
        v2 = np.array([seg_2.x, seg_2.y, seg_2.z])
        v1_normalized = v1 / np.linalg.norm(v1)
        v2_normalized = v2 / np.linalg.norm(v2)
        return np.arccos(np.dot(v1_normalized, v2_normalized))


