# encoding: utf-8
import sys
import copy
import random

sys.path.append('../')
from utils import *

# globals
max_locs = 8606
seq_len = 48
X, Y = get_gps('../../data/raw/Cellular_Baselocation_baidu')

def get_points():
    distance_sorted_point = []
    for i in range(len(X)):
        distance = []
        for j in range(1, len(X)):
            dx = X[j] - X[i]
            dy = Y[j] - Y[i]
            distance.append([j, dx**2 + dy**2])
        distance.sort(key=lambda x: x[1], reverse=True)
        distance_sorted_point.append([d[0] for d in distance[:100]])
    return distance_sorted_point


def distance_distortion_er(real_data, neg_samples):
    distance_sorted_point = get_points()
    distorted_data = []
    for r in real_data:
        for i in range(neg_samples):
            dt = copy.deepcopy(r)
            j = random.randint(0, 5)
            while j < 48:
                location = r[j]
                _location = np.random.choice(distance_sorted_point[location])
                dt[j] = _location
                j += random.randint(0, 5)
            distorted_data.append(dt)
    return distorted_data


# write_data_to_file('../../data/twodays_p75/dispre_1.data', distance_distortion_er(rdata, 1))
# write_data_to_file('../../data/twodays_p75/dispre_10.data', distance_distortion_er(rdata, 10))