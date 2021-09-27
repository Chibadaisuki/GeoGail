# -*- coding:utf-8 -*-
import os
import sys
sys.path.append('../')

from evaluations import *
from stat_data import *
from get_data import *
from aug_dis import *
from utils import read_data_from_file, write_data_to_file
import numpy as np
import bisect


# globals
seq_len = 48
X, Y = get_gps('../../data/geolife/GPS')
max_locs = len(X)


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


def prepare_augmentation(data_dir_name):
    trajs = read_data_from_file(f'../../data/{data_dir_name}/real.data')
    write_data_to_file(f'../../data/{data_dir_name}/dispre_10.data', distance_distortion_er(trajs, 10))


def stat_start_dist(data_dir_name, save=False):
    """
    """
    trajs = read_data_from_file(f'../../data/{data_dir_name}/real.data')
    start_dist = np.zeros(shape=(max_locs), dtype=float)
    for traj in trajs:
        start_dist[traj[0]] += 1
    start_dist = start_dist / np.sum(start_dist)
    if save:
        np.save(f'../../data/{data_dir_name}/start.npy', start_dist)
        

if __name__ == '__main__':

    data_dir_name = 'geolife'
    total_num = 10000
    print("preparing augmented data for pretrain...")
    prepare_augmentation(data_dir_name)
    print(f"finish prepare augmented data to directory {data_dir_name}")
    print("preparing starting distribution of real data...")
    stat_start_dist(data_dir_name, save=True)
    print(f"finish prepare starting distribution of real data to directory {data_dir_name}")
