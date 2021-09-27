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



def prep_raw_data(data_num, data_dir_name):
    ori_data = read_ori_data('../../data/raw/ori.data')
    twodays_data = get_twodays_data(ori_data, filt_locs=True, filt_period=0.9)
    twodays_data = twodays_data[np.random.choice(
        twodays_data.shape[0], data_num * 3, replace=False)]

    twodays_data_train = twodays_data[:data_num]
    twodays_data_val = twodays_data[data_num: data_num*2]
    twodays_data_test = twodays_data[data_num*2:]

    os.makedirs(name=f'../../data/{data_dir_name}', exist_ok=True)

    write_data_to_file(f'../../data/{data_dir_name}/real.data', twodays_data_train)
    write_data_to_file(f'../../data/{data_dir_name}/val.data', twodays_data_val)
    write_data_to_file(f'../../data/{data_dir_name}/test.data', twodays_data_test)


def prepare_augmentation(data_dir_name):
    trajs = read_data_from_file(f'../../data/{data_dir_name}/real.data')
    write_data_to_file(f'../../data/{data_dir_name}/dispre_10.data', distance_distortion_er(trajs, 10))


def prepare_awareness(data_dir_name):
    trajs = read_data_from_file(f'../../data/{data_dir_name}/real.data')
    total_locations = 8606
    distance = IndividualEval.get_distances(trajs)
    period = IndividualEval.get_periodicity(trajs)
    gradius = IndividualEval.get_gradius(trajs)

    stat_start_dist(trajs, total_locations, data_dir_name, save=True)
    stat_visit_dist(trajs, total_locations, data_dir_name, save=True)

    np.save(f'../../data/{data_dir_name}/distance', distance)
    np.save(f'../../data/{data_dir_name}/period', period)
    np.save(f'../../data/{data_dir_name}/gradius', gradius)

    distance_matrix = np.load('../../data/stats/dist.npy')
    
    bins = 10000
    distribution, base = EvalUtils.arr_to_distribution(
        distance, min_distance, max_distance, bins)
    weights = np.zeros((total_locations, total_locations))
    for i in range(total_locations):
        for j in range(total_locations):
            index = bisect.bisect_left(base, distance_matrix[i][j])
            weights[i][j] = distribution[index] if index < bins - 1 else distribution[index - 1]
    np.save(f'../../data/{data_dir_name}/dweights.npy', weights)


if __name__ == '__main__':

    data_dir_name = 'total10000'
    total_num = 10000
    print("preparing raw data...")
    prep_raw_data(total_num, data_dir_name)
    print(f"finish prepare raw data to directory {data_dir_name}, total num: {total_num}")
    print("preparing augmented data for pretrain...")
    prepare_augmentation(data_dir_name)
    print(f"finish prepare augmented data to directory {data_dir_name}")
    print("preparing awareness weights...")
    prepare_awareness(data_dir_name)
    print(f"finish prepare awareness weights to directory {data_dir_name}")
    
