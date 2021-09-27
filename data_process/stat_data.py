# -*- coding:utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def stat_single_period(data, time_unit):
    total_seq_len = data.shape[0]
    psum = 0.
    for j in range(int(time_unit)):
        p = data[range(j, total_seq_len, time_unit)]
        psum += len(set(p)) / len(p)
    psum = psum / time_unit
    return psum


def stat_single_locs(data):
    total_seq_len = data.shape[0]
    locs = len(set(data))
    return locs


def stat_periodicity(data, time_unit):
    """

    :param data: np.array
    :param time_unit: int, length of period
    :return: list, list of periodicity metrics
    """
    total_traj_num = data.shape[0]
    total_seq_len = data.shape[1]
    if total_seq_len % time_unit != 0:
        raise ValueError("incorrect time_unit!")

    plist = []
    for i in range(total_traj_num):
        psum = 0.
        for j in range(time_unit):
            p = data[i, range(j, total_seq_len, time_unit)]
            psum += len(set(p)) / len(p)
        psum = psum / time_unit
        plist.append(psum)

    plist = np.array(plist)
    return plist


def stat_distance(save=False):
    gps = []
    with open('../../data/raw/Cellular_Baselocation_baidu') as f:
        lines = f.readlines()
    for l in lines:
        gps.append([float(x) for x in l.split()])

    l = len(gps)
    dist = np.zeros(shape=(l, l), dtype=float)
    for i in range(l):
        for j in range(i, l):
            dx = gps[i][0] - gps[j][0]
            dy = gps[i][1] - gps[j][1]
            d = dx**2 + dy**2
            dist[j][i] = d
            dist[i][j] = d
    if save:
        np.save('../../data/stats/dist', dist)
    else:
        return dist


def vis_periodicity(periodicity_array):
    """

    :param periodicity_list: list, list of
    :return:
    """
    plt.title('Periodicity Visualization')
    sns.distplot(periodicity_array)
    plt.xlabel('Periodicity metrics')
    plt.ylabel('Count numbers')
    plt.show()


def stat_visit_dist(trajs, max_locs, data_dir_name, save=False):
    """
    """
    visit_dist = np.zeros(shape=(max_locs), dtype=float)
    for traj in trajs:
        for t in traj:
            visit_dist[t] += 1
    visit_dist = visit_dist / np.sum(visit_dist)
    if save:
        np.save(f'../../data/{data_dir_name}/visit.npy', visit_dist)
    else:
        return visit_dist


def stat_start_dist(trajs, max_locs, data_dir_name, save=False):
    """
    """
    start_dist = np.zeros(shape=(max_locs), dtype=float)
    for traj in trajs:
        start_dist[traj[0]] += 1
    start_dist = start_dist / np.sum(start_dist)
    if save:
        np.save(f'../../data/{data_dir_name}/start.npy', start_dist)


if __name__ == '__main__':
    pass
