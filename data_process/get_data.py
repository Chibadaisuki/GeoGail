# -*- coding:utf-8 -*-
import sys
sys.path.append('../')

from utils import write_data_to_file
from stat_data import stat_single_period, stat_single_locs
import numpy as np



# globals
weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', "Sat"]


def read_ori_data(fp):
    with open(fp) as f:
        ori_data_lines = f.readlines()
    ori_data = []
    for ori_data_line in ori_data_lines:
        data = ori_data_line.split()
        ori_data.append([int(d) - 1 for d in data])
    ori_data = np.array(ori_data, dtype=np.int)
    return ori_data


def split_weekday_data(data):
    """

    :param data: np.array
    :return: list
    """
    weekday_data = []
    for i in range(7):
        wdata = data[:, i * 48: (i + 1) * 48]
        weekday_data.append(wdata)
    return weekday_data


def get_twodays_data(data, save_locs=False, filt_period=0.75, filt_locs=False):
    """

    :param data: np.array
    :param save_locs
    :param filt_period
    :param filt_locs
    :return: np.array
    """
    data = data[:, 96: 192]
    total_traj_num = data.shape[0]
    total_seq_len = data.shape[1]
    twodays_data = []
    for i in range(total_traj_num):
        psum = stat_single_period(data[i], 48)
        if psum <= filt_period:
            twodays_data.append(data[i])
    twodays_data = np.array(twodays_data, dtype=int)

    if save_locs:
        locs = []
        for t in twodays_data:
            locs.append(stat_single_locs(t))
        np.save('../../data/stats/twodays_locs', np.array(locs))

    if filt_locs:
        d = twodays_data
        twodays_data = []
        for t in d:
            locs = stat_single_locs(t)
            if locs >= 10:
                twodays_data.append(t)
        twodays_data = np.array(twodays_data)

    # down sample
    twodays_data = twodays_data[:, range(0, total_seq_len, 2)]
    return twodays_data


def main():
    ori_data = read_ori_data('../../data/raw/ori.data')
    twodays_data = get_twodays_data(ori_data, filt_locs=True, filt_period=0.9)
    twodays_data = twodays_data[np.random.choice(
        len(twodays_data), 30000, replace=False)]

    twodays_data_train = twodays_data[:10000]
    twodays_data_val = twodays_data[10000: 20000]
    twodays_data_test = twodays_data[20000:]

    write_data_to_file('../../data/twodays_p75/real.data', twodays_data_train)
    write_data_to_file('../../data/twodays_p75/val.data', twodays_data_val)
    write_data_to_file('../../data/twodays_p75/test.data', twodays_data_test)


if __name__ == '__main__':
    main()
