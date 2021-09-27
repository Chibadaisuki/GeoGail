# encoding: utf-8
import os
import time
import json
from collections import Counter


def get_gps_bound():
    all_gps_x = []
    all_gps_y = []
    prefix = '../../data/geolife/Data/'
    users = range(182)
    users = [str(u).zfill(3) for u in users]
    for uindex in users:
        print(f"Currernt process user {uindex}")
        if os.path.exists(prefix + f'{uindex}/Trajectory'):
            traj_records = os.listdir(prefix + f'{uindex}/Trajectory')
            for traj_record in traj_records:
                with open(prefix + f'{uindex}/Trajectory/{traj_record}') as f:
                    recs = f.readlines()
                    for r in recs[6:]:
                        rsplit = r.split(',')
                        all_gps_x.append(float(rsplit[0]))
                        all_gps_y.append(float(rsplit[1]))
    return min(all_gps_x), max(all_gps_x), min(all_gps_y), max(all_gps_y)


print("Get gps bound...")
bound = get_gps_bound()
with open('../../data/geolife/geolife_gps_bound.txt') as f:
    json.dump(bound, f)
    

def grid(bound, gps_x, gps_y):
    minx, maxx = bound[0], bound[1]
    miny, maxy = bound[2], bound[3]
    x = int((gps_x - minx) * 100 / (maxx - minx))
    y = int((gps_y - miny) * 100 / (maxy - miny))
    return x * 100 + y


def get_raw_data():
    data = {}
    prefix = '../../data/geolife/Data/'
    users = range(182)
    users = [str(u).zfill(3) for u in users]
    for uindex in users:
        print(f"Currernt process user {uindex}")
        data[uindex] = {}
        if os.path.exists(prefix + f'{uindex}/Trajectory'):
            traj_records = os.listdir(prefix + f'{uindex}/Trajectory')
            last_date = None
            arr96 = [0 for i in range(96)]
            traj_records.sort()
            for traj_record in traj_records:
                cur_year = traj_record[:4]
                cur_month = traj_record[4:6]
                cur_day = traj_record[6:8]
                date = f'{cur_year}-{cur_month}-{cur_day}'
                tmparr48 = [[] for i in range(48)]
                if date in data[uindex]:
                    arr48 = data[uindex][date]
                else:
                    arr48 = [-1 for i in range(48)]
                with open(prefix + f'{uindex}/Trajectory/{traj_record}') as f:
                    recs = f.readlines()
                    for r in recs[6:]:
                        rsplit = r.split(',')
                        gps_x, gps_y = float(rsplit[0]), float(rsplit[1])
                        index = grid(bound, gps_x, gps_y)
                        ttime = rsplit[-1]
                        hour = int(ttime.split(':')[0])
                        minute = int(ttime.split(':')[1])
                        if minute <= 30:
                            tmparr48[2 * hour].append(index)
                        else:
                            tmparr48[2 * hour + 1].append(index)
                for i in range(48):
                    a = Counter(tmparr48[i]).most_common(1)
                    if a:
                        arr48[i] = a[0][0]
                data[uindex][date] = arr48
    return data


def get_twodays_data(raw):
    trajs = []
    for uindex, data in raw.items():
        print(f"Currernt process user {uindex}")
        last_data = None
        for date, ddata in data.items():
            if last_data:
                trajs.append(last_data + ddata)
                last_data = None
            else:
                last_data = ddata
    return trajs


print("Get raw data...")
data = get_raw_data()
with open('../../data/geolife/raw.json') as f:
    json.dump(data, f)

print("Get two days data...")
trajs = get_twodays_data(data)
with open('../../data/geolife/twodays.json') as f:
    json.dump(trajs, f)
