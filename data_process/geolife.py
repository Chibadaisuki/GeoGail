import os
import time

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
            traj_records.sort()
            for traj_record in traj_records:
                cur_year = traj_record[:4]
                cur_month = traj_record[4:6]
                cur_day = traj_record[6:8]
                date = f'{cur_year}-{cur_month}-{cur_day}'
                if date in data[uindex]:
                    arr = data[uindex][date]
                else:
                    arr = [[] for i in range(48)]
                with open(prefix + f'{uindex}/Trajectory/{traj_record}') as f:
                    recs = f.readlines()
                    for r in recs[6:]:
                        rsplit = r.split(',')
                        gps_x, gps_y = float(rsplit[0]), float(rsplit[1])
                        ttime = rsplit[-1]
                        hour = int(ttime.split(':')[0])
                        minute = int(ttime.split(':')[1])
                        if gps_x > 400:
                            continue
                        if gps_y < 0:
                            gps_y = -gps_y
                        arr[2 * hour + int(minute / 30)] = [gps_x, gps_y]
                data[uindex][date] = arr
    return data


def get_bound(data):
    locations = {}
    ID = 0
    for uindex, trajs in data.items():
        print(f"Currernt process user {uindex}")
        for locs in trajs.values():
            for xy in locs:
                if len(xy) == 0:
                    continue
                x = round(float(xy[0]), 3)
                y = round(float(xy[1]), 3)
                xf, yf = x % 1, y % 1
                xi, yi = x - xf, y - yf
                indexi = (xi + yi * 60) * 100
                indexf = round(xf + yf * 100, 3)
                index = indexi + indexf
                if index in locations:
                    locations[index][2] += 1
                else:
                    locations[index] = [x, y, 1, ID]
                    ID += 1
    return locations


def write_gps(locations):
    with open('../../data/geolife/Data/GPS', 'w') as f:
        for _, l in locations.items():
            f.write(str(l[0]))
            f.write(' ')
            f.write(str(l[1]))
            f.write('\n')

            
def get_trajs(data, locations):
    alltrajs = []
    for uindex, trajs in data.items():
        print(f"Currernt process user {uindex}")
        for locs in trajs.values():
            traj = []
            for xy in locs:
                if len(xy) == 0:
                    traj.append(-1)
                    continue
                x = round(float(xy[0]), 3)
                y = round(float(xy[1]), 3)
                xf, yf = x % 1, y % 1
                xi, yi = x - xf, y - yf
                indexi = (xi + yi * 60) * 100
                indexf = round(xf + yf * 100, 3)
                index = indexi + indexf
                l = locations[index]
                traj.append(l[-1])
            alltrajs.append(traj)
    return alltrajs


def simple_injection(trajs):
    new_trajs = []
    for P, traj in enumerate(trajs):
        print('Current processing: ', P)
        i = 0
        while i < 48:
            if traj[i] != -1:
                last = traj[i]
                i += 1
                continue
            j = i
            while j < 48 and traj[j] == -1:
                j += 1
            if j == 48:
                while i < j:
                    traj[i] = last
                    i += 1
            else:
                while i < j:
                    traj[i] = traj[j]
                    i += 1
        new_trajs.append(traj)
    return new_trajs


def write_trajs(trajs):
    with open('../../data/geolife/Data/real.data', 'w') as f:
        for i in range(len(trajs)):
            line = [str(p) for p in trajs[i]]
            line_s = ' '.join(line)
            f.write(line_s + '\n')
            

print("getting raw data...")
data = get_raw_data()
print("getting location bounds...")
locations = get_bound(data)

print("getting incomplete trajs...")
trajs = get_trajs(data, locations)
print("injecting trajs...")
ntrajs = simple_injection(trajs)

print("writing GPS...")
write_gps(locations)
print("writing trajs...")
write_trajs(ntrajs)