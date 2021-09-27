from pyspark import SparkContext
from Timegeo_core import *
import ast

performance_thresold = 0.005

def time_id(time_stamp):
    ti = stamp2array(time_stamp)
    return int(ti.tm_wday * 24 * 60 / time_slot + ti.tm_hour * 60 / time_slot + ti.tm_min / time_slot)


def predict_evaluate(info, rhythm_global, deltar_global):
    start_location = info['ground_truth'][0]
    end_location = info['ground_truth'][-1]
    start_time = start_location[1][2]
    end_time = end_location[1][3]
    ground_duration = (end_time - start_time) / 60 / time_slot

    n_w = info['n_w']
    beta1, beta2 = info['beta']
    deltar = deltar_global.value[0]
    rhythm = rhythm_global.value[0]
    location_type = ['home', 'work', 'other']

    location_history = {}
    for p in info['stays']:
        pid = str(p[1][0])[0:7] + ',' + str(p[1][1])[0:6]
        if pid in location_history.keys():
            location_history[pid][1] += 1
        else:
            location_history[pid] = [p[0], 1]
    s = len(location_history)
    for lo in location_history:
        location_history[lo][1] /= float(len(info['stays']))
    P_new = rho * s ** (-gamma)

    ground_trace = [0] * ground_duration
    for p in info['ground_truth']:
        p_in = p[1][2]
        p_out = p[1][3]
        bins = (p_out - p_in) / 60 / time_slot
        time_id_now = (p_in - start_time) / 60 / time_slot
        for t in range(time_id_now, time_id_now + bins):
            ground_trace[t] = [p[0], p[1][0], p[1][1], p_in + t * 60 * time_slot]

    predict_trace = [0] * ground_duration
    current_location = [start_location[0], start_location[1][0:2]]
    for tid in range(70, ground_duration):
        t = start_time + tid * 60 * time_slot
        ta = stamp2array(t)
        if ta.tm_hour in [2, 3, 4]:
            tmp = ground_trace[tid]
            predict_trace[tid] = tmp
            if tmp != 0:
                current_location = [tmp[0], [tmp[1], tmp[2]]]
        else:
            P_t = rhythm[time_id(t)]
            now_type, location_change = predict_next_place_time(n_w, P_t, beta1, beta2,
                                                                location_type[current_location[0]])
            if location_change:
                next_location = predict_next_place_location_simplify(P_new, deltar, location_history,
                                                                     current_location[1])
            else:
                next_location = [now_type, current_location[1]]
            current_location = next_location
            predict_trace[tid] = [next_location[0], next_location[1][0], next_location[1][1], t]

    predict_correct = 0
    for n in range(0, len(predict_trace)):
        if ground_trace[n] == 0 or predict_trace[n] == 0:
            continue
        elif distance(predict_trace[n][1:3], ground_trace[n][1:3]) < performance_thresold and stamp2array(
                t).tm_hour not in [2, 3, 4]:
            predict_correct += 1

    predict = 0
    ground = 0
    for i, j in zip(predict_trace, ground_trace):
        if i != 0:
            predict += 1
        if j != 0:
            ground += 1
    info['location_history'] = location_history
    info['predict_trace'] = predict_trace
    info['ground_trace'] = ground_trace
    info['performance'] = [predict_correct/float(ground), predict_correct, predict, ground]
    return info


pwd = '/user/fib/fengjie/telecom/'
pwd2 = 'noncommuter/'

sc = SparkContext(appName='fengjie:Timegeo')
traces = sc.textFile(pwd + pwd2 + 'trace_ready')
rhythm = sc.textFile(pwd + pwd2 + 'rhythm')
deltar = sc.textFile(pwd + pwd2 + 'deltar')

traces_ready = traces.map(lambda x: ast.literal_eval(x.strip('\n'))).filter(lambda x: len(x['ground_truth']) > 0)
rhythm_global = sc.broadcast(rhythm.map(lambda x: ast.literal_eval(x.strip('\n'))).collect())
deltar_global = sc.broadcast(deltar.map(lambda x: ast.literal_eval(x.strip('\n'))).collect())

traces_predict = traces_ready.map(lambda x: predict_evaluate(x, rhythm_global, deltar_global))
traces_predict.saveAsTextFile(pwd + pwd2 + 'performance')
