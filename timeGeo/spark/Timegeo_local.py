from Timegeo_core import *


class rhythm_broadcast(object):
    pass


class deltar_broadcast(object):
    pass


pwd = './trace_example/'
with open(pwd + 'sh_traces.txt', 'r')as fid, open(pwd + 'results.txt', 'w') as wid:
    traces = []
    rhythm = [0] * 7 * 24 * int(60.0/time_slot)
    # prepossing
    print "prepossing"
    for line in fid:
        info = preprocessing(line)
        rhythm_tmp = global_rhythm(info)
        for i, r in enumerate(rhythm_tmp[1]):
            rhythm[i] += r
        if filter_active(info):
            traces.append(info)
    # rhythm
    print "rhythm"
    rhythm_sum = float(sum(rhythm))
    for i, r in enumerate(rhythm):
        rhythm[i] /= rhythm_sum
    print rhythm
    wid.write(str(rhythm) + '\n')
    rhythm_broadcast.value = [rhythm]
    # beta
    print "beta"
    traces_beta = []
    deltar = [0] * max_explore_range
    for info in traces:
        info_beta = simulate_traces(info, rhythm_broadcast)
        print 'uid:' + str(info_beta['user_id']) + ' ' + 'beta:' + str(info_beta['beta'])
        traces_beta.append(info_beta)
        deltar_tmp = global_displacement(info)
        for i, r in enumerate(deltar_tmp[1]):
            deltar[i] += r
    # delta_r
    print "delta_r"
    deltar_sum = float(sum(deltar))
    for i, r in enumerate(deltar):
        deltar[i] /= deltar_sum
    print deltar
    wid.write(str(deltar) + '\n')
    deltar_broadcast.value = [deltar]
    # predict
    print "predict"
    for info in traces_beta:
        info_predict = predict_next_place(info, rhythm_broadcast, deltar_broadcast)
        wid.write(info_predict['user_id'] + ',' + info_predict['stays'][-1])