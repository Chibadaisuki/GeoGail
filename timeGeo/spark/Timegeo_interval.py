from pyspark import SparkContext
from Timegeo_core import *

pwd = '/user/fib/fengjie/'


def interval_statis(info):
    traces = info['stays']
    interval = [0] * 24

    for p in traces:
        time_start = stamp2array(p[1][2]).tm_hour
        time_end = stamp2array(p[1][3]).tm_hour
        if time_end < time_start:
            for i in range(time_start, 24):
                interval[i] += 1
            for i in range(0, time_end):
                interval[i] += 1
        else:
            for i in range(time_start, time_end + 1):
                interval[i] += 1
    return 'interval', interval


if __name__ == '__main__':
    sc = SparkContext(appName='fengjie:Timegeo')
    traces_telecom = sc.textFile(pwd + 'example/sh_traces.txt').map(preprocessing).filter(
        lambda x: x['user_stay'] > stay_threshold)
    traces_tencent = sc.textFile(pwd + 'tencent/wx_traces.txt').map(preprocessing).filter(
        lambda x: x['user_stay'] > stay_threshold)

    traces_telecom.map(interval_statis).groupByKey().map(global_reduce).saveAsTextFile(pwd + 'telecom/interval')
    traces_tencent.map(interval_statis).groupByKey().map(global_reduce).saveAsTextFile(pwd + 'tencent/interval')
    sc.stop()
