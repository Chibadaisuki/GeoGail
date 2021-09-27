from pyspark import SparkContext
from Timegeo_core import *

#pwd = '../trace_example/'
pwd = '/user/fib/fengjie/'
pwd1 = 'telecom/'
pwd2 = 'noncommuter/'
pwd3 = 'trace_clean/'
sc = SparkContext(appName='fengjie:Timegeo')
traces = sc.textFile(pwd + 'example/sh_traces.txt')

trace = traces.map(preprocessing)
trace_stay = trace.filter(lambda x: x['user_stay'] > 10)
trace_home = trace_stay.filter(lambda x: x['home_stay'] > 0)
trace_work = trace_home.filter(lambda x: x['work_stay'] > 0)

trace.saveAsTextFile(pwd+pwd1+pwd3+'orig')
trace_stay.saveAsTextFile(pwd+pwd1+pwd3+'stay')
trace_home.saveAsTextFile(pwd+pwd1+pwd3+'home')
trace_work.saveAsTextFile(pwd+pwd1+pwd3+'work')

rhythm_individual = trace_stay.map(global_rhythm)
rhythm = rhythm_individual.groupByKey().map(global_reduce)
rhythm.saveAsTextFile(pwd + pwd1+pwd2+'rhythm')
rhythm_global = sc.broadcast(rhythm.collect())

deltar_individual = trace_stay.map(global_displacement)
deltar = deltar_individual.groupByKey().map(global_reduce)
deltar.saveAsTextFile(pwd + pwd1+pwd2+'deltar')

trace_ready = trace_home.filter(lambda x: x['work_stay'] == 0).map(lambda x: simulate_traces(x, rhythm_global))
trace_ready.saveAsTextFile(pwd+pwd1+pwd2+'trace_ready')

sc.stop()
