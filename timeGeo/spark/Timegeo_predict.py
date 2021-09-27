from pyspark import SparkContext
from Timegeo_core import *
import ast

# pwd = '/user/fib/fengjie/example/'
pwd = 'parameter/'
sc = SparkContext(appName='fengjie:Timegeo')
traces = sc.textFile(pwd+'trace_ready.txt')
rhythm = sc.textFile(pwd+'rhythm.txt')
deltar = sc.textFile(pwd+'deltar.txt')

traces_ready = traces.map(lambda x: ast.literal_eval(x.strip('\n')))
rhythm_global = sc.broadcast(rhythm.map(lambda x: ast.literal_eval(x.strip('\n'))).collect())
deltar_global = sc.broadcast(deltar.map(lambda x: ast.literal_eval(x.strip('\n'))).collect())

next_predict = traces_ready.map(lambda x: predict_next_place(x, rhythm_global, deltar_global))
next_place = next_predict.map(lambda x: [x['user_id'], x['stays'][-1]])
next_place.saveAsTextFile(pwd+'nextplace')

sc.stop()