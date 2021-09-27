from pyspark import SparkContext
import ast


def extract_nw(info):
    return 'nw', [info['n_w'], info['beta'][0], info['beta'][1]]


def extract_reduce(pairs):
    df = {'nw': [0] * 20, 'nw_beta1': [0] * 200, 'nw_beta2': [0] * 2000}
    for i, p in enumerate(list(pairs[1])):
        df['nw'][min(int(round(p[0])), 19)] += 1
        df['nw_beta1'][min(int(round(p[0] * p[1])), 199)] += 1
        df['nw_beta2'][min(int(round(p[0] * p[2])), 1999)] += 1
    for n in df:
        sum_df = float(sum(df[n]))
        df[n] = [x / sum_df for x in df[n]]
    return df


# pwd = '/user/fib/fengjie/example/'
pwd = 'parameter/'
sc = SparkContext(appName='fengjie:Timegeo')
traces = sc.textFile(pwd + 'trace_ready.txt')

traces_ready = traces.map(lambda x: ast.literal_eval(x.strip('\n')))
traces_ready.map(extract_nw).groupByKey().map(extract_reduce).saveAsTextFile(pwd + 'nw_beta')

sc.stop()