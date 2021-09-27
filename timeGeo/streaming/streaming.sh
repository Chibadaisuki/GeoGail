#!/bin/bash

hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.5.jar\
 -D mapreduce.job.queuename='usera'\
 -D mapreduce.job.name='fengjie:streaming'\
 -D mapreduce.job.reduces=0\
 -files code\
 -input /user/fib/fengjie/streaming/simulate\
 -output /user/fib/fengjie/streaming/predict\
 -mapper "python code/eval_map.py predict"\
 #-reducer "python code/reduce.py deltar"\
 #-numReduceTasks 1
