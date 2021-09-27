import sys
from Timegeo_core import *


def read_from_input(file):
    for line in file:
        yield line


def main(rhythm_global, deltar_global,map_mode):
    for line in read_from_input(sys.stdin):
        info = eval(line)
        if map_mode == 'predict':
            if len(info['ground_truth']) > 0:
                predict = predict_evaluate(info, rhythm_global, deltar_global, run_mode='streaming')
                print predict['performance']
            else:
                sys.stderr.write('Gound-Truth is NULL!   ' + info['user_id'] + '\n')
        elif map_mode == 'simulate':
            if len(info['ground_truth']) > 0:
                info2 = simulate_traces(info, rhythm_global,run_mode='streaming')
                print info2
            else:
                sys.stderr.write('Simulate Failed!   ' + info['user_id'] + '\n')


if __name__ == '__main__':
    rhythm = eval(open('code/rhythm').readline())
    deltar = eval(open('code/deltar').readline())
    main(rhythm_global=rhythm, deltar_global=deltar,map_mode=sys.argv[1])
