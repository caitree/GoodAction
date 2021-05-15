"""
Authors: Selwyn Gomes & Xu Cai

Code for noisy/noiseless version of synthetic/real-world functions
"""

import os
import sys
import argparse
import time
import numpy as np

from good_action.functions import *
from good_action.GPBO import GPBO
from good_action.utils import FUNC, ALGO

parser = argparse.ArgumentParser(description='Arguments of good-action identification.')
parser.add_argument("function", type=str, choices=['keane', 'shubert', 'syn1', 'syn2', \
                        'drop', 'hart3', 'hart6', 'ack6', 'robot3', 'robot4', 'xgbb'],\
                         help='specify the function to test')
parser.add_argument("noise", type=str, choices=['noisy', 'noiseless'], help='Whether the black-box function has noise')
parser.add_argument("eps", type=float, help='specify good action threshold')
parser.add_argument("--save_path", type=str, default='./', help='specify the save path for the results')
args = parser.parse_args()


NOISY = True if args.noise == 'noisy' else False
NUM_TRIAL = 25
NUM_EXP = 10
MAXSTEP = 200
N_INITS = 3
ALGOS = ['gpucb', 'pg', 'pi', 'eg', 'ei', 'ts', 'gs', 'mes', 'sts']


result_path = os.path.join(args.save_path, 'results')
if not os.path.exists(result_path):
    os.mkdir(result_path)

result_path = os.path.join(result_path, args.noise)
if not os.path.exists(result_path):
    os.mkdir(result_path)

result_path = os.path.join(result_path, FUNC[args.function])
if not os.path.exists(result_path):
    os.mkdir(result_path)

result_path = os.path.join(result_path, str(args.eps))
if not os.path.exists(result_path):
    os.mkdir(result_path)


func = eval(FUNC[args.function])(noisy=NOISY)
noiseless_func = eval(FUNC[args.function])(noisy=False)


func_bounds=func.bounds
epsilon = float(args.eps)

for j in range(NUM_TRIAL):
    X_init = np.random.uniform(func_bounds[:, 0], func_bounds[:, 1], size=(N_INITS, func_bounds.shape[0]))

    for algo in ALGOS:

        save_path = os.path.join(result_path, ALGO[algo])
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        log_path = os.path.join(save_path, str(args.function) + '_' + str(algo) + '_log.log')
        log = open(log_path, 'a')
        sys.stdout = log

        history_query_path = os.path.join(save_path, str(args.function) + '_' + str(algo) + '_query_history.npy')
        if not os.path.exists(history_query_path):
            history = {'queries' : [],
                    'returnvalue' : [],
                    'returnpoint': [],
                    'success' : []
                    }
        else:
            history = np.load(history_query_path, allow_pickle=True)[()]

        previous_trial = len(history['queries']) // NUM_EXP

        print(str(previous_trial+1) + 'th trial of ' + ALGO[algo])
        
        for i in range(NUM_EXP):
            start_time = time.time()
            print('\t' + str(i+1) + 'th experiment of ' + ALGO[algo])
            sys.stdout.flush()

            Bo_test= GPBO(func, func_bounds, algo, epsilon)
            Bo_test.initiate(X_init)
            queries = N_INITS

            Found = np.zeros(MAXSTEP, dtype=bool) if NOISY else False
            return_value = None

            if Bo_test.Y.max() >= epsilon and not NOISY:
                Found = True
            else:
                while queries < MAXSTEP:
                    queries += 1
                    x_val_ori, y_obs = Bo_test.sample_new_value()
                    if NOISY:
                        cur_mus = Bo_test.gp.predict(Bo_test.X_S)[0]
                        best_idx = cur_mus.argmax()
                        report_x = Bo_test.X[best_idx]
                        if args.function in ['robot3', 'robot4', 'xgbb']:
                            raise Exception("No noisy version")
                        else:
                            y_actual = np.array(noiseless_func(report_x.squeeze())).squeeze()
                        if y_actual >= epsilon:
                            Found[queries-1] = True
                    else:       
                        if y_obs >= epsilon:
                            Found = True
                            break
                    
            return_point = Bo_test.X
            return_value = Bo_test.Y
            if not NOISY:
                if Found:
                    print('\tFound a good action after ' + str(queries) + ' queries', ' best found was ' + str(Bo_test.Y.max()), sep=',')
                else:
                    print('\tDo not think a good action exsit after ' + str(queries) + ' queries', ' force stop', ' best found was ' + str(Bo_test.Y.max()), sep=',')
            else:
                print('\tBest found was ' + str(y_actual) + ' after '+ str(MAXSTEP) + ' queries', sep=',')


            current_time = time.time()
            print('\tRunning time : ', current_time - start_time)
            print('\n')

            history['queries'].append(queries)
            history['returnpoint'].append(return_point)
            history['returnvalue'].append(return_value)
            history['success'].append(Found)
            sys.stdout.flush()

            del Bo_test

        np.save(history_query_path, np.asarray(history))