"""
Authors: Xu Cai

Code for comparing GP-UCB and Elimination algorithm on lenient and standard regret
"""

import os
import argparse
import numpy as np

from good_action.GPBO import GPBO
from good_action.utils import FUNC, ALGO
from good_action.functions import Syn_1


parser = argparse.ArgumentParser(description='Arguments of good action identification.')
parser.add_argument("--eps", type=float, default=0.9, help='good action threshold')
args = parser.parse_args()


def cal_regret(beta_func, epsilon):
    func = Syn_1(noisy=True)
    noiseless_func = Syn_1(noisy=False)

    func_bounds=func.bounds

    ucb_standard_cumu_arr = np.zeros((N_EXP, LEN))
    ucb_indicator_cumu_arr = np.zeros((N_EXP, LEN))
    ucb_gap_cumu_arr = np.zeros((N_EXP, LEN))
    ucb_hinge_cumu_arr = np.zeros((N_EXP, LEN))

    elim_standard_cumu_arr = np.zeros((N_EXP, LEN))
    elim_indicator_cumu_arr = np.zeros((N_EXP, LEN))
    elim_gap_cumu_arr = np.zeros((N_EXP, LEN))
    elim_hinge_cumu_arr = np.zeros((N_EXP, LEN))

    elim_s_list = []
    ucb_s_list = []

    meshgrid = np.array(np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60)))
    Mt = meshgrid.reshape(2,-1).T
    f_max = noiseless_func(Mt).max()
    X_init = np.random.uniform(func_bounds[:, 0], func_bounds[:, 1], size=(N_INITS, func_bounds.shape[0]))
    for i in range(N_EXP):
        
        
        Bo_ucb=GPBO(func, func_bounds, 'gpucb', epsilon)
        Bo_ucb.gp.noise_delta=0.000001
        Bo_ucb.initiate(X_init)
        Bo_ucb.Mt = np.vstack([Mt, Bo_ucb.X_S])
        Bo_ucb.set_ls(0.1, 1)
        Bo_ucb.beta_func = beta_func

        Bo_elim=GPBO(func, func_bounds, 'elim', epsilon)
        Bo_elim.gp.noise_delta=0.000001
        Bo_elim.initiate(X_init)
        Bo_elim.Mt = np.vstack([Mt, Bo_elim.X_S])
        Bo_elim.set_ls(0.1, 1)
        Bo_elim.beta_func = beta_func
        
        for j in range(LEN):
            x_ucb = Bo_ucb.sample_new_value_ucb()
            y_ucb = np.array(noiseless_func(x_ucb.squeeze())).squeeze()

            x_elim = Bo_elim.sample_new_value_elimination()
            y_elim = np.array(noiseless_func(x_elim.squeeze())).squeeze()

            ucb_standard_regret = f_max - y_ucb
            ucb_indicator_regret = 1 if y_ucb < epsilon else 0
            ucb_gap_regret = ucb_standard_regret if y_ucb < epsilon else 0
            ucb_hinge_regret = (epsilon - y_ucb) if y_ucb < epsilon else 0

            ucb_standard_cumu_arr[i, j] = ucb_standard_cumu_arr[i, j-1] + ucb_standard_regret
            ucb_indicator_cumu_arr[i, j] = ucb_indicator_cumu_arr[i, j-1] + ucb_indicator_regret
            ucb_gap_cumu_arr[i, j] = ucb_gap_cumu_arr[i, j-1] + ucb_gap_regret
            ucb_hinge_cumu_arr[i, j] = ucb_hinge_cumu_arr[i, j-1] + ucb_hinge_regret


            elim_standard_regret = f_max - y_elim
            elim_indicator_regret = 1 if y_elim < epsilon else 0
            elim_gap_regret = elim_standard_regret if y_elim < epsilon else 0
            elim_hinge_regret = (epsilon - y_elim) if y_elim < epsilon else 0

            elim_standard_cumu_arr[i, j] = elim_standard_cumu_arr[i, j-1] + elim_standard_regret
            elim_indicator_cumu_arr[i, j] = elim_indicator_cumu_arr[i, j-1] + elim_indicator_regret
            elim_gap_cumu_arr[i, j] = elim_gap_cumu_arr[i, j-1] + elim_gap_regret
            elim_hinge_cumu_arr[i, j] = elim_hinge_cumu_arr[i, j-1] + elim_hinge_regret

            print("Experiment %i iter %i : elim num %i, y_ucb %f, y_elim %f " % (i,j,len(Bo_elim.Mt),y_ucb,y_elim))

        if elim_indicator_regret == 0:
            elim_s_list.append(i)
        
        if ucb_indicator_regret == 0:
            ucb_s_list.append(i)

        del Bo_ucb
        del Bo_elim

    print('elim success', elim_s_list)
    print('ucb success', ucb_s_list)
    return [ucb_standard_cumu_arr, ucb_indicator_cumu_arr, ucb_gap_cumu_arr, ucb_hinge_cumu_arr],\
            [elim_standard_cumu_arr, elim_indicator_cumu_arr, elim_gap_cumu_arr, elim_hinge_cumu_arr]


if __name__ == '__main__':
    MAXSTEP = 800
    N_INITS = 2
    N_EXP = 5
    LEN = MAXSTEP - N_INITS

    regrets_ucb, regrets_elim = cal_regret(lambda x : np.log(2*x)**1.5, float(args.eps))

    regrets = {'ucb' : regrets_ucb,
                'elim' : regrets_elim,
                }

    np.save('./syn1_lenient.npy', np.asarray(regrets))