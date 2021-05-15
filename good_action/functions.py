"""
Authors: Selwyn Gomes & Xu Cai

Implementation of all the testing functions
"""

import numpy as np
from numpy import *
import math
from numpy.matlib import *
from scipy.stats import multivariate_normal


class Keane:
    def __init__(self, noisy=False):
        self.dim=2
        self.bounds=np.array([[-4., 4.]] * self.dim)
        self.noisy=noisy
        self.noise_std = 0.05
        self.max=1.0104

    def __call__(self,X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        out =  np.abs((np.cos(X[:,0])**4 + np.cos(X[:,1])**4 \
                    - 2 * (np.cos(X[:,0])**2) * (np.cos(X[:,1])**2))) \
                / np.sqrt(1*X[:,0]**2 + 1.5*X[:,1]**2)
        out *= 1.5

        if self.noisy:
            return out + np.random.normal(0, self.noise_std, size=(X.shape[0], ))
        else:
            return out


class Hartmann_3:
    def __init__(self, noisy=False):
        self.dim=3
        self.bounds=np.array([[0., 1.]] * self.dim)
        self.noisy=noisy
        self.noise_std = 0.05
        self.max = 3.86278

    def __call__(self,X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        c = array([1, 1.2, 3, 3.2])
        A = array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        P = array([[0.3689, 0.1170, 0.2673], 
                   [0.4699, 0.4387, 0.747], 
                   [0.1091, 0.8732, 0.5547],
                   [0.0382, 0.5743, 0.8828]])
        out = sum(c * exp(-sum(A * (repmat(X, 4, 1) - P) ** 2, axis = 1)))
        
        if self.noisy:
            return out + np.random.normal(0, self.noise_std)
        else:
            return out


class Syn_1:
    def __init__(self, noisy=False):
        import GPy
        self.dim=2
        self.bounds=np.array([[-3., 3.]] * self.dim)
        self.noisy=noisy
        self.noise_std = 0.02
        self.max = 1.55

        X_1 = np.asarray([[-2, -1.6],[-2, -2.2], [-1.2, -1.5], [-1.6, 0.6]])
        Y_1 = np.asarray([[0.6], [0.4], [0.3], [-0.4]])
        X_2 = np.asarray([[-0.7, -0.5], [-0.5, 0.3], [0.1, -0.3], [0.3, -1], [0.7, -0.6], [0.3, 0.1]])
        Y_2 = np.asarray([[-0.7], [0.7], [1], [-0.3], [0.1], [0.4]])
        X_3 = np.asarray([[2.1, -2], [1., 0.1]])
        Y_3 = np.asarray([[0.7], [-0.35]])
        X_4 = np.asarray([[1.7, 1.9], [0.5, 1.], [0.2, 1.3], [1.2, 1.4]])
        Y_4 = np.asarray([[0.9], [0.7], [0.5], [0.5]])
        X_5 = np.asarray([[-2.1, 1.8]])
        Y_5 = np.asarray([[-0.5]])
        X = np.vstack([X_1, X_2, X_3, X_4, X_5])
        Y = np.vstack([Y_1, Y_2, Y_3, Y_4, Y_5])

        kern_syn = GPy.kern.RBF(2, variance=1, lengthscale=(0.1, 0.15), ARD=True)
        self.gp = GPy.models.GPRegression(X, Y, kern_syn)
        self.gp.optimize()

    def __call__(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        out = self.gp.predict_noiseless(X)[0].squeeze()
        out *= 2.7
        
        if self.noisy:
            return out + np.random.normal(0, self.noise_std, size=(X.shape[0], ))
        else:
            return out


# class Syn_2:
#     def __init__(self, noisy=False):
#         import GPy
#         self.dim=2
#         self.bounds=np.array([[-4., 4.]] * self.dim)
#         self.noisy=noisy
#         self.noise_std = 0.02
#         self.max = 1.6606

#         X_1 = np.asarray([[-4, -1.6],[-3, -4.2], [-0.2, -1.5], [-2.6, 0.6]])
#         Y_1 = np.asarray([0.5, 0.4, 0.3, -0.1])

#         X_2 = np.asarray([[-0.7, -3.5], [-0.5, 0.3], [3.1, -0.3], [2.7, -0.6]])
#         Y_2 = np.asarray([-0.1, 0.8, 0.5, 0.1])

#         X_3 = np.asarray([[2.1, -2], [1.6, 0.1]])
#         Y_3 = np.asarray([1.6, -0.1])

#         X_4 = np.asarray([[2.9, 1.9]])
#         Y_4 = np.asarray([1.3])

#         X_5 = np.asarray([[-3.1, -2.0]])
#         Y_5 = np.asarray([1.1])

#         X = np.vstack([X_1, X_2, X_3, X_4, X_5])
#         Y = np.hstack([Y_1, Y_2, Y_3, Y_4, Y_5]).reshape(-1,1)

#         kern_syn = GPy.kern.RBF(2, variance=1, lengthscale=(0.3, 0.3), ARD=True)
#         self.gp = GPy.models.GPRegression(X, Y, kern_syn)
#         self.gp.optimize()

#     def __call__(self, X):
#         X = np.array(X)
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
        
#         if X.ndim == 1:
#             X = X[np.newaxis, :]

#         out = self.gp.predict_noiseless(X)[0].squeeze()
        
#         if self.noisy:
#             return out + np.random.normal(0, self.noise_std, size=(X.shape[0], ))
#         else:
#             return out



class Ackley_6:
    def __init__(self, noisy=False):
        self.dim=6
        self.bounds=np.array([[-32.768, 32.768]] * self.dim)
        self.noisy=noisy
        self.noise_std = 0.05
        self.max = 40.82

    def __call__(self,X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        out = []
        for _ in range(X.shape[0]):
            firstSum = 0.0
            secondSum = 0.0
            for c in X:
                firstSum += c**2.0
                secondSum += math.cos(2.0*math.pi*c)
            n = float(len(X))
            _out = 20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e
            out.append(_out)

        out = np.array(out)
        if self.noisy:
            return out + np.random.normal(0, self.noise_std, size=(X.shape[0], ))
        else:
            return out


class Alpine:
    def __init__(self, noisy=False):
        self.dim=6
        self.bounds=np.array([[0., 10.]] * self.dim)
        self.noisy=noisy
        self.noise_std = 0.05
        
    def __call__(self,X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        out = []
        for _ in range(X.shape[0]):
            fitness = 0
            for i in range(len(X)):
                fitness += math.fabs(0.1*X[i]+X[i]*math.sin(X[i]))
            out.append(-fitness)

        out = np.array(out)
        if self.noisy:
            return out + np.random.normal(0, self.noise_std, size=(X.shape[0], ))
        else:
            return out


class Eggholder:
    def __init__(self, noisy=False):
        self.dim=2
        self.bounds=np.array([[-512., 512.]] * self.dim)
        self.noisy=noisy
        self.noise_std = 0.05
        self.max=959.64

    def __call__(self,X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        func_val = -(X[:,1]+47) * np.sin(np.sqrt(abs(X[:,2]+X[:,1]/2+47))) \
                    + -X[:,1] * np.sin(np.sqrt(abs(X[:,1]-(X[:,2]+47))))
        out = - func_val

        if self.noisy:
            return out + np.random.normal(0, self.noise_std, size=(X.shape[0], ))
        else:
            return out


class Dropwave:
    def __init__(self, noisy=False):
        self.dim=2
        self.bounds=np.array([[-5.12, 5.12]] * self.dim)
        self.noisy=noisy
        self.noise_std = 0.05
        self.max=1
    def __call__(self,X):
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        
        fval = - (1+np.cos(12*np.sqrt(x1**2+x2**2))) / (0.5*(x1**2+x2**2)+2) 
        out = - fval.squeeze()

        if self.noisy:
            return out + np.random.normal(0,self.noise_std)
        else:
            return out


class Robot_Push_3D:
    def __init__(self, oshape = 'circle', osize = 1., ofriction = 0.01, \
                odensity = 0.05, bfriction = 0.01, hand_shape = 'rectangle', \
                hand_size  = (0.3, 1.), noisy=False ):
        from push_world import b2WorldInterface, make_thing, end_effector, simu_push

        global b2WorldInterface
        global make_thing
        global end_effector
        global simu_push


        self.oshape = oshape
        self.osize = osize
        self.ofriction = ofriction
        self.odensity = odensity
        self.bfriction = bfriction
        self.hand_shape = hand_shape
        self.hand_size = hand_size
        self.noisy = noisy

        self.goal = np.array([3., 4.])

        self.dim=3
        self.bounds=np.array([
                            [-5., 5.],          # x
                            [-5, 5],            # y
                            [1., 30.],          # push dist
                                ])

        self.max = 5.

    
    def _calc_dist(self, rx, ry, simu_steps):
        world = b2WorldInterface(False)
        thing, base = make_thing(500, 500, world, self.oshape, self.osize, self.ofriction, self.odensity, self.bfriction, (0,0))

        init_angle = np.arctan(ry/rx)
        robot = end_effector(world, (rx, ry), base, \
                                    init_angle, self.hand_shape, self.hand_size)
        ret = simu_push(world, thing, robot, base, simu_steps, self.noisy)
        del world
        dist = np.linalg.norm(self.goal - ret)
        dist = 5. - dist
        return dist

    def __call__(self, x):

        rx, ry, simu_steps = x[0], x[1], x[2]
        rx = np.float(rx)
        ry = np.float(ry)

        simu_steps = np.int(simu_steps * 10)

        dist = self._calc_dist(rx, ry, simu_steps)

        return dist


class Robot_Push_4D:
    def __init__(self, oshape = 'circle', osize = 1., ofriction = 0.01, \
                odensity = 0.05, bfriction = 0.01, hand_shape = 'rectangle', \
                hand_size  = (0.3, 1.), noisy=False):
        from push_world import b2WorldInterface, make_thing, end_effector, simu_push2

        global b2WorldInterface
        global make_thing
        global end_effector
        global simu_push2

        self.oshape = oshape
        self.osize = osize
        self.ofriction = ofriction
        self.odensity = odensity
        self.bfriction = bfriction
        self.hand_shape = hand_shape
        self.hand_size = hand_size
        self.noisy = noisy


        self.goal = np.array([3., 4.])


        self.dim=4
        self.bounds=np.array([
                            [-5., 5.],          # x
                            [-5, 5],            # y
                            [0., 2*math.pi],    # angle
                            [1., 30.],          # push dist
                                ])

        self.max = 5.
    
    def _calc_dist(self, rx, ry, xvel, yvel, init_angle, simu_steps):
        world = b2WorldInterface(False)
        thing, base = make_thing(500, 500, world, self.oshape, self.osize, self.ofriction, self.odensity, self.bfriction, (0,0))

        robot = end_effector(world, (rx, ry), base, \
                            init_angle, self.hand_shape, self.hand_size)
        ret = simu_push2(world, thing, robot, base, xvel, yvel, simu_steps, self.noisy)
        del world

        dist = np.linalg.norm(self.goal - ret)
        dist = 5. - dist
        return dist


    def __call__(self, x):
        
        rx, ry, init_angle, simu_steps = x[0], x[1], x[2], x[3]

        rx = np.float(rx)
        ry = np.float(ry)
        init_angle = np.float(init_angle)

        simu_steps = np.int(simu_steps * 10)

        xvel = -rx
        yvel = -ry
        regu = np.linalg.norm([xvel, yvel])
        xvel = xvel / regu * 10
        yvel = yvel / regu * 10

        dist = self._calc_dist(rx, ry, xvel, yvel, init_angle, simu_steps)

        return dist



class XGB_Boston():
    def __init__(self, noisy=False):
        import xgboost as xgb
        from sklearn import datasets

        self.noisy = noisy
        self.dim=5
        self.max=10

        self.bounds=np.array([
                            [2, 15],        # max_depth
                            [0.01, 0.3],    # learning_rate
                            [0, 10],        # max_delta_step
                            [0, 1],         # colsample_bytree
                            [0, 1],         # subsample
                            # [1, 20],        # min_child_weight
                            # [0, 10],        # gamma
                            # [0, 10],        # reg_alpha
                                ])

        X = datasets.load_boston()
        Y = X.target
        self.data_dmatrix = xgb.DMatrix(data=X['data'],label=Y)

    def __call__(self, x):
        max_depth, lr, max_delta_step, colsample_bytree, subsample = x

        params =  {
                    'objective': 'reg:squarederror',
                    'max_depth': int(max_depth),
                    'learning_rate': lr,
                    'max_delta_step': int(max_delta_step),
                    'colsample_bytree': colsample_bytree,
                    'subsample' : subsample
                    }

        cv_results = xgb.cv(params=params, 
                            dtrain=self.data_dmatrix, 
                            nfold=3, 
                            seed=3,
                            num_boost_round=50000,
                            early_stopping_rounds=50,
                            metrics='rmse')

        return 10 - cv_results['test-rmse-mean'].min()
