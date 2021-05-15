"""
Authors: Selwyn Gomes & Xu Cai

Class implementation for general Gaussian process bandit optimization
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from .BO_methods import BO_methods
from .GP import GaussianProcess, unique_rows


class GPBO:
     def __init__(self, func, bounds, acq_name,epsilon):
          self.X = None # The sampled point in original domain
          self.Y = None  # original output
          self.X_S=None   # scaled output (The input  is scaled [0,1] in all D)
          self.Y_S=None   # scaled inpout ( output is scaled as  (Y - mu) / sigma )
          self.bounds=bounds
          self.dim = len(bounds) # original bounds
          self.bounds_s=np.array([np.zeros(self.dim), np.ones(self.dim)]).T  # scaled bounds
          self.func = func
          self.acq_name = acq_name
          self.epsilon = epsilon # original epsilon
          self.epsilon_s=0  # Scaled epsilon. The value keeps changing after normalizing at each iteration
          scaler = MinMaxScaler()  # Tranform for moving from orignal to scaled vales
          scaler.fit(self.bounds.T)
          self.Xscaler=scaler
          self.gp=GaussianProcess(self.bounds_s, verbose=0)

          self.beta_func = lambda x: np.sqrt(np.log(x))
          
     
     def initiate(self, X_init):
          self.X = np.asarray(X_init[0])
          self.X_S = self.Xscaler.transform(X_init[0].reshape((1, -1)))
          y_init=self.func(X_init[0])
          self.Y = np.asarray(y_init)
          for i in range (1, X_init.shape[0]):
               self.X=np.vstack((self.X, X_init[i]))
               x_s=self.Xscaler.transform(X_init[i].reshape((1, -1)))
               self.X_S = np.vstack((self.X_S, x_s))
               self.Y = np.append(self.Y, self.func(X_init[i]))
          self.Y_S=(self.Y-np.mean(self.Y))/np.std(self.Y)      
          self.epsilon_s=(self.epsilon-np.mean(self.Y))/np.std(self.Y)

          self.gp.fit(self.X_S, self.Y_S)
     

     def set_ls(self,lengthscale,variance):
          """
          Manually set the GP hyperparameters
          """
          self.gp.set_hyper(lengthscale,variance)  


     def sample_new_value(self, learn=True):
          """
          Sample the next best query point based on historical observations
          """
          ur = unique_rows(self.X_S)
          self.gp.fit(self.X_S[ur], self.Y_S[ur])
          if  len(self.Y)%(3)==0 and learn:
               self.gp.optimise()

          y_max=max(self.Y_S)
          query_num=len(self.Y_S)

          objects = BO_methods(self.gp, self.acq_name, self.bounds_s, y_max,\
                         self.epsilon_s, self.X_S, self.Y_S, self.beta_func(query_num))
          x_val = objects.method_val()

          x_val_ori=self.Xscaler.inverse_transform(np.reshape(x_val,(-1,self.dim)))
     
          y_obs= self.func(x_val_ori[0]) 
          self.X_S = np.vstack((self.X_S, x_val.reshape((1, -1))))
          self.X=np.vstack((self.X, x_val_ori))
          self.Y = np.append(self.Y, y_obs)
          self.Y_S=(self.Y-np.mean(self.Y))/np.std(self.Y)
          self.epsilon_s=(self.epsilon-np.mean(self.Y))/np.std(self.Y)

          return x_val_ori, y_obs


     def sample_new_value_elimination(self):
          """
          Discretized version of the elimination algorithm
          """
          if self.Mt.shape[0] != 0:
               counts = len(self.Y_S)
               beta_sqrt = self.beta_func(counts)

               _, var_tm1 = self.gp.predict(self.Mt)

               var_tm1 = var_tm1.squeeze()
               select_idx = np.random.choice(np.where(var_tm1 == var_tm1.max())[0])
               x_val = self.Mt[select_idx]

               x_val_ori=self.Xscaler.inverse_transform(np.reshape(x_val,(-1,self.dim)))
               y_obs= self.func(x_val_ori[0]) 
               self.X_S = np.vstack((self.X_S, x_val.reshape((1, -1))))
               self.X=np.vstack((self.X, x_val_ori))
               self.Y = np.append(self.Y, y_obs)
               self.Y_S=(self.Y-np.mean(self.Y))/np.std(self.Y)

               ur = unique_rows(self.X_S)
               self.gp.fit(self.X_S[ur], self.Y_S[ur])
               # self.gp.optimise()

               mu_t, var_t = self.gp.predict(self.Mt)
               mu_t = mu_t.squeeze()
               std_t = np.sqrt(var_t).squeeze()

               ucb_t = mu_t + beta_sqrt * std_t
               lcb_t = mu_t - beta_sqrt * std_t
               max_lcb_score = lcb_t.max()

               preserve_idx = np.where(ucb_t >= max_lcb_score)[0]
               self.Mt = self.Mt[preserve_idx]

               return x_val_ori

          else:
               return self.X[self.Y.argmax()]
     
     def sample_new_value_ucb(self):
          """
          Discretized version of the GP-UCB algorithm
          """
          ur = unique_rows(self.X_S)
          self.gp.fit(self.X_S[ur], self.Y_S[ur])

          query_num = len(self.Y_S)
          beta_sqrt = self.beta_func(query_num)

          mu, var = self.gp.predict(self.Mt)
          
          mu = mu.squeeze()
          std = np.sqrt(var).squeeze()

          ucb_scores = mu + beta_sqrt * std

          x_val = self.Mt[ucb_scores.argmax()]

          x_val_ori=self.Xscaler.inverse_transform(np.reshape(x_val,(-1,self.dim)))
          y_obs= self.func(x_val_ori[0]) 
          self.X_S = np.vstack((self.X_S, x_val.reshape((1, -1))))
          self.X=np.vstack((self.X, x_val_ori))
          self.Y = np.append(self.Y, y_obs)
          self.Y_S=(self.Y-np.mean(self.Y))/np.std(self.Y)

          return x_val_ori
