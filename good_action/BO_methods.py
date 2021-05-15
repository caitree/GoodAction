"""
Authors: Selwyn Gomes & Xu Cai

Implementations of acquisition functions
"""

import numpy as np
import scipy.optimize as spo
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.stats import norm
from torch.quasirandom import SobolEngine


class BO_methods(object):

    def __init__(self,model,acq_name,bounds,y_max,epsilon,X,Y,beta_sqrt):

        self.model=model
        self.acq_name = acq_name
        self.bounds=bounds
        self.y_max=y_max
        self.epsilon=epsilon
        self.dim = len(self.bounds)
        self.sobol = SobolEngine(self.dim, scramble=True)
        self.lb=np.array(self.bounds)[:,0]
        self.ub=np.array(self.bounds)[:,1]
        self.center=(self.lb +self.ub)/2 # The center of the unit domain
        self.X=X
        self.Y=Y
        self.beta_sqrt = beta_sqrt


    def method_val(self):
        
        if self.acq_name in 'pi' :
            x_init, acq_max = self.acq_maximize(self.pi_acq)
            x_return = self.multi_restart_maximize(self.pi_acq, x_init, acq_max)
            return x_return

        elif self.acq_name == 'pg' :
            x_init, acq_max = self.acq_maximize(self.pg_acq)
            x_return = self.multi_restart_maximize(self.pg_acq, x_init, acq_max)
            return x_return

        elif self.acq_name == 'ei' :
            x_init, acq_max = self.acq_maximize(self.ei_acq)
            x_return = self.multi_restart_maximize(self.ei_acq, x_init, acq_max)
            return x_return
        
        elif self.acq_name == 'eg' :
            x_init, acq_max = self.acq_maximize(self.eg_acq)
            x_return = self.multi_restart_maximize(self.eg_acq, x_init, acq_max)
            return x_return

        elif self.acq_name == 'gpucb' :
            x_init, acq_max = self.acq_maximize(self.gpucb_acq)
            x_return = self.multi_restart_maximize(self.gpucb_acq, x_init, acq_max)
            return x_return

        elif self.acq_name == 'ts' :
            x_init, acq_max = self.acq_maximize(self.ts_acq)
            x_return = self.multi_restart_maximize(self.ts_acq, x_init, acq_max)
            return x_return
        
        elif self.acq_name == 'sts' :
            x_tries= self.sobol.draw(100*self.dim).cpu().numpy()
            samples = self.model.sample(x_tries,size=1)
            valid_idx = np.where(samples >= self.epsilon)[0]

            if valid_idx.shape[0]:
                X_good=x_tries[valid_idx]
                origin= np.linalg.norm(X_good - self.center, axis=-1)
                return X_good[np.argmin(origin)]

            else:
                x_init, acq_max = self.acq_maximize(self.ts_acq)
                x_return = self.multi_restart_maximize(self.ts_acq, x_init, acq_max)
                return x_return

        elif self.acq_name == 'mes' :
            self.y_maxes = self.sample_maxes_G()
            x_tries= self.sobol.draw(100*self.dim).cpu().numpy()
            ys=np.array([])
            for x_try in x_tries:
                saved = self.mes_acq(x_try.reshape(1, -1))
                ys=np.append(ys,saved)
            x_init = x_tries[np.random.choice(np.where(ys == ys.max())[0])]
            acq_max = ys.max()

            x_return = self.multi_restart_maximize(self.mes_acq, x_init, acq_max)
            return x_return
            

        elif self.acq_name == 'gs' :
            x_tries= self.sobol.draw(100*self.dim).cpu().numpy()
            samples = self.model.sample(x_tries,size=100)

            if np.all(samples < self.epsilon):
                max_idx, _, _ = np.unravel_index(samples.argmax(), samples.shape)
                return x_tries[max_idx]

            else:
                ys = (samples >= self.epsilon).mean(axis=-1)
                x_init = x_tries[np.random.choice(np.where(ys == ys.max())[0])]
                acq_max = ys.max()

                x_return = self.multi_restart_maximize(self.gs_acq, x_init, acq_max)
                return x_return

        else:
            err = "The acquisition function " \
                  "{} has not been implemented, " \
                  "please choose one from the given list".format(acq_name)
            raise NotImplementedError(err)
    
        
    def pi_acq(self,x):
        mean, var = self.model.predict(x)
        std=np.sqrt(var)
        z = (mean - self.y_max)/np.maximum(std,1e-8)
        prob= norm.cdf(z)
        return prob
    
    def pg_acq(self,x):
        mean, var = self.model.predict(x)
        std=np.sqrt(var)
        z = (mean - self.epsilon)/np.maximum(std,1e-8)
        return z
    
    def ei_acq(self,x):
        mean, var = self.model.predict(x)
        std=np.sqrt(var)
        a = (mean - self.y_max)
        z = a / np.maximum(std,1e-8)
        improve= a * norm.cdf(z) + std * norm.pdf(z)
        return improve

    def eg_acq(self,x):
        mean, var = self.model.predict(x)
        std=np.sqrt(var)
        a = (mean - self.epsilon)
        z = a / np.maximum(std,1e-8)
        improve= a * norm.cdf(z) + std * norm.pdf(z)
        return improve

    def gpucb_acq(self,x):
        mean, var = self.model.predict(x)
        val= mean + self.beta_sqrt * np.sqrt(var)
        return val
    
    def ts_acq(self,x):
        return self.model.sample(x, size=1)

    def mes_acq(self,x):
        x = np.atleast_2d(x)
        mu,var = self.model.predict(x)
        std=np.sqrt(var)
        mu=mu.flatten()
        std=std.flatten()
        gamma_maxes = (self.y_maxes - mu) / np.maximum(std[:,None],1e-8)
        tmp = 0.5 * gamma_maxes * norm.pdf(gamma_maxes) / np.maximum(norm.cdf(gamma_maxes), 1e-8) - \
            np.log(np.maximum(norm.cdf(gamma_maxes), 1e-8))
        mes = np.mean(tmp, axis=1, keepdims=True)
        mes= np.nan_to_num(mes)
        return mes
    
    def gs_acq(self, x):
        samples = self.model.sample(x, size=100)
        gs_score = (samples >= self.epsilon)
        return gs_score.mean()

    
    # Gumble sampling for sampling max values in MES
    def sample_maxes_G(self):
        x_grid = self.sobol.draw(100*self.dim).cpu().numpy()
        mu,var = self.model.predict(x_grid)
        std = np.sqrt(var)

        def cdf_approx(z):
            z = np.atleast_1d(z)
            ret_val = np.zeros(z.shape)
            for i, zi in enumerate(z):
                ret_val[i] = np.prod(norm.cdf((zi - mu) / np.maximum(std,1e-8)))
            return ret_val

        lower = np.max(self.Y)
        upper = np.max(mu + 5*std)
        if cdf_approx(upper) <= 0.75:
            upper += 1

        grid = np.linspace(lower, upper, 100)

        cdf_grid = cdf_approx(grid)
        r1, r2 = 0.25, 0.75

        y1 = grid[np.argmax(cdf_grid >= r1)]
        y2 = grid[np.argmax(cdf_grid >= r2)]

        beta = (y1 - y2) / (np.log(-np.log(r2)) - np.log(-np.log(r1)))
        alpha = y1 + (beta * np.log(-np.log(r1)))
        maxes = alpha - beta*np.log(-np.log(np.random.rand(1000,)))
        return maxes
     
    # Thompsons sampling for finding max values in MES
    def sample_maxes_T(self):
        X_tries = self.sobol.draw(100*self.dim).cpu().numpy()
        samples = self.model.sample(X_tries,size=100)
        samples=samples.detach().cpu().numpy()
        maxs = np.max(samples, axis=0)
        percentiles = np.linspace(50, 95, 1000)
        reduced_maxes = np.percentile(maxs, percentiles)
        print(reduced_maxes)
        return reduced_maxes
    
    
    def acq_maximize(self,acq):
        x_tries = self.sobol.draw(1000).cpu().numpy()
        ys = acq(x_tries)
        x_max = x_tries[np.random.choice(np.where(ys == ys.max())[0])]
        acq_max = ys.max()
        return x_max, acq_max


    # Explore the parameter space more throughly
    def multi_restart_maximize(self, acq_func, x_max, acq_max, seed_num=10):
        x_seeds = self.sobol.draw(seed_num).cpu().numpy()
        for x_try in x_seeds:
            res = minimize(lambda x: -acq_func(x.reshape(1, -1)).squeeze(),\
                    x_try.reshape(1, -1),\
                    bounds=self.bounds,\
                    method="L-BFGS-B")
            if not res.success:
                continue
            if acq_max is None or -res.fun >= acq_max:
                x_max = res.x
                acq_max = -res.fun
        return x_max
    