from warnings import catch_warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from numpy.random import random
from numpy import argmax
from numpy import asarray
from numpy import vstack
from warnings import simplefilter


from layers.nonlinearrecurrent import objective_function, generate_random_samples

class BayesianOptimization:
    def __init__(self, num_iter, objective_func, generate_random_samples, num_samples, num_params, params):
        self.num_iter = num_iter
        self.objective_func = objective_func
        self.generate_random_samples = generate_random_samples
        self.num_samples = num_samples
        self.num_params = num_params
        self.params = params
        self.model = GaussianProcessRegressor()
    
    #approximation for the actual objective function
    def surrogate(self, X):
        with catch_warnings():
            simplefilter('ignore')
            return self.model.predict(X, return_std=True)
    
    #probability of improvement acquisition function
    def acquisition(self, X, Xsamples):
        yhat, _ = self.surrogate(X)
        best = max(yhat)

        mu, std = self.surrogate(Xsamples)
        print(mu)
        mu = mu[:,0]

        eps = 1e-9
        prob_of_imp = norm.cdf((mu - best) / (std + eps))
        return prob_of_imp
    
    def optimize_acquisition(self, X):
        Xsamples = self.generate_random_samples(self.num_samples, self.num_params, self.params)

        #Xsamples = Xsamples.reshape(len(Xsamples), 1)

        scores = self.acquisition(X, Xsamples)

        ix = argmax(scores)
        return Xsamples[ix]
    
    def tune_hyperparams(self):
        X = self.generate_random_samples(self.num_samples, self.num_params, self.params)
        print('Samples generated', X)
        y = asarray([self.objective_func(x) for x in X])
        print('X evaluated', y)

        #X = X.reshape(X.shape[1], 1)
        y = y.reshape(len(y), 1)

        self.model.fit(X, y)
        print('Model fitted')

        for _ in range(self.num_iter):
            x = self.optimize_acquisition(X)
            print('acquisition optimized', x)

            actual = self.objective_func(x)
            print('objective function evaluated')

            est, _ = self.surrogate([x])
            print(x, est, actual)

            X = vstack((X, [x]))
            y = vstack((y, [[actual]]))

            self.model.fit(X, y)
        
        ix = argmax(y)
        print('Best Result:')
        print(X[ix], y[ix])

num_samples = 5
num_params = 6
params = [[0.1, 1.0, 0.1],[0.01, 0.1, 0.01],[0.1,1.0,0.1],[10, 100, 10],[1,10,1],[1,4,1]]


BO = BayesianOptimization(5, objective_function, generate_random_samples, num_samples, num_params, params)

BO.tune_hyperparams()