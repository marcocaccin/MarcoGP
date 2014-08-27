import os, pickle, random, time
from scipy.linalg import norm, eigh, inv, pinv, pinv2, pinvh
import numpy as np
from sklearn.gaussian_process import GaussianProcess


ttt = time.clock()
if not os.path.exists('qm7.pkl'): os.system('wget http://www.quantum-machine.org/data/qm7.pkl')
dataset = pickle.load(open('qm7.pkl','r'))

# --------------------------------------------
# Extract training data
# --------------------------------------------
split = 1
N_models = 11

P = dataset['P'][range(0,split)+range(split+1,5)].flatten()
X = dataset['X'][P]
T = dataset['T'][P]
Ptest  = dataset['P'][split]
Xtest = dataset['X'][Ptest]
Ttest = dataset['T'][Ptest]
print "TIMER load_data", time.clock() - ttt

# --------------------------------------------
# Extract feature(s) from training data
# --------------------------------------------
# in this case, only sorted eigenvalues of Coulomb matrix
ttt = time.clock()
eigX = [(eigh(M, eigvals_only=True))[::-1] for M in X]
eigt = [(eigh(M, eigvals_only=True))[::-1] for M in Xtest]
print "TIMER eval_features", time.clock() -ttt

# Observations
y = T.ravel()

alpha = []
for theta0 in np.linspace(0.01,1,N_models):

    # Setup a Gaussian Process model
    ttt = time.clock()
    gp = GaussianProcess(corr='absolute_exponential', theta0=theta0,
                         nugget=1e-3, random_start=100, verbose=True)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(eigX, y)
    print "TIMER teach", time.clock() - ttt

    # # Make the prediction on training set
    # y_pred, MSE = gp.predict(eigX, eval_MSE=True)
    # sigma = np.sqrt(MSE)
    # print('\n training set:')
    # print('MAE:  %5.2f kcal/mol'%np.abs(y_pred-y).mean(axis=0))
    # print('RMSE: %5.2f kcal/mol'%np.square(y_pred-y).mean(axis=0)**.5)
    # Make the prediction on test set
    y_pred, MSE = gp.predict(eigt, eval_MSE=True)
    sigma = np.sqrt(MSE)
    print('\n test set:')
    print('MAE:  %5.2f kcal/mol'%np.abs(y_pred-Ttest.ravel()).mean(axis=0))
    print('RMSE: %5.2f kcal/mol'%np.square(y_pred-Ttest.ravel()).mean(axis=0)**.5)
    alpha.append(gp.gamma)
