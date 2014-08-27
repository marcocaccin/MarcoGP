#!/usr/bin/env python

import os, pickle, random, time
try:
    os.remove('my_gp_module.pyc')
except OSError:
    pass
import scipy as sp
from scipy.linalg import eigh
from my_gp_module import GaussianProcess
import matplotlib.pyplot as plt


# --------------------------------------------
# Parameters for the run
# --------------------------------------------
do_predict = False
split = 1
N_models = 1
theta0 = 10.0
Ntest = 100

# --------------------------------------------
# Load all database
# --------------------------------------------
ttt = time.clock()
if not os.path.exists('qm7.pkl'): os.system('wget http://www.quantum-machine.org/data/qm7.pkl')
dataset = pickle.load(open('qm7.pkl','r'))

# --------------------------------------------
# Extract training data and test set
# --------------------------------------------
allP = dataset['P'][range(0,split)+range(split+1,5)].flatten()
print "TIMER load_data", time.clock() - ttt
nteach = sp.int32(sp.exp(sp.linspace(sp.log(2*Ntest), sp.log(allP.size), 25)))

# --------------------------------------------
# Loop over different training set sizes
# --------------------------------------------
alpha = []
alpha_std = []
mae_error = []
for Nteach in nteach:

    # --------------------------------------------
    # First time include the test set to calculate their alpha
    # --------------------------------------------    
    print "\n", "-"*60, "\n"
    print "N teach = %d" % Nteach
    # Select training data
    P = allP[:Nteach]
    X = dataset['X'][P]
    T = dataset['T'][P]
    # --------------------------------------------
    # Extract feature(s) from training data and test set
    # --------------------------------------------
    # in this case, only sorted eigenvalues of Coulomb matrix
    ttt = time.clock()
    eigX = [(eigh(M, eigvals_only=True))[::-1] for M in X]
    print "TIMER eval_features", time.clock() - ttt
    # Observations
    y = T.ravel()

    # Setup a Gaussian Process model
    ttt = time.clock()
    gp = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0]),
                         nugget=1e-3, verbose=True, normalise=True, do_features_projection=False, low_memory=False)

    # Fit to data
    gp.fit(eigX, y)
    print "TIMER teach", time.clock() - ttt
    local_alpha = gp.alpha[:Ntest]
    print "alpha STD: %f" % sp.std(local_alpha)
    print "alpha MAV: %f" % sp.mean(sp.absolute(local_alpha))
    # alpha.append(local_alpha.flatten())
    alpha_std.append(sp.std(local_alpha))


    # --------------------------------------------
    # Second time dont' include the test set and predict
    # --------------------------------------------    
    print "\n", "-"*60, "\n"
    # Select training data
    X = dataset['X'][P]
    T = dataset['T'][P]
    # --------------------------------------------
    # Extract feature(s) from training data and test set
    # --------------------------------------------
    # in this case, only sorted eigenvalues of Coulomb matrix
    eigt = eigX[:Ntest]
    eigX = eigX[Ntest:]

    # Observations
    y = T.ravel()[Ntest:]
    y_test = T.ravel()[:Ntest]
    # Setup a Gaussian Process model
    ttt = time.clock()
    gp = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0]),
                         nugget=1e-3, verbose=True, normalise=True, do_features_projection=False, low_memory=False)

    # Fit to data
    gp.fit(eigX, y)
    print "TIMER teach", time.clock() - ttt

    ttt = time.clock()

    # Make the prediction on test set
    y_pred, MSE = gp.predict(eigt, eval_MSE=True)
    sigma = sp.sqrt(MSE)
    mae_error.append(sp.absolute(y_pred-y_test).mean(axis=0))
    print('\n test set:')
    print('MAE:  %5.2f kcal/mol' % sp.absolute(y_pred-y_test).mean(axis=0))
    print('RMSE: %5.2f kcal/mol' % sp.square(y_pred-y_test).mean(axis=0)**.5)
    print "TIMER predict", time.clock() - ttt

plt.plot(alpha_std, mae_error, 'o')
plt.xlabel("regression coefficients STD")
plt.ylabel("mean absolute error")
plt.savefig('alphastd_vs_maeerror.png')
# alpha = sp.array(alpha).T
# for i, a in enumerate(alpha):
#     plt.subplot(521+i)
#     plt.semilogx(nteach, a)
# plt.subplots_adjust(hspace=1)
# plt.subplots_adjust(wspace=1)
# plt.savefig('myplot2.png')
# # sp.save("alphas.npy", sp.array(alpha))
