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

# --------------------------------------------
# Load all database
# --------------------------------------------
ttt = time.clock()
if not os.path.exists('qm7.pkl'): os.system('wget http://www.quantum-machine.org/data/qm7.pkl')
dataset = pickle.load(open('qm7.pkl','r'))

# --------------------------------------------
# Extract training data and test set
# --------------------------------------------
split = 1
N_models = 1
theta0 = 10.0
Nfixed = 100

allP = dataset['P'][range(0,split)+range(split+1,5)].flatten()
nteachs = sp.int32(sp.exp(sp.linspace(sp.log(Nfixed+0.0), sp.log(allP.size), 30)))
Ptest  = dataset['P'][split]
Xtest = dataset['X'][Ptest]
Ttest = dataset['T'][Ptest]
print "TIMER load_data", time.clock() - ttt


alpha = []
covmat = []
alpha_std = []
for Nteach in nteachs:

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
    eigt = [(eigh(M, eigvals_only=True))[::-1] for M in Xtest]
    print "TIMER eval_features", time.clock() - ttt

    # Observations
    y = T.ravel()

    # Setup a Gaussian Process model
    ttt = time.clock()
    gp = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0]),
                         nugget=1e-3, verbose=True, normalise=True, do_features_projection=False, low_memory=False)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(eigX, y)
    print "TIMER teach", time.clock() - ttt
    # alpha.append(gp.alpha.flatten()[:Nfixed].copy())
    # covmat.append(gp.K.copy())
    # print "projections STD"
    # print [sp.std(proj) for proj in gp.feat_proj]
    print "alpha STD: %f" % sp.std(gp.alpha[:Nfixed])
    print "alpha MAV: %f" % sp.mean(sp.absolute(gp.alpha[:Nfixed]))
    alpha_std.append(sp.std(gp.alpha.flatten()[:Nfixed]))
    alpha.append(gp.alpha.flatten()[:Nfixed])
    if do_predict:
        ttt = time.clock()
        # # Make the prediction on training set
        # y_pred, MSE = gp.predict(eigX, eval_MSE=True)
        # sigma = sp.sqrt(MSE)
        # print('\n training set:')
        # print('MAE:  %5.2f kcal/mol' % sp.absolute(y_pred-y).mean(axis=0))
        # print('RMSE: %5.2f kcal/mol' % sp.square(y_pred-y).mean(axis=0)**.5)
        # Make the prediction on test set
        y_pred, MSE = gp.predict(eigt, eval_MSE=True)
        sigma = sp.sqrt(MSE)
        print('\n test set:')
        print('MAE:  %5.2f kcal/mol' % sp.absolute(y_pred-Ttest.ravel()).mean(axis=0))
        print('RMSE: %5.2f kcal/mol' % sp.square(y_pred-Ttest.ravel()).mean(axis=0)**.5)
        print "TIMER predict", time.clock() - ttt

# alpha = sp.array(alpha).T
# plt.subplot(111)

for i, a in enumerate(alpha):
    plt.semilogx(nteachs, a)
plt.xlabel("Learning set size")
plt.ylabel("regression coefficients")
# plt.subplots_adjust(hspace=1)
# plt.subplots_adjust(wspace=1)

# plt.semilogx(nteachs, alpha_std, 'o')
# plt.xlabel("Learning set size")
# plt.ylabel("regression coefficients STD")
plt.savefig('alphastd_vs_Nteach_spaghetti.png')
# sp.save("alphas.npy", sp.array(alpha))
