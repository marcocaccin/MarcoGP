#!/usr/bin/env python

import os, pickle, time
try:
    os.remove('my_gp_module.pyc')
except OSError:
    pass
import scipy as sp
from scipy.linalg import eigh
from my_gp_module import GaussianProcess
import matplotlib.pyplot as plt


def randint_norepeat(low, exclude=None, high=None, size=None):
    l = list(sp.random.randint(low, high=high, size=size))
    if exclude is None:
        return l
    else:
        # remove elements already present in exclude
        l = [x for x in l if x not in exclude]
        for i in range(size-len(l)):
            while True:
                new = sp.random.randint(low, high=high)
                if new not in exclude and new not in l:
                    l.append(new)
                    break
        return l

# --------------------------------------------
# WHAT IT DOES:
# Pick at random Ntest configurations to keep track of,
# Select a random set of Nteach configurations:
#    - Teach Nteach+Ntest and save test regression coefficients alpha
#    - Do a second teaching not including the test configurations
#    - Predict energy of test configurations using 2nd teaching and evaluate error
# Plot alpha vs. error
# NB: the tests are done 10 at a time simply to have clear graphs
#     where 1 colour = 1 configuration
# --------------------------------------------

# --------------------------------------------
# Parameters for the run
# --------------------------------------------
split = 1
N_models = 1
theta0 = 10.0
Ntest = 10
Nteach = 1000
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


for outiter in range(10):
    # Pick Ntest configurations randomly
    test_indices = list(sp.random.randint(0, high=allP.size, size=Ntest))
    # --------------------------------------------
    # Loop over different training sets of the same size
    # --------------------------------------------
    alpha = []
    alpha_std = []
    mae_error = []
    errors = []
    for iteration in range(20):
        db_indices = randint_norepeat(0, exclude=test_indices, high=allP.size, size=Nteach)
        # --------------------------------------------
        # First time include the test set to calculate their alpha
        # --------------------------------------------    
        print "\n", "-"*60, "\n"
        print "N teach = %d, iteration %03d" % (Nteach, iteration)
        # Select training data
        P = allP[test_indices + db_indices]
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
        alpha.append(local_alpha.flatten())
        alpha_std.append(sp.std(local_alpha))

        # --------------------------------------------
        # Second time don't include the test set and predict
        # --------------------------------------------    
        # Extract feature(s) from training data and test set
        # --------------------------------------------
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
        errors.append(sp.absolute(y_pred-y_test))
        print('\n test set:')
        print('MAE:  %5.2f kcal/mol' % sp.absolute(y_pred-y_test).mean(axis=0))
        print('RMSE: %5.2f kcal/mol' % sp.square(y_pred-y_test).mean(axis=0)**.5)
        print "TIMER predict", time.clock() - ttt

    alpha = sp.array(alpha).T
    errors = sp.array(errors).T
    plt.clf()

    # Plot alpha STD vs. MAE error scatter (1 plot, dots, ~ 1 line)
    # plt.plot(alpha_std, mae_error, 'o')
    # plt.xlabel("regression coefficients STD")
    # plt.ylabel("mean absolute error")
    # plt.savefig('alphastd_vs_maeerror.png')


    # Plot alpha vs. error scatter for selected test confs (1 plot, nplots <= Ntest lines)
    # nplots = 8
    # for a, err in zip(alpha[:nplots], errors[:nplots]):
    #     plt.plot(a, err, 'o')
    # plt.xlabel("regression coefficient")
    # plt.ylabel("absolute error [kcal/mol]")
    # plt.savefig('alpha_vs_error.png')
    # plt.clf()

    for a, err in zip(alpha, errors):
        plt.plot(a, err, 'o')
    plt.xlabel("regression coefficient")
    plt.ylabel("absolute error [kcal/mol]")
    plt.savefig('alpha_vs_error_N1000-%03d.png' % outiter)
    plt.clf()

# sp.save("alphas.npy", sp.array(alpha))
