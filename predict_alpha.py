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
import matplotlib.cm as cm

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


def teach_database_plusone(GP, X, y, X_t, y_t):
    # From a fixed database (X,y), get alpha of some new configurations if added one at a time
    if type(y_t) is float:
        X_t = [X_t]
        y_t = [y_t]
    alphas = []
    for i, (X_test, y_test) in enumerate(zip(X_t, y_t)):
        X_plus = X_test + X # add element to the lists
        y_plus = y_test + y
        ttt = time.clock()

        GP.fit(X_plus, y_plus)
        print "TIMER teach", time.clock() - ttt
        alphas.append((gp.alpha[0]).flatten())
        GP.flush_data()
    return sp.array(alphas).flatten()


# --------------------------------------------
# WHAT IT DOES:
# Latest idea by Anatole
# Ntest test configurations, Ndatabases databases. Teach db+1 and check if inverse*inverse*k works
# --------------------------------------------

# --------------------------------------------
# Parameters for the run
# --------------------------------------------
split = 1
N_models = 1
theta0 = 1.0e+2
nugget = 1.0e-3
normalise = -1

Ntest = 10
Nteach = 2000
Ndatabases = 1

# --------------------------------------------
# Load all database
# --------------------------------------------
ttt = time.clock()
if not os.path.exists('qm7.pkl'): os.system('wget http://www.quantum-machine.org/data/qm7.pkl')
dataset = pickle.load(open('qm7.pkl','r'))

# --------------------------------------------
# Extract training data and test set
# --------------------------------------------
print "TIMER load_data", time.clock() - ttt


test_indices_rec = []
teach_indices_rec = []

alpha_predicted = []
alpha_target = []
energy_target = []
energy_error = []

# Setup a Gaussian Process once and for all so that parameters do not change
gp = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0]),
                     nugget=nugget, verbose=True, normalise=normalise, do_features_projection=False, low_memory=False)

# --------------------------------------------
# Loop over different training sets of the same size
# --------------------------------------------    
for iteration in range(Ndatabases):

    # Pick Ntest configurations randomly
    test_indices = list(sp.random.randint(0, high=dataset['T'].size, size=Ntest))
    db_indices = randint_norepeat(0, exclude=test_indices, high=dataset['T'].size, size=Nteach)
    teach_indices_rec.append(db_indices)
    
    print "\n", "-"*60, "\n"
    print "db size = %d, iteration %03d" % (Nteach, iteration)
    # Select training data
    X = dataset['X'][test_indices + db_indices]
    T = dataset['T'][test_indices + db_indices]
    
    # --------------------------------------------
    # Extract feature(s) from training data and test set
    # --------------------------------------------
    # in this case, only sorted eigenvalues of Coulomb matrix
    ttt = time.clock()
    eigX = [(eigh(M, eigvals_only=True))[::-1] for M in X]
    print "TIMER eval_features", time.clock() - ttt
    eigX_t  = eigX[:Ntest]
    eigX_db = eigX[Ntest:]
    y = T.ravel()
    y_t  = y[:Ntest]
    y_db = y[Ntest:]

    # do len(y_t) teachings by including db + 1 configurations
    alphas = teach_database_plusone(gp, eigX_db, y_db, eigX_t, y_t)
    alpha_target.append(alphas)


    # --------------------------------------------
    # Second time don't include the test set and predict
    # --------------------------------------------
    ttt = time.clock()
    gp.flush_data()
    # Fit to data
    gp.fit(eigX_db, y_db)
    print "TIMER teach", time.clock() - ttt

    beta = sp.dot(gp.inverse, gp.alpha)
    y_pred, k = gp.predict(eigX_t, return_k=True)
    # predict the alphas the K-1 * K-1 * k way
    alpha_predicted.append(sp.dot(k, beta.flatten()))
    energy_target.append(y_t)
    energy_error.append(y_pred - y_t)

    # check whether the ML itself is doing sensible things
    print "ERROR = ", energy_error[-1]
    print "ALPHA TRUE vs. PREDICTED:", alphas, alpha_predicted[-1]

with open('alpha_predictions.txt', 'a') as f:
    f.write("n_test_molecules=%d n_databases=%d db_size=%d\n" % (Ntest, Ndatabases, Nteach))
    output_data = sp.vstack((sp.array(alpha_target).flatten(), sp.array(alpha_predicted).flatten(), sp.array(energy_target).flatten(), sp.array(energy_error).flatten()))
    sp.savetxt(f, output_data.T)

f.close()

