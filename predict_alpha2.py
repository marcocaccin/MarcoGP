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
    if exclude is not None:
        # remove elements already present in exclude
        l = [x for x in l if x not in exclude]
        for i in range(size-len(l)):
            while True:
                new = sp.random.randint(low, high=high)
                if new not in exclude and new not in l:
                    l.append(new)
                    break
    l.sort()
    return l


def teach_database_plusone(GP, X, y, X_t, y_t):
    GP.flush_data()
    # Force all data to be numpy arrays
    X, y = sp.asarray(X), sp.asarray(y)
    X_t, y_t = sp.asarray(X_t), sp.asarray(y_t)
    # From a fixed database (X,y), get alpha of some new configurations if added one at a time
    alphas = []
    for i, (X_test, y_test) in enumerate(zip(X_t, y_t)):
        if y_test.size != 1:
            print "ERROR: output space must be 1D. Exiting..."
            return
        # Test configuration is placed at position 0
        X_plus = sp.row_stack((X_test, X))
        y_plus = sp.append(y_test, y)
        ttt = time.clock()
        GP.fit(X_plus, y_plus)
        print "TIMER teach", time.clock() - ttt
        alphas.append((GP.alpha[0]).flatten().copy())
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
theta0 = 2.0e1
theta0_level2 = 1.0e1
nugget = 1.0e-15
normalise = 1
metric = 'cityblock'
Ntest = 50
Nteach = 1200
Ndatabases = 1
target_property = 'U_0'
dataset_loc = 'dsC7O2H10nsd_db.pkl'
# --------------------------------------------
# Load all database
# --------------------------------------------
ttt = time.clock()
dataset = pickle.load(open(dataset_loc, 'r'))

print "TIMER load_data", time.clock() - ttt

test_indices_rec, teach_indices_rec = [], []
alpha_predicted, alpha_target = [], []
energy_target, energy_error = [], []
 
# --------------------------------------------
# Setup a Gaussian Process 
# --------------------------------------------
gp = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0]),
                     nugget=nugget, verbose=True, normalise=normalise, do_features_projection=False, low_memory=False, metric=metric)
gp_level2 = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0_level2]),
                     nugget=nugget, verbose=True, normalise=normalise, do_features_projection=False, low_memory=False, metric=metric)


# --------------------------------------------
# Loop over different training sets of the same size
# --------------------------------------------    
for iteration in range(Ndatabases):
    # --------------------------------------------
    # Pick Ntest configurations randomly
    # --------------------------------------------
    test_indices = list(sp.random.randint(0, high=dataset[target_property].size, size=Ntest))
    db_indices = randint_norepeat(0, exclude=test_indices, high=dataset[target_property].size, size=Nteach)
    sp.save('db_indices_%d-%s' % (iteration, time.ctime()), db_indices)
    teach_indices_rec.append(db_indices)
    
    X = dataset['X'][test_indices + db_indices]
    T = dataset[target_property][test_indices + db_indices]
    print "\n", "-"*60, "\n"
    print "db size = %d, iteration %03d" % (Nteach, iteration)
   # --------------------------------------------
    # Extract feature(s) from training data and test set:
    # only sorted eigenvalues of Coulomb matrix in this case
    # --------------------------------------------
    ttt = time.clock()
    eigX = [(eigh(M, eigvals_only=True))[::-1] for M in X]
    print "TIMER eval_features", time.clock() - ttt
    eigX_t  = eigX[:Ntest]
    eigX_db = eigX[Ntest:]
    y = T.ravel()
    y_t  = y[:Ntest]
    y_db = y[Ntest:]

    # --------------------------------------------
    # Do len(y_t) teachings by including db + 1 configurations
    # --------------------------------------------
    alphas = teach_database_plusone(gp, eigX_db, y_db, eigX_t, y_t)
    alpha_target.append(alphas)
    gp.flush_data()

    # --------------------------------------------

    # --------------------------------------------
    # Second time don't include the test set and predict
    # --------------------------------------------
    ttt = time.clock()
    # Fit to data
    gp.fit(eigX_db, y_db)
    gp_level2.fit(eigX_db, gp.alpha.flatten())
    
    print "TIMER teach", time.clock() - ttt

    y_pred = gp.predict(eigX_t)
    
    # --------------------------------------------
    # predict the alphas 
    # --------------------------------------------
    alpha_pred = gp_level2.predict(eigX_t)

    alpha_predicted.append(alpha_pred.flatten())

    energy_target.append(y_t)
    energy_error.append(y_pred - y_t)

    # check whether the level 1 ML itself is predicting the property correctly
    print "ERROR = ", energy_error[-1]
    print "ALPHA TRUE vs. PREDICTED:", alphas, alpha_predicted[-1]

    # Save round of predictions
    with open('alpha_predictions.txt', 'a') as f:
        if iteration == 0: f.write("n_test_molecules=%d n_databases=%d db_size=%d\n" % (Ntest, Ndatabases, Nteach))
        output_data = sp.vstack((sp.array(alpha_target).flatten(), sp.array(alpha_predicted).flatten(), sp.array(energy_target).flatten(), sp.array(energy_error).flatten()))
        sp.savetxt(f, output_data.T)

f.close()

for at, ap in zip(alpha_target, alpha_predicted):
    plt.plot(at,ap,'x')
plt.xlabel("actual regression coefficient")
plt.ylabel("predicted regression coefficient")
plt.savefig('regr_actual_vs_pred-%s.pdf' % time.ctime())
plt.clf()

for at, err in zip(alpha_target, energy_error):
    plt.plot(at,err,'x')
plt.xlabel("actual regression coefficient")
plt.ylabel("error on property %s" % target_property)
plt.title("MAE = %.3f" % sp.absolute(sp.array(energy_error)).mean())
plt.savefig('regr_actual_vs_error-%s.pdf' % time.ctime())
plt.clf()
