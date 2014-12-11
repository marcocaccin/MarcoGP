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

# --------------------------------------------
# WHAT IT DOES:
# Loads a database. Reads all properties. Plots histograms of properties distributions.
# --------------------------------------------

# --------------------------------------------
# Parameters for the run
# --------------------------------------------
target_properties = 'all' # ['alpha'] 
database_file = 'dsC7O2H10nsd_db.pkl'

# --------------------------------------------
# Load all database
# --------------------------------------------
ttt = time.clock()
dataset = pickle.load(open(database_file, 'r'))
print "TIMER load_data", time.clock() - ttt


if target_properties == 'all':
    target_properties = [k for k in dataset.keys() if k not in ['description', 'idx', 'X']]
else:
    target_properties = [k for k in dataset.keys() if k in target_properties]


# histograms of properties
histograms = [] # sp.empty((len(target_properties), 2))
for i, prop in enumerate(target_properties):
    y = (dataset[prop] - sp.asarray(dataset[prop]).mean()) / sp.asarray(dataset[prop]).std()
    frequency, bins, patches = plt.hist(y, bins=50, normed=True)
    bins_dummy = list(bins)
    bins_dummy.append(bins_dummy.pop(0))
    bins = ((bins + sp.asarray(bins_dummy)) / 2)[:-1]
    histograms.append(sp.row_stack((bins, frequency)))

plt.clf()
for i, h in enumerate(histograms):
    plt.plot(h[0], h[1],'--', label=target_properties[i])
plt.xlabel("normalised property")
plt.ylabel("frequency")
legend = plt.legend()
plt.show()

raw_input("Press a key to continue")
plt.clf()

# Output space pair distance histograms
histograms = [] 
for i, prop in enumerate(target_properties):
    y = (dataset[prop] - sp.asarray(dataset[prop]).mean()) / sp.asarray(dataset[prop]).std()
    y = sp.spatial.distance.pdist(y[:,None])
    frequency, bins, patches = plt.hist(y, bins=53, normed=True)
    bins_dummy = list(bins)
    bins_dummy.append(bins_dummy.pop(0))
    bins = ((bins + sp.asarray(bins_dummy)) / 2)[:-1]
    histograms.append(sp.row_stack((bins, frequency)))

# --------------------------------------------
# Setup a Gaussian Process
# --------------------------------------------
theta0 = 1.0e1
nugget = 1.0e-15
normalise = 1
metric = 'cityblock'

gp = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0]),
                     nugget=nugget, verbose=True, normalise=normalise, do_features_projection=False, low_memory=False, metric=metric)

eigX = [(eigh(M, eigvals_only=True))[::-1] for M in dataset['X']]
gp.calc_kernel_matrix(eigX)
d = sp.spatial.distance.squareform(gp.D)
frequency, bins, patches = plt.hist(y, bins=47, normed=True)
bins_dummy = list(bins)
bins_dummy.append(bins_dummy.pop(0))
bins = ((bins + sp.asarray(bins_dummy)) / 2)[:-1]
plt.clf()

for i, h in enumerate(histograms):
    plt.plot(h[0], h[1],'-', label=target_properties[i])
plt.plot(bins, frequency, 'o', label='eigenvalues of Coulomb matrix')
plt.xlabel("distance in normalised property space")
plt.ylabel("frequency")
plt.legend()
plt.show()
