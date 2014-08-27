#!/usr/bin/env python

import os, pickle, random, time
try:
    os.remove('my_gp_module.pyc')
except OSError:
    pass
import scipy as sp
from scipy.linalg import norm, eigh, inv, pinv, pinv2, pinvh
from my_gp_module import GaussianProcess

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

# P = dataset['P'][range(0,split)+range(split+1,5)].flatten()
P = dataset['P'][split + 1]
X = dataset['X'][P]
T = dataset['T'][P]
Ptest  = dataset['P'][split]
Xtest = dataset['X'][Ptest]
Ttest = dataset['T'][Ptest]
print "TIMER load_data", time.clock() - ttt

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

alpha = []
covmat = []
for theta0 in [10.0**i for i in sp.linspace(-2,5,7)]: # sp.linspace(1,1, N_models):
    # Setup a Gaussian Process model
    ttt = time.clock()
    gp = GaussianProcess(corr='absolute_exponential', theta0=sp.asarray([theta0]),
                         nugget=1e-3, verbose=True, low_memory=False)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(eigX, y)
    print "TIMER teach", time.clock() - ttt

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
    print('RMSE: %5.2f kcal/mol' %sp.square(y_pred-Ttest.ravel()).mean(axis=0)**.5)
    print "TIMER predict", time.clock() - ttt
    alpha.append(gp.alpha)
    covmat.append(gp.K)

    print r"\alpha STD: %f" % (sp.std(gp.alpha) / sp.mean(sp.absolute(gp.alpha)))

    print "\n", "-"*60, "\n"

# P = dataset['P'][range(0,split)+range(split+1,5)].flatten()
# X = dataset['X'][P]
# T = dataset['T'][P]
# print "TIMER load_data", time.clock() - ttt

# # --------------------------------------------
# # Extract feature(s) from training data
# # --------------------------------------------
# # in this case, only sorted eigenvalues of Coulomb matrix
# ttt = time.clock()
# eigX = [(eigh(M, eigvals_only=True))[::-1] for M in X]
# print "TIMER eval_features", time.clock() -ttt
# # --------------------------------------------
# # Generate distance matrix (currently it's only one for all)
# # --------------------------------------------
# ttt = time.clock()
# D = sp.zeros((len(eigX), len(eigX)))
# for i, v1 in enumerate(eigX):
#     for j, v2 in enumerate(eigX[i:]):
#         if i == j: D[i,j] = 0.0
#     else:
#         D[i,j] = norm(v1 - v2)
#         D[j,i] = D[i,j]
# print "TIMER distance_matrix", time.clock() -ttt
# # --------------------------------------------
# # Generate multiple kernel matrices
# # --------------------------------------------
# list_k_I = []
# list_alphas_I = []
# for l in sp.linspace(0.01, 1,1):
#     for reg_param in sp.linspace(0.01, 0.1,1):
#         print "STATUS:", l, reg_param
#         k_I = sp.exp(-D/l)
#         list_k_I.append(k_I)
#         # --------------------------------------------
#         # Calculate regression coefficients
#         # --------------------------------------------
#         ttt = time.clock()
#         inv_mat = inv(k_I + reg_param * sp.eye(len(eigX)))
#         print "TIMER matrix_inversion", time.clock() -ttt
#         alphas = sp.dot(inv_mat, T)
#         list_alphas_I.append(alphas)
#         sp.save("alphas_%.3f_%.3f.npy" % (l, reg_param), alphas)
#         # --------------------------------------------
#         # Predict something already known
#         # --------------------------------------------
#         print "TEST length scale: %.3f, regularisation parameter: %.3f" % (l, reg_param)
#         for counter in range(5):
#             idx = random.randint(0, len(eigX) - 1)
#             eigt = eigX[idx]
#             kt = [kernel(v1, eigt, l) for v1 in eigX]
#             print "PREDICTION: %.3f, TARGET: %.3f" % (sp.dot(kt, alphas), T[idx])
