from forcegp_module import GaussianProcess
from quippy import AtomsList
import random
import pickle
from forcegp_module import GaussianProcess, iv_default_params
import numpy as np


tmp = iv_default_params()
exps = tmp[:,0]
r_cuts = tmp[:,1]
database_whole = AtomsList('silicon_db.xyz')
teach_idx = random.sample(range(len(database_whole)), 40)
teaching_data = database_whole[teach_idx]
gp = GaussianProcess(theta0 = 2.0, nugget = 0.01, iv_params=[exps, r_cuts])
ivs_db, eigs_db, y_db, iv_corr_db, iv_means, eig_means, eig_stds = gp.atomsdb_get_features(teaching_data, return_features=True)

results = {}
results['ivs_db'] = ivs_db
results['eigs_db'] = eigs_db
results['y_db'] = y_db
results['iv_corr_db'] = iv_corr_db

pickle.dump(results, open('Si-50db.pkl', 'w'))


1/0 
gp.ivs = ivs_db
gp.eigs = eigs_db
gp.iv_corr = iv_corr_db
gp.y = y_db
gp.iv_means, gp.eig_means, gp.eig_stds = iv_means, eig_means, eig_stds


pf = []
for corr in [0.1]: #in np.linspace(0.1,10,2, endpoint=True):
    gp.theta0 = corr
    gp.fit_sollich()

    pred_idx = random.sample(range(len(database_whole)), 2)
    pred_data = database_whole[pred_idx]
    predicted_forces = gp.predict_sollich(atomslist=pred_data)
    pf.append(predicted_forces)
    # actual_forces = 
    #print predicted_forces
