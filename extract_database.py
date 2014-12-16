#!/usr/bin/env python


comment = """
Properties in 2nd line of xyz file (-1 for python indexing):

1	tag	-	gdb9 string to facilitate extraction
2	idx	-	Consecutive, 1-based integer identifier
3	A	GHz	Rotational constant
4	B	GHz	Rotational constant
5	C	GHz	Rotational constant
6	\mu	D	Dipole moment
7	\alpha	a_0^3	Isotropic polarizability
8	\epsilon_{HOMO}	Ha	Energy of HOMO
9	\epsilon_{LUMO}	Ha	Energy of LUMO
10	\epsilon_{gap}	Ha	Gap (\epsilon_{LUMO} - \epsilon_{HOMO})
11	R^2	a_0^2	Electronic spatial extent
12	zpve	Ha	Zero point vibrational energy
13	U_0	Ha	Internal energy at 0 K
14	U	Ha	Internal energy at 298.15 K
15	H	Ha	Enthalpy at 298.15 K
16	G	Ha	Free energy at 298.15 K
17	C_v	\frac{cal}{molK}	Heat capacity at 298.15 K
"""

import glob, pickle, sys, os
import scipy as sp
import scipy.spatial.distance as dist


def species_to_nuc_charge(species):
    
    conversion = {'H':1, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'P':15, 'S':16, 'Cl':17}
    charge = []
    for s in species:
        charge.append(conversion[s])
    return sp.array(charge)



basename = sys.argv[1]
ext = "xyz"
cwd = os.getcwd()

keywords = ['idx', 'A', 'B', 'C', 'mu', 'alpha',
		   'e_homo', 'e_lumo', 'e_gap', 'rsquare',
		   'zpve', 'U_0', 'U', 'H', 'G', 'C_v']
# make a Dictionary with empty lists as words
dataset = {}
for k in keywords:
    dataset[k] = []
ordering = []
# find in which order the keywords entered in the Dictionary
for k in dataset.keys():
    ordering.append(sp.where(k == sp.asarray(keywords))[0][0])

    
X_all = []
charge = []

files = glob.glob(os.path.join(cwd, "%s*.%s" % (basename, ext)))
for file in files:
    with open(file, 'r') as f:
        natoms = int(f.readline())
        # extract all properties in a list of floats
        properties = [float(p) for i,p in enumerate(f.readline().split()) if i > 0]

        # construct molecule from species x y z charge
        molecule = []
        try:
            for j in range(natoms):
                # molecule.append([float(x) for i, x in enumerate(f.readline().split()) if i > 0])
                molecule.append([x for i, x in enumerate(f.readline().split())])
            molecule = sp.array(molecule)
            nuc_charge = species_to_nuc_charge(molecule[:,0])
            molecule = sp.array(molecule[:,1:], dtype='float64')
            pos = molecule[:,:-1]
            # construct Coulomb matrix
            X = sp.outer(nuc_charge, nuc_charge) / (dist.squareform(dist.pdist(pos)) + sp.eye(natoms))
            sp.fill_diagonal(X, 0.5*sp.absolute(nuc_charge)**2.4)
            # add all properties of current molecule to dataset
            [dataset[k].append(properties[ordering[i]]) for i, k in enumerate(dataset.keys())]
            dataset['idx'][-1] = int(dataset['idx'][-1]) # index is an integer

            X_all.append(X)
            charge.append(molecule[:,-1])
        except Exception as err:
            print "Molecule %s skipped, malformed file" % file
            print err
            pass

        
dataset['X'] = X_all
dataset['charge'] = charge
for k,w in dataset.iteritems():
    dataset[k] = sp.array(w)

dataset['description'] = comment # add (sort of) header to the dataset
f = open(os.path.join(cwd, '%s_db.pkl' % basename), 'w')
pickle.dump(dataset, f)
f.close()

