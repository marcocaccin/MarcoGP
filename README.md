# MarcoGP
Playground for wacky implementations of GP regression, used in various contexts of modelling of forces/energies in atomistic systems.

## forcegp_module
GP regression model for vector-valued functions, à la Alvarez et al. arXiv:1106.6251. 

Why: modelling force acting on an atom given its neighbouring atomic environment

How: Force (output) is a 3D vector that covariates with the rotation of the atomic positions (input). For direct embedding of the system symmetries, the kernel matrix should become a block matrix of shape 3N x 3N (where N is the number of samples). Each 3x3 block should be the product of a scalar and symmetry invariant kernel and a 3x3 matrix composed of the sum of outer products of some internal vectors.

In this preliminary study, the internal vectors correspond to the definition originally proposed in http://dx.doi.org/10.1103/PhysRevLett.114.096405. The scalar-valued kernel, instead, is calculated from the descriptor initially introduced in arxiv.org/abs/1109.2618E (sorted eigenvalues of the local Coulomb matrix).

The module also allows for simply feature extraction from atomistic system (given in the silicon_db.xyz)

Plenty of snazzy maths, like a 4D tensor kernel that is then flattened into 2 dimensions.

## gogogo
Simple script to extract features from atomic configurations database.

### predict_alpha, alpha_trends, extract_database

Defunct project that was trying to model the error on predictions of a GPR model by building a 2nd layer GPR model able to *learn* the dual coefficients of the 1st layer. The idea has proven to be completely unpractical or unsound (non-exclusive or). It reproduces the results obtained in arxiv.org/abs/1109.2618E
