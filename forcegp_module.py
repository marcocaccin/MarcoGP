"""
Gaussian process regression module. Inspired by sklearn.gaussian_process module and stripped out naked to the bare minimum
"""
# Author: Marco Caccin <marco DOT caccin AT gmail DOT com>
#
# License: Apache License

from __future__ import print_function, division

from scipy import linalg as LA
import scipy as sp
import numpy as np
import scipy.spatial.distance as spdist
from quippy import *
from matscipy.neighbours import neighbour_list

MACHINE_EPSILON = sp.finfo(sp.double).eps



def rotmat_from_u2v(u,v):
    """
    Return the rotation matrix associated with rotation of vector u onto vector v. Euler-Rodrigues formula.
    """
    u, v = u / LA.norm(u), v / LA.norm(v)
    axis = sp.cross(u,v)
    theta = sp.arcsin(LA.norm(axis))
    axis = axis/LA.norm(axis) # math.sqrt(np.dot(axis, axis))
    a = sp.cos(theta/2)
    b, c, d = -axis * sp.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotmat_multi(us, vs):
    """
    Return the matrix of rotation matrices associated with rotation of vector u onto vector v
    for all u in us and v in vs
    """
    us, vs = sp.asarray(us), sp.asarray(vs)
    nu, nv = us.shape[0], vs.shape[0]
    mat = sp.zeros((nu, nv, 3, 3))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            if j > i:
                R = rotmat_from_u2v(u,v)
                mat[i,j] = R
                mat[i,j] = R.T
            elif j == i:
                mat[i,j] = sp.eye(3)
            else:
                continue
    return mat


def internal_vector(atoms, at_idx, exponent, r_cut, do_calc_connect=True):
    iv = sp.zeros(3)
    atom, r = neighbour_list("iD", atoms, 8.0)
    r = np.asarray(r)[atom == at_idx]
    # if do_calc_connect:
    #     atoms.set_cutoff(8.0)
    #     atoms.calc_connect()
    # # each copy of each atom within the cutoff radius contribute to iv
    # r = np.asarray([neighbours.diff for neighbours in atoms.neighbours[at_idx]])
    r_mag = np.asarray(map(LA.norm, r))
    return (r / r_mag[:,None] * sp.exp(- (r_mag / r_cut) ** exponent)[:,None]).sum(axis=0)


def internal_vectors(atoms, at_idx, exps, r_cuts, do_calc_connect=True):
    return [internal_vector(atoms, at_idx, exponent, r_cut, do_calc_connect=do_calc_connect) for exponent, rcut in zip(exps, r_cuts)]


def outer_product_multi(u,v):
    """
    outer product for 2 arrays of vectors
    """
    return u[:,None, :,None] * v[None,:,None,:]


def my_tensor_reshape(A):
    """
    reshaped = sp.zeros((Ashape[0]*Ashape[2], Ashape[1]*Ashape[3]))
    for i, row in enumerate(A):
        for j, element in enumerate(row):
            reshaped[i:i+3,j:j+3] = element
    return reshaped
    """
    Ashape = A.shape
    return A.swapaxes(1,2).reshape((A.shape[0]*A.shape[2],A.shape[1]*A.shape[3]))


def iv_default_params():
    """
    cutoff radius, exponent, sigma of internal vectors, as of PRL.
    """

    v = sp.array([
        [ 0.5000000000000000 , 1.0000000000000000,  0.5336331762080207 ],
        [ 1.4375000000000000 , 1.8750000000000000,  4.5908332134557153 ],
        [ 1.4375000000000000 , 2.7500000000000000,  2.8859968632833466 ],
        [ 2.3750000000000000 , 2.7500000000000000,  14.197236995498168 ],
        [ 1.4375000000000000 , 3.6250000000000000,  1.2065280950214012 ],
        [ 2.3750000000000000 , 3.6250000000000000,  16.572655752944318 ],
        [ 3.3125000000000000 , 3.6250000000000000,  26.842580061743142 ],
        [ 1.4375000000000000 , 4.5000000000000000,  0.3008147421149200 ],
        [ 2.3750000000000000 , 4.5000000000000000,  19.385902296845487 ],
        [ 3.3125000000000000 , 4.5000000000000000,  27.561787309821216 ],
        [ 4.2500000000000000 , 4.5000000000000000,  41.544248494327334 ],
        [ 2.3750000000000000 , 5.3750000000000000,  22.225299759904363 ],
        [ 3.3125000000000000 , 5.3750000000000000,  27.054271806690156 ],
        [ 4.2500000000000000 , 5.3750000000000000,  47.148445225323542 ],
        [ 2.3750000000000000 , 6.2500000000000000,  25.119522871501516 ],
        [ 3.3125000000000000 , 6.2500000000000000,  26.197634029913022 ],
        [ 4.2500000000000000 , 6.2500000000000000,  53.884205731369029 ],
        [ 5.1875000000000000 , 6.2500000000000000,  52.725598065166771 ],
        [ 2.3750000000000000 , 7.1250000000000000,  28.009249074189647 ],
        [ 3.3125000000000000 , 7.1250000000000000,  25.373905940522256 ],
        [ 4.2500000000000000 , 7.1250000000000000,  61.103256188283005 ],
        [ 5.1875000000000000 , 7.1250000000000000,  55.286697365645992 ]])
    return v


def coulomb_mat_eigvals(atoms, at_idx, r_cut, do_calc_connect=True, n_eigs=20):

    if do_calc_connect:
        atoms.set_cutoff(8.0)
        atoms.calc_connect()
    pos = sp.vstack((sp.asarray([sp.asarray(a.diff) for a in atoms.neighbours[at_idx]]), sp.zeros(3)))
    Z = sp.hstack((sp.asarray([atoms.z[a.j] for a in atoms.neighbours[at_idx]]), atoms.z[at_idx]))

    M = sp.outer(Z, Z) / (sp.spatial.distance_matrix(pos, pos) + np.eye(pos.shape[0]))
    sp.fill_diagonal(M, 0.5 * Z ** 2.4)

    # data = [[atoms.z[a.j], sp.asarray(a.diff)] for a in atoms.neighbours[at_idx]]
    # data.append([atoms.z[at_idx], sp.array([0,0,0])]) # central atom
    # M = sp.zeros((len(data), len(data)))
    # for i, atom1 in enumerate(data):
    #     M[i,i] = 0.5 * atom1[0] ** 2.4
    #     for j, atom2 in enumerate(data[i+1:]):
    #         j += i+1
    #         M[i,j] =  atom1[0] * atom2[0] / LA.norm(atom1[1] - atom2[1])
    # M = 0.5 * (M + M.T)
    eigs = (LA.eigh(M, eigvals_only=True))[::-1]
    if n_eigs == None:
        return eigs # all
    elif eigs.size >= n_eigs:
        return eigs[:n_eigs] # only first few eigenvectors
    else:
        return sp.hstack((eigs, sp.zeros(n_eigs - eigs.size))) # zero-padded extra fields


def scalar_kernel(d, theta, correlation='squared_exponential'):
    if correlation is 'absolute_exponential':
        return sp.exp(-d / theta) # correlation_models.absolute_exponential(theta, d)
    elif  correlation is 'squared_exponential':
        return sp.exp(-d**2 / (2.0 * theta**2)) # correlation_models.squared_exponential(theta, d)
    else:
        print("Correlation model %s not understood" % correlation)
        return None


class GaussianProcess:
    """
    corr : string or callable, optional
        A stationary autocorrelation function returning the autocorrelation
        between two points x and x'.
        Default assumes a squared-exponential autocorrelation model.
        Built-in correlation models are::
            'absolute_exponential', 'squared_exponential',
            NOT YET 'generalized_exponential', 'cubic', 'linear'

    verbose : boolean, optional
        A boolean specifying the verbose level.
        Default is verbose = False.

    theta0 : double array_like, optional
        An array with shape (n_features, ) or (1, ).
        The parameters in the autocorrelation model.
        If thetaL and thetaU are also specified, theta0 is considered as
        the starting point for the maximum likelihood estimation of the
        best set of parameters.
        Default assumes isotropic autocorrelation model with theta0 = 1e-1.

    normalise : boolean, optional
        Input X and observations y are centered and reduced wrt
        means and standard deviations estimated from the n_samples
        observations provided.
        Default is normalise = 1 so that both input and output data are normalised

    nugget : double or ndarray, optional
        Introduce a nugget effect to allow smooth predictions from noisy
        data.  If nugget is an ndarray, it must be the same length as the
        number of data points used for the fit.
        The nugget is added to the diagonal of the assumed training covariance;
        in this way it acts as a Tikhonov regularization in the problem.  In
        the special case of the squared exponential correlation function, the
        nugget mathematically represents the variance of the input values.
        Default assumes a nugget close to machine precision for the sake of
        robustness (nugget = 10. * MACHINE_EPSILON).
    """


    def __init__(self, corr='squared_exponential', verbose=False, theta0=1e-1,
                 normalise_scalar=True, normalise_ivs=True, nugget=10. * MACHINE_EPSILON,
                 low_memory=False, metric='euclidean', n_eigs=20,
                 iv_params=[iv_default_params()[:5,0], iv_default_params()[:5,1]]):

        self.corr = corr
        self.verbose = verbose
        self.theta0 = theta0
        self.normalise_scalar = normalise_scalar
        self.normalise_ivs = normalise_ivs
        self.nugget = nugget
        self.low_memory = low_memory
        self.metric = metric
        self.n_eigs = n_eigs
        self.iv_params = iv_params


    def flush_data(self):
        self.ivs = None
        self.eigs = None
        self.D = None
        self.K = None
        self.inverse = None
        self.y = None
        self.iv_corr = None


                # for at_idx in frange(atoms.n):
                #     print internal_vector(atoms, at_idx, exp, r_cut, do_calc_connect=False)
                #     print coulomb_mat_eigvals(atoms, at_idx, r_cut, do_calc_connect=False, n_eigs=self.n_eigs)

    def atomsdb_get_features(self, at_db, return_features=False):
        """
        type(at_db) == quippy.io.AtomsList
        """
        IVs = []
        EIGs = []
        Y = []
        exps, r_cuts = self.iv_params[0], self.iv_params[1]

        # each iv is an independent information channel
        for feature, (r_cut, exp) in enumerate(zip(r_cuts, exps)):
            print("Evaluating database feature %d of %d..." % (feature+1, exps.size))
            ivs = sp.zeros((sp.asarray(at_db.n).sum(), 3))
            eigs = sp.zeros((sp.asarray(at_db.n).sum(), self.n_eigs))
            i = 0
            for atoms in at_db:
                atoms.set_cutoff(8.0)
                atoms.calc_connect()
                for at_idx in frange(atoms.n):
                    if feature == 0: Y.append(sp.array(atoms.force[at_idx])) # get target forces
                    ivs[i] = internal_vector(atoms, at_idx, exp, r_cut, do_calc_connect=False)
                    eigs[i] = coulomb_mat_eigvals(atoms, at_idx, r_cut, do_calc_connect=False, n_eigs=self.n_eigs)
                    i+=1
            IVs.append(ivs)
            EIGs.append(eigs)
        # rescale eigenvalues descriptor
        if self.normalise_scalar:
            eig_means = sp.array([e[e.nonzero()[0], e.nonzero()[1]].mean() for e in EIGs])
            eig_stds = sp.array([e[e.nonzero()[0], e.nonzero()[1]].std() for e in EIGs])
            eig_stds[eig_stds == 0.] = 1.
            EIGs = [(e - mean) / std for e, mean, std in zip(EIGs, eig_means, eig_stds)]
        # rescale internal vector to have average length = 1
        if self.normalise_ivs:
            # iv_stds = [e[e.nonzero()[0], e.nonzero()[1]].std() for e in IVs]
            # iv_stds[iv_stds == 0.] = 1.
            iv_means = [sp.array([LA.norm(vector) for vector in e]).mean() for e in IVs]
            IVs = [iv / mean for iv, mean in zip(IVs, iv_means)]

        # output cleanup: add machine epsilon if force is exactly zero
        Y = sp.asarray(Y)
        Y[sp.array(map(LA.norm, Y)) <= MACHINE_EPSILON] = 10 * MACHINE_EPSILON * sp.ones(3)

        # correlations wrt actual forces
        IV_corr = sp.array([sp.diagonal(spdist.cdist(Y, iv, metric='correlation')).mean() for iv in IVs])

	if return_features:
	    return IVs, EIGs, Y, IV_corr, iv_means, eig_means, eig_stds
	else:
	    self.ivs = IVs
	    self.eigs = EIGs
	    self.y = Y
	    self.iv_corr = IV_corr
        self.iv_means = iv_means
        self.eig_means, self.eig_stds = eig_means, eig_stds


    def testatoms_get_features(self, atomslist, iv_means=None, eig_means=None, eig_stds=None):
        """
        type(atomslist) == quippy.io.AtomsList
        """
        IVs = []
        EIGs = []
        exps, r_cuts = self.iv_params[0], self.iv_params[1]

        if not iv_means:
            iv_means = self.iv_means
        if not (eig_means or eig_stds):
            eig_means, eig_stds = self.eig_means, self.eig_stds

        # each iv is an independent information channel
        for feature, (r_cut, exp) in enumerate(zip(r_cuts, exps)):
            print("Evaluating test set feature %d of %d..." % (feature+1, exps.size))
            ivs = sp.zeros((sp.asarray(atomslist.n).sum(), 3))
            eigs = sp.zeros((sp.asarray(atomslist.n).sum(), self.n_eigs))
            i = 0
            for atoms in atomslist:
                atoms.set_cutoff(8.0)
                atoms.calc_connect()
                for at_idx in frange(atoms.n):
                    ivs[i] = internal_vector(atoms, at_idx, exp, r_cut, do_calc_connect=False)
                    eigs[i] = coulomb_mat_eigvals(atoms, at_idx, r_cut, do_calc_connect=False, n_eigs=self.n_eigs)
                    i+=1
            IVs.append(ivs)
            EIGs.append(eigs)
        # rescale eigenvalues descriptor
        if self.normalise_scalar:
            EIGs = [(e - eig_means[i]) / eig_stds[i] for i,e in enumerate(EIGs)]
        # rescale internal vector to have average length = 1
        if self.normalise_ivs:
            IVs = [iv / mean for iv, mean in zip(IVs, iv_means)]
	    return IVs, EIGs


    def calc_scalar_kernel_matrices(self, X=None):
        """
        Perform only the calculation of the covariance matrix given the GP and a dataset

        Parameters
        ----------
        X : list
            A list of arrays, each with shape (n_samples, n_features) with the input at which
            observations were made. len(X) is the number of vectorial features.

        Returns
        -------
        gp : adds properties self.D and self.K
        """
        if not X:
            X = self.eigs

        self.D = []
        # Calculate distance matrix in vector form. The matrix form of X is obtained by scipy.spatial.distance.squareform(X).
        # One distance matrix per channel
        for x in X:
            D = spdist.pdist(x, metric = self.metric)
            self.D.append(spdist.squareform(D))

        # Covariance matrix K. One per channel
        # sklearn correlation doesn't work. Probably correlation_models needs some different inputs
        K = []
        for D in self.D:
            K.append(scalar_kernel(D, self.theta0, correlation=self.corr))
        if self.low_memory:
            self.D = None
        else:
            self.K = K
        self.X = X
        return K


    def fit(self, X=None, y=None):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : array_like, shape (n_samples, 3)
            An array with shape (n_eval, 3) with the observations of the output to be predicted.
            of shape (n_samples, 3) with the Best Linear Unbiased Prediction at x.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """

        if X:
            K_list = self.calc_scalar_kernel_matrices(X)
        else:
            K_list = self.calc_scalar_kernel_matrices()

        # add diagonal noise to each scalar kernel matrix
        K_list = [K + self.nugget * sp.ones(K.shape[0]) for K in K_list]

        Kglob = None
        # outer_iv = [sp.outer(iv, iv.T) for iv in self.ivs] # NO, wrong
        for K, ivs, iv_corr in zip(K_list, self.ivs, self.iv_corr):
            # make the outer product tensor of shape (N_ls, N_ls, 3, 3) and multiply it with the scalar kernel
            K3D = iv_corr * K[:, :, None, None] * rotmat_multi(ivs, ivs)
            # reshape tensor onto a 2D array tiled with 3x3 matrix blocks
            if Kglob is None:
                Kglob = K3D
            else:
                Kglob += K3D
        Kglob = my_tensor_reshape(Kglob)
        # # all channels merged into one covariance matrix
        # # K^{glob}_{ij} = \sum_{k = 1}^{N_{IVs}} w_k D_{k, ij} |v_k^i\rangle \langle v_k^j |

        try:
            inv = LA.pinv2(Kglob)
        except LA.LinAlgError as err:
            print("pinv2 failed: %s. Switching to pinvh" % err)
            try:
                inv = LA.pinvh(Kglob)
            except LA.LinAlgError as err:
                print("pinvh failed: %s. Switching to pinv2" % err)
                inv = None

        # alpha is the vector of regression coefficients of GaussianProcess
        alpha = sp.dot(inv, self.y.ravel())

        if not self.low_memory:
            self.inverse = inv
            self.Kglob = Kglob
        self.alpha = sp.array(alpha)


    def predict(self, atomslist=None, eigs_t=None, ivs_t=None):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters

        atomslist : list of Atoms or AtomsList
            An list giving the atomic configurations at
            which the prediction(s) should be made.

        Returns
        -------
        y : array_like, shape (n_samples, 3)
            An array with shape (n_eval, 3) for a Gaussian Process trained on an array
            of shape (n_samples, 3) with the Best Linear Unbiased Prediction at x.
        """

        if atomslist is not None:
            ivs_t, eigs_t = self.testatoms_get_features(atomslist)
        elif (eigs_t is None or ivs_t is None):
            return None

        # Check input shapes
        n_eval, _ = ivs_t[0].shape
        n_samples_y, _ = self.y.shape
        n_features = len(ivs_t)

        # Get scalar distances between each new point in X and all input training set
        if self.metric == 'euclidean':
            dx = [(((eig_db - eig_t[:,None])**2).sum(axis=2))**0.5 for eig_db, eig_t in zip(self.eigs, eigs_t)]
        elif self.metric == 'cityblock':
            dx = [(sp.absolute(self.X - X[:,None])).sum(axis=2) for eig_db, eig_t in zip(self.eigs, eigs_t)]
        else:
            print("ERROR: metric not understood")

        # Evaluate scalar correlation
        klist = [scalar_kernel(d, self.theta0) for d in dx]

        # join vectorial features and scalar correlation together
        kglob = None
        # outer_iv = [sp.outer(iv, iv.T) for iv in self.ivs] # NO, wrong
        for k, iv_t, iv_db, iv_corr in zip(klist, ivs_t, self.ivs, self.iv_corr):
            # make the outer product tensor of shape (N_ls, N_ls, 3, 3) and multiply it with the scalar kernel
            k3D = iv_corr * k[:, :, None, None] * rotmat_multi(iv_t, iv_db)
            if kglob is None:
                kglob = k3D
            else:
                kglob += k3D

        # reshape tensor onto a 2D array tiled with 3x3 matrix blocks
        k3D = my_tensor_reshape(kglob)

        # Predictor
        return sp.dot(k3D, self.alpha).reshape(n_eval,3)


    ####################################################################################################


    def fit_sollich(self, X=None, y=None):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : array_like, shape (n_samples, 3)
            An array with shape (n_eval, 3) with the observations of the output to be predicted.
            of shape (n_samples, 3) with the Best Linear Unbiased Prediction at x.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """

        if X:
            K_list = self.calc_scalar_kernel_matrices(X)
        else:
            K_list = self.calc_scalar_kernel_matrices()

        # add diagonal noise to each scalar kernel matrix
        K_list = [K + self.nugget * sp.ones(K.shape[0]) for K in K_list]

        Kglob = None
        for K, ivs, iv_corr in zip(K_list, self.ivs, self.iv_corr):
            # make the outer product tensor of shape (N_ls, N_ls, 3, 3) and multiply it with the scalar kernel
            K3D = iv_corr * K[:, :, None, None] * outer_product_multi(ivs, ivs)
            # reshape tensor onto a 2D array tiled with 3x3 matrix blocks
            if Kglob is None:
                Kglob = K3D
            else:
                Kglob += K3D
        Kglob = my_tensor_reshape(Kglob)
        # # all channels merged into one covariance matrix
        # # K^{glob}_{ij} = \sum_{k = 1}^{N_{IVs}} w_k D_{k, ij} |v_k^i\rangle \langle v_k^j |

        try:
            inv = LA.pinv2(Kglob)
        except LA.LinAlgError as err:
            print("pinv2 failed: %s. Switching to pinvh" % err)
            try:
                inv = LA.pinvh(Kglob)
            except LA.LinAlgError as err:
                print("pinvh failed: %s. Switching to pinv2" % err)
                inv = None

        # alpha is the vector of regression coefficients of GaussianProcess
        alpha = sp.dot(inv, self.y.ravel())

        if not self.low_memory:
            self.inverse = inv
            self.Kglob = Kglob
        self.alpha = sp.array(alpha)


    def predict_sollich(self, atomslist=None, eigs_t=None, ivs_t=None):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters

        atomslist : list of Atoms or AtomsList
            An list giving the atomic configurations at
            which the prediction(s) should be made.

        Returns
        -------
        y : array_like, shape (n_samples, 3)
            An array with shape (n_eval, 3) for a Gaussian Process trained on an array
            of shape (n_samples, 3) with the Best Linear Unbiased Prediction at x.
        """

        if atomslist is not None:
            ivs_t, eigs_t = self.testatoms_get_features(atomslist)
        elif (eigs_t is None or ivs_t is None):
            return None

        # Check input shapes
        n_eval, _ = ivs_t[0].shape
        n_samples_y, _ = self.y.shape
        n_features = len(ivs_t)

        # Get scalar distances between each new point in X and all input training set
        if self.metric == 'euclidean':
            dx = [(((eig_db - eig_t[:,None])**2).sum(axis=2))**0.5 for eig_db, eig_t in zip(self.eigs, eigs_t)]
        elif self.metric == 'cityblock':
            dx = [(sp.absolute(self.X - X[:,None])).sum(axis=2) for eig_db, eig_t in zip(self.eigs, eigs_t)]
        else:
            print("ERROR: metric not understood")

        # Evaluate scalar correlation
        klist = [scalar_kernel(d, self.theta0) for d in dx]

        # join vectorial features and scalar correlation together
        kglob = None
        # outer_iv = [sp.outer(iv, iv.T) for iv in self.ivs] # NO, wrong
        for k, iv_t, iv_db, iv_corr in zip(klist, ivs_t, self.ivs, self.iv_corr):
            # make the outer product tensor of shape (N_ls, N_ls, 3, 3) and multiply it with the scalar kernel
            k3D = iv_corr * k[:, :, None, None] * outer_product_multi(iv_t, iv_db)
            if kglob is None:
                kglob = k3D
            else:
                kglob += k3D

        # reshape tensor onto a 2D array tiled with 3x3 matrix blocks
        k3D = my_tensor_reshape(kglob)

        # Predictor
        return sp.dot(k3D, self.alpha).reshape(n_eval,3)


    def atomsdb_get_scalar_features(self, at_db, return_features=False):
        """
        type(at_db) == quippy.io.AtomsList
        """
        EIGs = []
        Y = []
        exps, r_cuts = self.iv_params[0], self.iv_params[1]

        # each iv is an independent information channel
        for feature, (r_cut, exp) in enumerate(zip(r_cuts, exps)):
            print("Evaluating database feature %d of %d..." % (feature+1, exps.size))
            ivs = sp.zeros((sp.asarray(at_db.n).sum(), 3))
            eigs = sp.zeros((sp.asarray(at_db.n).sum(), self.n_eigs))
            i = 0
            for atoms in at_db:
                atoms.set_cutoff(8.0)
                atoms.calc_connect()
                for at_idx in frange(atoms.n):
                    if feature == 0: Y.append(sp.array(atoms.force[at_idx])) # get target forces
                    ivs[i] = internal_vector(atoms, at_idx, exp, r_cut, do_calc_connect=False)
                    eigs[i] = coulomb_mat_eigvals(atoms, at_idx, r_cut, do_calc_connect=False, n_eigs=self.n_eigs)
                    i+=1
            IVs.append(ivs)
            EIGs.append(eigs)
        # rescale eigenvalues descriptor
        if self.normalise_scalar:
            eig_means = sp.array([e[e.nonzero()[0], e.nonzero()[1]].mean() for e in EIGs])
            eig_stds = sp.array([e[e.nonzero()[0], e.nonzero()[1]].std() for e in EIGs])
            eig_stds[eig_stds == 0.] = 1.
            EIGs = [(e - mean) / std for e, mean, std in zip(EIGs, eig_means, eig_stds)]
        # rescale internal vector to have average length = 1
        if self.normalise_ivs:
            # iv_stds = [e[e.nonzero()[0], e.nonzero()[1]].std() for e in IVs]
            # iv_stds[iv_stds == 0.] = 1.
            iv_means = [sp.array([LA.norm(vector) for vector in e]).mean() for e in IVs]
            IVs = [iv / mean for iv, mean in zip(IVs, iv_means)]

        # output cleanup: add machine epsilon if force is exactly zero
        Y = sp.asarray(Y)
        Y[sp.array(map(LA.norm, Y)) <= MACHINE_EPSILON] = 10 * MACHINE_EPSILON * sp.ones(3)

        # correlations wrt actual forces
        IV_corr = sp.array([sp.diagonal(spdist.cdist(Y, iv, metric='correlation')).mean() for iv in IVs])

	if return_features:
	    return IVs, EIGs, Y, IV_corr, iv_means, eig_means, eig_stds
	else:
	    self.ivs = IVs
	    self.eigs = EIGs
	    self.y = Y
	    self.iv_corr = IV_corr
        self.iv_means = iv_means
        self.eig_means, self.eig_stds = eig_means, eig_stds


    def testatoms_get_scalar_features(self, atomslist, iv_means=None, eig_means=None, eig_stds=None):
        """
        type(atomslist) == quippy.io.AtomsList
        """
        IVs = []
        EIGs = []
        exps, r_cuts = self.iv_params[0], self.iv_params[1]

        if not iv_means:
            iv_means = self.iv_means
        if not (eig_means or eig_stds):
            eig_means, eig_stds = self.eig_means, self.eig_stds

        # each iv is an independent information channel
        for feature, (r_cut, exp) in enumerate(zip(r_cuts, exps)):
            print("Evaluating test set feature %d of %d..." % (feature+1, exps.size))
            ivs = sp.zeros((sp.asarray(atomslist.n).sum(), 3))
            eigs = sp.zeros((sp.asarray(atomslist.n).sum(), self.n_eigs))
            i = 0
            for atoms in atomslist:
                atoms.set_cutoff(8.0)
                atoms.calc_connect()
                for at_idx in frange(atoms.n):
                    ivs[i] = internal_vector(atoms, at_idx, exp, r_cut, do_calc_connect=False)
                    eigs[i] = coulomb_mat_eigvals(atoms, at_idx, r_cut, do_calc_connect=False, n_eigs=self.n_eigs)
                    i+=1
            IVs.append(ivs)
            EIGs.append(eigs)
        # rescale eigenvalues descriptor
        if self.normalise_scalar:
            EIGs = [(e - eig_means[i]) / eig_stds[i] for i,e in enumerate(EIGs)]
        # rescale internal vector to have average length = 1
        if self.normalise_ivs:
            IVs = [iv / mean for iv, mean in zip(IVs, iv_means)]
	    return IVs, EIGs
