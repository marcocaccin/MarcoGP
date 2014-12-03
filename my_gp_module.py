
"""
Taken from sklearn.gaussian_process module and stripped out naked to the bare minimum
"""

from scipy import linalg as LA
import scipy as sp
from sklearn.utils import array2d
from sklearn.gaussian_process import correlation_models
import Cholesky

MACHINE_EPSILON = sp.finfo(sp.double).eps    


def kernel(d, theta, correlation='squared_exponential'):
    if correlation is 'absolute_exponential':
        return sp.exp(-d / theta) # correlation_models.absolute_exponential(theta, d)
    elif  correlation is 'squared_exponential':
        return sp.exp(-d**2 / (2.0 * theta**2)) # correlation_models.squared_exponential(theta, d)
    elif  correlation is 'generalized_exponential':
        return correlation_models.generalized_exponential(theta, d)
    elif  correlation is 'cubic':
        return correlation_models.cubic(theta, d)
    elif  correlation is 'linear':
        return correlation_models.linear(theta, d)
    else:
        print "Correlation model %s not understood" % correlation
        return None

    
def matrix_distance(A, B):
    # matrix distance = sum of distances of columns
    A = sp.asarray(A)
    B = sp.asarray(B)
    if not shape(A) == shape(B):
        exit
    return sp.array([sp.linalg.norm(u-v) for u, v in zip(A,B)]).sum()
        

def symmat_to_vector(A):
    n = A.shape[0]
    v = [] # sp.zeros(n * (n-1) / 2)
    for i, row in enumerate(A):
        for a in row[i+1:]:
            v.append(a)
    return sp.array(v)


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
                 normalise=1, nugget=10. * MACHINE_EPSILON,
                 low_memory=False, do_features_projection=False, metric='euclidean'):

        self.corr = corr
        self.verbose = verbose
        self.theta0 = theta0
        self.normalise = normalise
        self.nugget = nugget
        self.low_memory = low_memory
        self.do_features_projection = do_features_projection
        self.metric = metric

    def flush_data(self):
        self.X = None
        self.y = None
        if not self.low_memory:
            self.D = None
            self.K = None
        self.inverse = None
        self.alpha = None
        self.X_mean, self.X_std = None, None
        self.y_mean, self.y_std = None, None


    def calc_kernel_matrix(self, X):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """
        
        # Force data to 2D numpy.array
        X = array2d(X)
        n_samples, n_features = X.shape

        # Normalise input data or not. Do if normalise is 1 (all normalise) or 2 (input normalise)
        if self.normalise > 0:
            X_mean = sp.mean(X, axis=0)
            X_std = sp.std(X, axis=0)
            X_std[X_std == 0.] = 1.
            # center and scale X if necessary
            X = (X - X_mean) / X_std
        else:
            X_mean = 0.0 
            X_std  = 1.0 

        # Calculate distance matrix in vector form. The matrix form of X is obtained by scipy.spatial.distance.squareform(X)
        D = sp.spatial.distance.pdist(X, metric = self.metric)
        D = sp.spatial.distance.squareform(D)
        
        # Divide each distance ij by sqrt(N_i * N_j)
        if self.normalise == -1:
            natoms = (X != 0.).sum(1)
            D = D / sp.sqrt(sp.outer(natoms, natoms))
            
        # Covariance matrix K
        # sklearn correlation doesn't work. Probably correlation_models needs some different inputs 
        K = kernel(D, self.theta0, correlation=self.corr) 
        self.X = X
        if not self.low_memory:
            self.D = D
            self.K = K
        self.X_mean, self.X_std = X_mean, X_std
        return K


    def fit(self, X, y):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """

        K = self.calc_kernel_matrix(X)
        # # Force data to 2D numpy.array
        X = array2d(X)
        n_samples, n_features = X.shape
        y = sp.asarray(y)
        self.y_ndim_ = y.ndim
        if y.ndim == 1:
            y = y[:, sp.newaxis]
        _, n_targets = y.shape

        # # Normalise output data or not
        if self.normalise == 1:
            y_mean = sp.mean(y, axis=0)
            y_std = sp.std(y, axis=0)
            y_std[y_std == 0.] = 1.
            y = (y - y_mean) / y_std
        else:
            y_mean = 0.0
            y_std  = 1.0

        err = 'Dummy error message'
        inverse = K + self.nugget * sp.ones(n_samples)
        try:
            # print "is symmetric", Cholesky.isSymmetric(inverse)
            # upper_triang = Cholesky.Cholesky(inverse)
            # inverse = Cholesky.CholeskyInverse(upper_triang)
            inverse = LA.inv(inverse)
        except LA.LinAlgError as err:
            print "inv failed: %s. Switching to pinvh" % err
            try:
                inverse = LA.pinvh(inverse)
            except LA.LinAlgError as err:
                print "pinvh failed: %s. Switching to pinv2" % err
                try:
                    inverse = LA.pinv2(inverse)
                except LA.LinAlgError as err:
                    print "pinv2 failed: %s. Failed to invert matrix." % err
                    inverse = None

        # alpha is the vector of regression coefficients of GaussianProcess
        alpha = sp.dot(inverse, y)

        self.y = y
        self.y_mean, self.y_std = y_mean, y_std
        if not self.low_memory:
            self.inverse = inverse
        self.alpha = sp.array(alpha)
        

    def predict(self, X, eval_MSE=False, return_k=False):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters

        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the prediction(s) should be made.

        eval_MSE : boolean, optional
            A boolean specifying whether the Mean Squared Error should be
            evaluated or not.
            Default assumes evalMSE = False and evaluates only the BLUP (mean
            prediction).

        batch_size : integer, optional
            An integer giving the maximum number of points that can be
            evaluated simultaneously (depending on the available memory).
            Default is None so that all given points are evaluated at the same
            time.

        Returns
        -------
        y : array_like, shape (n_samples, ) or (n_samples, n_targets)
            An array with shape (n_eval, ) if the Gaussian Process was trained
            on an array of shape (n_samples, ) or an array with shape
            (n_eval, n_targets) if the Gaussian Process was trained on an array
            of shape (n_samples, n_targets) with the Best Linear Unbiased
            Prediction at x.

        MSE : array_like, optional (if eval_MSE == True)
            An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
            with the Mean Squared Error at x.
        """
        
        # Check input shapes
        X = array2d(X)
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape
        
        if X.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X.shape[1], n_features))

        X = (X - self.X_mean) / self.X_std

        # Initialize output
        y = sp.zeros(n_eval)
        if eval_MSE:
            MSE = sp.zeros(n_eval)

        # Get distances between each new point in X and all input training set
        # dx = sp.asarray([[ LA.norm(p-q) for q in self.X] for p in X]) # SLOW!!!

        if self.metric == 'euclidean':
            dx = (((self.X - X[:,None])**2).sum(axis=2))**0.5
        elif self.metric == 'cityblock':
            dx = (sp.absolute(self.X - X[:,None])).sum(axis=2)
        else:
            print "ERROR: metric not understood"

        if self.normalise == -1:
            natoms_db = (self.X != 0.).sum(1)
            natoms_t = (X != 0.).sum(1)
            dx = dx / sp.sqrt(natoms_db * natoms_t[:, None])

        # Evaluate correlation
        k = kernel(dx, self.theta0, self.corr)

        # UNNECESSARY: feature relevance
        if self.do_features_projection:
            self.feat_proj = self.alpha.flatten() * k
            y_scaled = self.feat_proj.sum(axis=1)
        else:
            # Scaled predictor
            y_scaled = sp.dot(k, self.alpha)
        # Predictor
        y = (self.y_mean + self.y_std * y_scaled).reshape(n_eval, n_targets)
        if self.y_ndim_ == 1:
            y = y.ravel()

        # Calculate mean square error of each prediction
        if eval_MSE:
            MSE = sp.dot(sp.dot(k, self.inverse), k.T)
            if k.ndim > 1: MSE = sp.diagonal(MSE)
            MSE = kernel(0.0, self.theta0, self.corr) + self.nugget - MSE
            # Mean Squared Error might be slightly negative depending on
            # machine precision: force to zero!
            MSE[MSE < MACHINE_EPSILON] = 0.
            if self.y_ndim_ == 1:
                MSE = MSE.ravel()
                if return_k:
                    return y, MSE, k
                else:
                    return y, MSE
        elif return_k:
            return y, k
        else:
            return y


#     def fitKoK(self, X, y):
        
#         # Business as usual, but now X is a list of matrices and y is a list of vectors:
#         # each element of X is the kernel matrix for a given \theta_i, each element of y is the regression coefficients vector for a given \theta_i 

#         # Force data to numpy.array
#         X = sp.asarray(X)
#         y = sp.asarray(y)

#         D = sp.zeros([len(X), len(X)])
#         for i, A in enumerate(X):
#             for j,B in enumerate(X[:i]):
#                 D[i,j] = matrix_distance(A,B)
#                 D[j,i] = D[i,j]
#         # D = sp.spatial.distance.squareform(D)
#         # Covariance matrix K
#         # sklearn correlation doesn't work. Probably correlation_models needs some different inputs 
#         K = kernel(D, self.theta0, correlation=self.corr) 
#         err = 'bb'
#         # Cholesky.CholeskyInverse(Cholesky.Cholesky(K + self.nugget * sp.eye(n_samples))) This method should work but doesn't
#         try:
#             inverse = LA.inv(K + self.nugget * sp.ones(n_samples))
#         except  LA.LinAlgError as err:
#             print "inv failed: %s. Switching to pinvh" % err
#             try:
#                 inverse = LA.pinvh(K + self.nugget * sp.eye(n_samples))
#             except LA.LinAlgError as err:
#                 print "pinvh failed: %s. Switching to pinvh" % err
#                 try:
#                     inverse = LA.pinv2(K + self.nugget * sp.eye(n_samples))
#                 except LA.LinAlgError as err:
#                     print "pinv2 failed: %s. Failed to invert matrix." % err
#                     inverse = None

#         # alpha is the vector of regression coefficients of GaussianProcess
#         alpha = sp.dot(inverse, y)

#         self.X = X
#         self.y = y
#         if not self.low_memory:
#             self.D = D
#             self.K = K
#         self.inverse = inverse
#         self.alpha = sp.array(alpha)
#         self.X_mean, self.X_std = 1.0, 0.0
#         self.y_mean, self.y_std = 1.0, 0.0



#     def predict_KoK(self, X):
#         """
#         This function evaluates the Gaussian Process model at a set of points X.

#         Parameters

#         X : array_like
#             An array with shape (n_eval, n_features) giving the point(s) at
#             which the prediction(s) should be made.

#         Returns
#         -------
#         y : array_like, shape (n_samples, ) or (n_samples, n_targets)
#             An array with shape (n_eval, ) if the Gaussian Process was trained
#             on an array of shape (n_samples, ) or an array with shape
#             (n_eval, n_targets) if the Gaussian Process was trained on an array
#             of shape (n_samples, n_targets) with the Best Linear Unbiased
#             Prediction at x.
#         """
        
#         # Check input shapes
#         X = array2d(X)
#         n_eval, _ = X.shape
#         n_samples, n_features = self.X.shape
#         n_samples_y, n_targets = self.y.shape
        
#         if X.shape[1] != n_features:
#             raise ValueError(("The number of features in X (X.shape[1] = %d) "
#                               "should match the number of features used "
#                               "for fit() "
#                               "which is %d.") % (X.shape[1], n_features))

#         X = (X - self.X_mean) / self.X_std

#         # Initialize output
#         y = sp.zeros(n_eval)
#         if eval_MSE:
#             MSE = sp.zeros(n_eval)

#         # Get distances between each new point in X and all input training set
#         # dx = sp.asarray([[ LA.norm(p-q) for q in self.X] for p in X]) # SLOW!!!
#         dx = (((self.X - X[:,None])**2).sum(axis=2))**0.5
#         # Evaluate correlation
#         k = kernel(dx, self.theta0, self.corr)

#         # UNNECESSARY: feature relevance
#         if self.do_features_projection:
#             self.feat_proj = self.alpha.flatten() * k
#             y_scaled = self.feat_proj.sum(axis=1)
#         else:
#         # Scaled predictor
#             y_scaled = sp.dot(k, self.alpha)
#         # Predictor
#         y = (self.y_mean + self.y_std * y_scaled).reshape(n_eval, n_targets)
#         if self.y_ndim_ == 1:
#             y = y.ravel()

#         # Calculate mean square error of each prediction
#         if eval_MSE:
#             MSE = sp.dot(sp.dot(k, self.inverse), k.T)
#             if k.ndim > 1: MSE = sp.diagonal(MSE)
#             MSE = kernel(0.0, self.theta0, self.corr) + self.nugget - MSE
#             # Mean Squared Error might be slightly negative depending on
#             # machine precision: force to zero!
#             MSE[MSE < MACHINE_EPSILON] = 0.
#             if self.y_ndim_ == 1:
#                 MSE = MSE.ravel()
#         return y, MSE
