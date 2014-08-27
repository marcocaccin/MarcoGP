"""
file  Cholesky.py
author Ernesto P. Adorio, Ph.D.
       ernesto.adorio@gmail.com
       UPDEPP at Clark Field, Pampanga
version 0.0.2 april 3, 2012  added try block for zero diagonals.
"""
 
from math import *
# from matlib import *
from scipy.linalg import *
 
def isSymmetric(A, ztol = 1.0e-5, force=False):
    """
    Returns True if A is symmetric and if force is True,
    makes it purely symmetric.
    """
    nrows = len(A)
    if len(A) != len(A[0]):
       return False
    for i in range(nrows):
       for j in range(i+1, nrows):
           d = A[i][j] - A[j][i]
           if abs(d) > ztol:
              return False
           if force:
              A[i][j] = A[j][i] = A[i][j] - d / 2.0
    return True
 
def Cholesky(A, ztol= 1.0e-5):
    """
    Computes the upper triangular Cholesky factorization of  
    a positive definite matrix A.
    """
    # Forward step for symmetric triangular t.
    nrows = len(A)
    t = matzero(nrows, nrows)
    for i in range(nrows):
        S = sum([(t[k][i])**2 for k in range(i)])
        d = A[i][i] -S
        if abs(d) < ztol:
           t[i][i] = 0.0
        else: 
           if d < 0.0:
              raise ValueError, "Matrix not positive-definite"
           t[i][i] = sqrt(d)
        for j in range(i+1, nrows):
           S = sum([t[k][i] * t[k][j] for k in range(i)])
           if abs(S) < ztol:
              S = 0.0
           try:
              t[i][j] = (A[i][j] - S)/t[i][i]
           except:
              raise ValueError, "Zero diagonal"
    return(t)
 
def CholeskyInverse(t):
    """ 
    Computes inverse of matrix given its Cholesky upper Triangular
    decomposition t.
    """
    nrows = len(t)
    B = matzero(nrows, nrows)
 
    # Backward step for inverse.
    for j in reversed(range(nrows)):
        tjj = t[j][j]
        S = sum([t[j][k]*B[j][k] for k in range(j+1, nrows)])
        B[j][j] = 1.0/ tjj**2 - S/ tjj
        for i in reversed(range(j)):
            B[j][i] = B[i][j] = -sum([t[i][k]*B[k][j] for k in range(i+1,nrows)])/t[i][i]
    return B
 
if __name__ == "__main__":
   A = [[ 2,-1,  0], 
        [-1, 2,-1],
        [ 0,-1, 2]]
   print "Input symmetic matrix A:"
   matprint(A)
   print "Matrix inverse by matlib matinverse Ainv:"
   Ainv = matinverse(A)
   matprint(Ainv)
   print "Matrix product A Ainv:"
   matprint(matmul(Ainv, A))
 
   print "Cholesky T from scipy"
   matprint(cholesky(A))                        
   print "Cholesky T from our Cholesky() function."
   t = Cholesky(A)
   matprint(t)
   print "Matrix inverse from Cholesky factorization B:"
   B = CholeskyInverse(t)
   matprint(B)
   print "Product A B:"
   matprint(matmul(A,B))
   #added jan 4,2010
   print "product of T T^t != A:"
   matprint(matprod(t, transpose(t)))
   print "product of T^t T = A:"
   matprint(matprod(transpose(t), t))
