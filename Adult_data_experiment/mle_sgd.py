import numpy as np
from scipy.stats import logistic
from sklearn.utils.random import sample_without_replacement

def projPSD(Sigma):
  try:
    L = np.linalg.cholesky(Sigma)
    SigmaHat = np.dot(L, L.T)
  except np.linalg.LinAlgError:
    d, V = np.linalg.eigh(Sigma)
    SigmaHat = np.dot(V[:,d >= 1e-8], d[d >= 1e-8].reshape(-1,1) * V[:,d >= 1e-8].T)
  return SigmaHat
    
def grad_likelihood(X,Y,Sigma):
    """ This function calculates the gradient of the likelihood function using sigmoidal link """
    
    diag = np.einsum('ij,ij->i', np.matmul(X, Sigma), X)
    diag = np.squeeze(diag)
    diag = np.maximum(diag,1e-10)
    prVec = logistic.cdf(diag)
    sclVec = 2./(np.exp(diag) - 1)
    Vec = Y * prVec - (1-Y)*prVec*sclVec
    
    mat1 = np.matmul(X.T*Vec,X)/X.shape[0]
    
    return mat1

def proximal_gd_sigmoid(X, Y, mbs, maxiter=1e2):
    """ This function used proximal gradient descent algorithm to get an initial estimate of Sigma 
    based on sigmoid link function """
    
    P = X.shape[1]
    N = X.shape[0]
    m = 0
    Sigma_now = np.eye(P)
    while(m < maxiter):
        batch_idx = sample_without_replacement(N, mbs)
        X_batch, Y_batch = X[batch_idx], Y[batch_idx]
        gradnow = grad_likelihood(X_batch,Y_batch,Sigma_now)
        t = 1./(1 + m // 100)
        Sigma_new = projPSD(Sigma_now - t * gradnow)
        Sigma_now = Sigma_new        
        m += 1
        if m % 100 == 0:
            print('Done with iteration %d' % m)
            
    return Sigma_now