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
 
def likelihood(X,Y,Sigma):
    """ This function calculates the likelihood of U under sigmoidal link"""
     
    Vec1 = np.einsum('ij,ij->i', np.matmul(X, Sigma), X)

    return (np.log(1 + np.exp(Vec1)) - (1-Y)*Vec1).mean()
 
     
def grad_likelihood(X,Y,Sigma):
    """ This function calculates the gradient of the likelihood function using sigmoidal link """
     
    diag = np.einsum('ij,ij->i', np.matmul(X, Sigma), X)
    diag = np.maximum(diag,1e-10)
    prVec = logistic.cdf(diag)
    sclVec = 2./(np.exp(diag) - 1)
    Vec = Y * prVec - (1-Y)*prVec*sclVec
     
    mat1 = np.matmul(X.T*Vec,X)/X.shape[0]
     
    return mat1
 
def proximal_gd_sigmoid(X,Y,mbs,gradtol=1e-4, maxiter=1e2, alpha=0.01, beta=0.5):
    """ This function used proximal gradient descent algorithm to get an initial estimate of Sigma 
    based on sigmoid link function """
    P = X.shape[1]
    N = X.shape[0]
    m = 0
    Sigma_now = np.random.normal(0,1,P**2).reshape(P,P)
    Sigma_now = np.matmul(Sigma_now, Sigma_now.T)
    Sigma_now = Sigma_now/np.linalg.norm(Sigma_now)
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

    print('Proximal GD done!')
    return Sigma_now