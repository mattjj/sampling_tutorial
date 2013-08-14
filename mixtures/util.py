from __future__ import division
import numpy as np
import scipy.linalg
import scipy.stats
from matplotlib import pyplot as plt

def getdatasize(data):
    if isinstance(data,np.ndarray):
        return data.shape[0]
    elif isinstance(data,list):
        return sum(getdatasize(d) for d in data)
    else:
        assert isinstance(data,int) or isinstance(data,float)
        return 1

def flattendata(data):
    # data is either an array (possibly a maskedarray) or a list of arrays
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,list) or isinstance(data,tuple):
        if any(isinstance(d,np.ma.MaskedArray) for d in data):
            return np.ma.concatenate(data).compressed()
        else:
            return np.concatenate(data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return np.atleast_1d(data)


def sample_discrete_from_log(p_log,axis=0,dtype=np.int32):
    cumvals = np.exp(p_log - np.expand_dims(p_log.max(axis),axis)).cumsum(axis) # cumlogaddexp
    thesize = np.array(p_log.shape)
    thesize[axis] = 1
    randvals = np.random.random(size=thesize) * \
            np.reshape(cumvals[[slice(None) if i is not axis else -1
                for i in range(p_log.ndim)]],thesize)
    return np.sum(randvals > cumvals,axis=axis,dtype=dtype)

def sample_invwishart(lmbda,dof):
    n = lmbda.shape[0]
    chol = np.linalg.cholesky(lmbda)

    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(scipy.stats.chi2.rvs(dof-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)


def sample_niw(mu,lmbda,kappa,nu):
    lmbda = sample_invwishart(lmbda,nu)
    mu = np.random.multivariate_normal(mu,lmbda / kappa)
    return mu, lmbda


def plot_gaussian_2D(mu, lmbda, color='b', centermarker=True,label=''):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    if centermarker:
        plt.plot([mu[0]],[mu[1]],marker='D',color=color,markersize=4)
    plt.plot(ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],linestyle='-',linewidth=2,color=color,label=label)


def plot_gaussian_projection(mu, lmbda, vecs, **kwargs):
    '''
    Plots a ndim gaussian projected onto 2D vecs, where vecs is a matrix whose two columns
    are the subset of some orthonomral basis (e.g. from PCA on samples).
    '''
    plot_gaussian_2D(project_data(mu,vecs),project_ellipsoid(lmbda,vecs),**kwargs)

def project_data(data,vecs):
    return np.dot(data,vecs.T)

