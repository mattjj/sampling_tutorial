from __future__ import division
import numpy as np
import scipy.linalg
from numpy.core.umath_tests import inner1d
from matplotlib import pyplot as plt

import util

class Categorical(object):
    def __init__(self,weights=None,alpha=None):
        self.weights = weights
        self.alpha = alpha

    def log_likelihood(self,x):
        return np.log(self.weights)[x]

    def resample(self,data=[]):
        posterior_alpha = self._posterior_hypparams(*self._get_statistics(data))
        self.weights = np.random.dirichlet(posterior_alpha)

    def _get_statistics(self,data):
        K = len(self.alpha)
        if isinstance(data,np.ndarray):
            counts = np.bincount(data,minlength=K)
        else:
            counts = sum(np.bincount(d,minlength=K) for d in data)

        return counts,

    def _posterior_hypparams(self,counts):
        return self.alpha + counts


class Gaussian(object):
    def __init__(self,mu=None,Sigma=None,mu_0=None,Sigma_0=None,kappa_0=None,nu_0=None):
        self.mu = mu
        self.Sigma = Sigma
        self.mu_0 = mu_0
        self.Sigma_0 = Sigma_0
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0

    def log_likelihood(self,x):
        mu, Sigma = self.mu, self.Sigma
        D = len(mu) if isinstance(mu,np.ndarray) else 1
        L = np.linalg.cholesky(Sigma)
        xc = np.reshape(x,(-1,D)) - mu
        xs = scipy.linalg.solve_triangular(L,xc.T,lower=True)
        return -1./2. * inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi) \
                - np.log(L.diagonal()).sum()

    def resample(self,data=[]):
        self.mu, self.Sigma = util.sample_niw(
                *self._posterior_hypparams(*self._get_statistics(data)))

    def _get_statistics(self,data):
        D = len(self.mu) if isinstance(self.mu,np.ndarray) else 1
        n = util.getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                xbar = np.reshape(data,(-1,D)).mean(0)
                centered = data - xbar
                sumsq = np.dot(centered.T,centered)
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,(np.reshape(d,(-1,D))-xbar))
                        for d in data)
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq

    def _posterior_hypparams(self,n,xbar,sumsq):
        mu_0, Sigma_0, kappa_0, nu_0 = self.mu_0, self.Sigma_0, self.kappa_0, self.nu_0
        if n > 0:
            mu_n = self.kappa_0 / (self.kappa_0 + n) * self.mu_0 + n / (self.kappa_0 + n) * xbar
            kappa_n = self.kappa_0 + n
            nu_n = self.nu_0 + n
            sigma_n = self.Sigma_0 + sumsq + \
                    self.kappa_0*n/(self.kappa_0+n) * np.outer(xbar-self.mu_0,xbar-self.mu_0)

            return mu_n, sigma_n, kappa_n, nu_n
        else:
            return mu_0, Sigma_0, kappa_0, nu_0

    ### plotting

    def plot(self,data=None,indices=None,color='b',plot_params=True,label=''):
        from util import project_data, plot_gaussian_projection, plot_gaussian_2D
        if data is not None:
            data = util.flattendata(data)

        D = len(self.mu) if isinstance(self.mu,np.ndarray) else 1
        assert D >= 2

        if D > 2 and ((not hasattr(self,'plotting_subspace_basis'))
                or (self.plotting_subspace_basis.shape[1] != D)):

            subspace = np.random.randn(D,2)
            self.__class__.plotting_subspace_basis = np.linalg.qr(subspace)[0].T.copy()

        if data is not None:
            if D > 2:
                data = project_data(data,self.plotting_subspace_basis)
            plt.plot(data[:,0],data[:,1],marker='.',linestyle=' ',color=color)

        if plot_params:
            if D > 2:
                plot_gaussian_projection(self.mu,self.Sigma,self.plotting_subspace_basis,
                        color=color,label=label)
            else:
                plot_gaussian_2D(self.mu,self.Sigma,color=color,label=label)

    def to_json_dict(self):
        assert len(self.mu) == 2
        U,s,_ = np.linalg.svd(self.Sigma)
        U /= np.linalg.det(U)
        theta = np.arctan2(U[0,0],U[0,1])*180/np.pi
        return {'x':self.mu[0],'y':self.mu[1],'rx':np.sqrt(s[0]),'ry':np.sqrt(s[1]),
                'theta':theta}


# TODO diagonal gaussian, diagonal gaussian with truncated obs

