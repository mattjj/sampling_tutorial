from __future__ import division
from numpy import *
from numpy.random import *

import models as m
import distributions as d

obs_hypparams = dict(mu_0=zeros(2),Sigma_0=eye(2),kappa_0=0.1,nu_0=4)

model = m.Mixture(alpha=3,components=[d.Gaussian(**obs_hypparams) for i in range(10)])
model.resample_model() # initialize from prior

# TODO load this
data = randn(20,2) + array([1,4])

model.add_data(data)
for i in range(25):
    model.resample_model()

