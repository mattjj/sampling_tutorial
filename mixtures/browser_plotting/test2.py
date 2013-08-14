from __future__ import division
from numpy import *
from numpy.random import *

from models import *
from distributions import *

obs_hypparams = dict(mu_0=zeros(2),Sigma_0=eye(2),kappa_0=0.05,nu_0=5)

model = Mixture(alpha=3,components=[Gaussian(**obs_hypparams) for i in range(10)])
model.resample_model() # initialize from prior

data = np.loadtxt('data.txt')

model.add_data(data)


import itertools, sys, json
for i in itertools.count():
    model.resample_model()
    if i % 3 == 0:
        print json.dumps(model.to_json_dict())
        sys.stdout.flush()



