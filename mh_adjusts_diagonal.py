from __future__ import division
import sys

import numpy as np
na = np.newaxis

def mh_matrix(trans,target):
    N = target.shape[0]
    trans = trans / trans.sum(1)[:,na]
    errs = np.seterr(divide='ignore',invalid='ignore')

    # Pr[accept | propose i -> j] is the Metropolis acceptance probability
    conditional_accepts = np.minimum(1., target/target[:,na] * trans.T/trans)
    conditional_accepts = np.where(np.isfinite(conditional_accepts),conditional_accepts,0)

    # Pr[accept, propose i -> j] = Pr[propose i -> j] Pr[accept | proposed i->j]
    jointprobs = conditional_accepts * trans

    # Pr[i self-transitions] = 1 - Pr[i accepts exit]
    #    = 1 - \sum_j Pr[accept, propose i -> j]
    jointprobs.flat[::N+1] += 1-jointprobs.sum(1)

    np.seterr(**errs)
    return jointprobs


def test_mh_matrix(N):
    target = np.random.uniform(size=N) # no need to normalize!
    trans = np.random.uniform(size=(N,N))
    trans /= trans.sum(1)[:,na]

    P = mh_matrix(trans,target)

    evals, evecs = np.linalg.eig(P.T)
    d = evecs[:,np.abs(evals).argmax()]

    print 'max-modulus eigenvalue: %f' % np.abs(evals).max()
    print 'spectral gap: %f' % (1.-sorted(np.abs(evals))[-2])
    print 'true target: %s' % (target/target.sum())
    print 'MH invariant: %s' % (d/d.sum())

if __name__ == '__main__':
    test_mh_matrix(int(sys.argv[1]) if len(sys.argv) > 1 else 5)


