from __future__ import division
import numpy as np

import util
import distributions as d

class Mixture(object):
    def __init__(self,alpha,components):
        self.components = components
        self.weights = d.Categorical(alpha=np.repeat(alpha,len(components))/len(components))
        self.labels_list = []

    def add_data(self,data):
        self.labels_list.append(Labels(self,data))

    def resample_model(self):
        for l in self.labels_list:
            l.resample()

        self.weights.resample(data=[l.z for l in self.labels_list])

        for label_idx, distn in enumerate(self.components):
            distn.resample(data=[l.data[l.z == label_idx] for l in self.labels_list])

    ### plotting

    def plot(self,legend=True):
        from matplotlib import pyplot as plt
        from matplotlib import cm
        cmap = cm.get_cmap()

        if len(self.labels_list) > 0:
            label_colors = {}

            used_labels = reduce(set.union,[set(l.z) for l in self.labels_list],set([]))
            num_labels = len(used_labels)
            num_subfig_rows = len(self.labels_list)

            for idx,label in enumerate(used_labels):
                label_colors[label] = idx/(num_labels-1 if num_labels > 1 else 1)

            for subfigidx,l in enumerate(self.labels_list):
                # plot the current observation distributions (and obs. if given)
                plt.subplot(num_subfig_rows,1,1+subfigidx)
                for label, o in enumerate(self.components):
                    if label in l.z:
                        o.plot(color=cmap(label_colors[label]),
                                data=(l.data[l.z == label] if l.data is not None else None),
                                label='%d' % label)

            if legend:
                plt.legend(
                        [plt.Rectangle((0,0),1,1,fc=cmap(c))
                            for i,c in label_colors.iteritems() if i in used_labels],
                        [i for i in label_colors if i in used_labels],
                        loc='best'
                        )

        else:
            top10 = np.array(self.components)[np.argsort(self.weights.weights)][-1:-11:-1]
            colors = [cmap(x) for x in np.linspace(0,1,len(top10))]
            for i,(o,c) in enumerate(zip(top10,colors)):
                o.plot(color=c,label='%d' % i)

    def to_json_dict(self):
        assert len(self.labels_list) == 1
        data = self.labels_list[0].data
        z = self.labels_list[0].z
        assert data.ndim == 2 and data.shape[1] == 2

        return  {
                    'points':[{'x':x,'y':y,'label':int(label)} for x,y,label in zip(data[:,0],data[:,1],z)],
                    'ellipses':[dict(c.to_json_dict().items() + [('label',i)])
                        for i,c in enumerate(self.components) if i in z]
                }


class Labels(object):
    def __init__(self,model,data):
        self.data = data
        self.model = model

    def resample(self):
        data, model = self.data, self.model
        N, K = len(data), len(model.components)
        log_scores = np.empty((N,K))

        for idx, distn in enumerate(model.components):
            log_scores[:,idx] = distn.log_likelihood(data)
        log_scores += model.weights.log_likelihood(np.arange(K))

        self.z = util.sample_discrete_from_log(log_scores,axis=1)

