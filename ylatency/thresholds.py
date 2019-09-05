from sklearn.cluster import MeanShift
import numpy as np
from operator import itemgetter

class MSSelector:
    def __init__(self, traces, bandwidth=10, min_bin_freq=None):
        min_bin_freq = min_bin_freq or traces.count()*0.01
        self.traces = traces
        self.ms = MeanShift(bandwidth=bandwidth,
                            bin_seeding=True,
                            min_bin_freq=min_bin_freq)

    def select(self, col):
        it = map(itemgetter(col), self.traces.select(col).collect())
        X = np.fromiter(it, float).reshape(-1, 1)
    
        self.ms.fit(X)
        split_points = {}
        for x in X:
            label = self.ms.predict([x])[0]
            val = x[0]
            if label not in split_points:
                split_points[label] = val
            else:
                split_points[label] = min(val, split_points[label])
        min_ = self.traces.select(col).rdd.min()[0]
        max_ = self.traces.select(col).rdd.max()[0]
        sp = [min_]
        sp += list(split_points.values())
        sp += [max_+ 1]
        return sp

    def select_foreach(self, cols):
        return {c: self.select(c) for c in cols}
