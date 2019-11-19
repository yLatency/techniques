from functools import reduce

from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from operator import itemgetter, add


class MSSelector:
    def __init__(self, traces, bandwidth=None, min_bin_freq=1):
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
        max_ = self.traces.select(col).rdd.max()[0]
        sp = list(split_points.values())
        sp += [max_ + 1]
        return sorted(sp)

    def select_foreach(self, cols):
        return {c: self.select(c) for c in cols}


class CacheMaker:
    def __init__(self, traces, backends,
                 frontend, from_, to):
        self.traces = traces.toPandas()
        self.frontend = frontend
        self.backends = backends
        self.from_ = from_
        self.to = to

    def create(self, thr_dict):
        pos = self.get_positives().count()[self.frontend]
        cache = {'p': pos,
                 'n': self.traces.count()[self.frontend] - pos}

        for b in self.backends:
            tp_intlist = self.create_tp(b, thr_dict[b])
            fp_intlist = self.create_fp(b, thr_dict[b])
            size = len(thr_dict[b])
            for i in range(size):
                cache[b, i] = tp_intlist[i], fp_intlist[i]

        return cache

    def get_positives(self):
        df = self.traces
        return df[(df[self.frontend] > self.from_) & (df[self.frontend] <= self.to)]

    def get_negatives(self):
        df = self.traces
        return df[(df[self.frontend] <= self.from_) | (df[self.frontend] > self.to)]

    def create_tp(self, backend, thresholds):
        pos = self.get_positives()
        return self.create_bitslists(pos, backend, thresholds)

    def create_bitslists(self, df, backend, thresholds):
        list_bitstring = []
        for t in thresholds:
            bitstring = reduce(add, df[backend].map(lambda x: '1' if x >= t else '0'))
            num = int(bitstring, 2)
            list_bitstring.append(num)
        return list_bitstring

    def create_fp(self, backend, thresholds):
        neg = self.get_negatives()
        return self.create_bitslists(neg, backend, thresholds)