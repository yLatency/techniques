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


class Hashtable:
    def __init__(self, traces, backends,
                 frontend, from_, to):
        self.traces = traces.toPandas()
        self.frontend = frontend
        self.backends = backends
        self.from_ = from_
        self.to = to

    # it returns a single hashtable where keys are pair (col, indexofthreshold)
    # and values are pairs (bitstring positives, bitstring negatives)
    def all_in_one(self, thr_dict):
        pos = self._get_positives().count()[self.frontend]
        cache = {'p': pos,
                 'n': self.traces.count()[self.frontend] - pos}

        for b in self.backends:
            tp_intlist = self._create_tp(b, thr_dict[b])
            fp_intlist = self._create_fp(b, thr_dict[b])
            size = len(thr_dict[b])
            for i in range(size):
                cache[b, i] = tp_intlist[i], fp_intlist[i]

        return cache

    # it returns a hashtable where keys are pair (col, threshold) and values are bitstrings representing positives
    def positives(self, thr_dict):
        return self._create_hashtable(thr_dict, positives=True)

    # it returns a hashtable where keys are pair (col, threshold) and values are bitstrings representing negatives
    def negatives(self, thr_dict):
        return self._create_hashtable(thr_dict, positives=False)

    def _create_hashtable(self, thr_dict, positives=True):
        hashtable = {'cardinality':  self._count_positives() if positives else self._count_negatives()}
        create_bitstrings = self._create_tp if positives else self._create_fp
        for b in self.backends:
            bitstrings = create_bitstrings(b, thr_dict[b])
            for t, bs in zip(thr_dict[b], bitstrings):
                hashtable[b, t] = bs
        return hashtable

    def _get_positives(self):
        df = self.traces
        return df[(df[self.frontend] > self.from_) & (df[self.frontend] <= self.to)]

    def _get_negatives(self):
        df = self.traces
        return df[(df[self.frontend] <= self.from_) | (df[self.frontend] > self.to)]

    def _count_positives(self):
        return self._get_positives().count()[self.frontend]

    def _count_negatives(self):
        return self._get_negatives().count()[self.frontend]

    def _create_tp(self, backend, thresholds):
        pos = self._get_positives()
        return self._create_bitslists(pos, backend, thresholds)

    def _create_bitslists(self, df, backend, thresholds):
        list_bitstring = []
        for t in thresholds:
            bitstring = reduce(add, df[backend].map(lambda x: '1' if x >= t else '0'))
            num = int(bitstring, 2)
            list_bitstring.append(num)
        return list_bitstring

    def _create_fp(self, backend, thresholds):
        neg = self._get_negatives()
        return self._create_bitslists(neg, backend, thresholds)