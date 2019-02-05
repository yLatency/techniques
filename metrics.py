from functools import reduce
from pyspark.sql.functions import col

class Metrics:
    def __init__(self, traces, backends,
                 thresholds, frontend, frontendSLA):
        self.posTraces = traces.filter(col(frontend) > frontendSLA)
        self.negTraces = traces.filter(col(frontend) <= frontendSLA)
        self.posCount = self.posTraces.count()
        self.thresholdDict = dict(zip(backends, thresholds))
        self.frontend = frontend
        self.frontendSLA = frontendSLA
        if self.posCount <= 0:
            raise Exception('No positives')

    def _countAboveThresholds(self, traces, features):
        filtered = reduce(lambda df, b: df.filter(col(b) >= self.thresholdDict[b]),
                          features,
                          traces)
        return filtered.count()

    def _computeTpAndFp(self, features):
        return (self._countAboveThresholds(traces, features)
                for traces in [self.posTraces, self.negTraces])

    def compute(self, features):
        tp, fp = self._computeTpAndFp(features)
        rec = tp / self.posCount
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        fmeasure = 2 * (prec * rec) / (prec + rec) if prec > 0 or rec > 0 else 0
        support = tp + fp
        return fmeasure, prec, rec, support


class Explanation:
    def __init__(self, features, fmeasure, precision, recall, support):
        self.features = features
        self.fmeasure = fmeasure
        self.precision = precision
        self.recall = recall
        self.support = support