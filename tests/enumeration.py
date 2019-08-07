from itertools import combinations
from ks.metrics import Metrics, Explanation


class Enumeration:
    def __init__(self, traces, backends,
                 thresholds, frontend, from_, to):
        self.metrics = Metrics(traces, backends,
                               thresholds, frontend, from_, to)
        self.features = set(backends)
        self.bestExp = None

    def allCombinations(self):
        for s in range(1, len(self.features) + 1):
            for comb in combinations(self.features, s):
                yield frozenset(comb)

    def compute(self):
        emptySet = frozenset()
        metrics = self.metrics.compute(emptySet)
        self.bestExp = Explanation(emptySet, *metrics)
        for featuresComb in self.allCombinations():
            metrics = self.metrics.compute(featuresComb)
            exp = Explanation(featuresComb, *metrics)
            if exp.fmeasure > self.bestExp.fmeasure:
                self.bestExp = exp
        return self.bestExp
