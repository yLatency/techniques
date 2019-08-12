from ylatency.ga import GA
from ks.branchandbound import BranchAndBound


class RangeAnalysis:
    def __init__(self, traces, backends,
                 frontend, splitPoints):
        self.traces = traces
        self.backends = backends
        self.frontend = frontend
        self.splitPoints = splitPoints

    def _explainSingleIntervalGA(self, from_, to, k):
        ga = GA(self.traces,
                self.backends,
                self.frontend,
                from_, to)
        return ga.compute()

    def _explainSingleIntervalBnB(self, from_, to, thresholds):
        exp = BranchAndBound(self.traces,
                             self.backends,
                             thresholds,
                             self.frontend,
                             from_,
                             to).compute()
        return exp.features, exp.fmeasure, exp.precision, exp.recall

    def _computeBestSplits(self, i, f):
        if i == 0:
            return [], 0, ()
        else:
            bestSelectedSplits, bestSumFMeas, bestExplanations = None, None, None
            for j, v in enumerate(self.splitPoints[:i]):
                selectedSplits, sumFmeas, fmeasures = self._computeBestSplits(j, f)
                exp = f(self.splitPoints[j], self.splitPoints[i])
                fmeas = exp[1]
                sumFmeas += fmeas
                if bestSumFMeas is None or bestSumFMeas < sumFmeas:
                    bestSelectedSplits = selectedSplits + [self.splitPoints[j]]
                    bestSumFMeas = sumFmeas
                    bestExplanations = *fmeasures, exp

            return bestSelectedSplits, bestSumFMeas, bestExplanations

    def explainsWithGA(self, k):
        n = len(self.splitPoints) - 1
        f = lambda from_, to: self._explainSingleIntervalGA(from_, to, k)
        return self._computeBestSplits(n, f)

    def explainsWithBnB(self, thresholds):
        n = len(self.splitPoints) - 1
        f = lambda from_, to: self._explainSingleIntervalBnB(from_, to, thresholds)
        return self._computeBestSplits(n, f)
