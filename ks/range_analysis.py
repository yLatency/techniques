from ylatency.ga import GA
from ks.branchandbound import BranchAndBound


class RangeAnalysis:
    def __init__(self, traces, backends,
                 frontend, splitPoints):
        self.traces = traces
        self.backends = backends
        self.frontend = frontend
        self.splitPoints = splitPoints

    def _explainSingleIntervalGA(self, from_, to, thresholds_dict):
        ga = GA(self.traces,
                self.backends,
                self.frontend,
                thresholds_dict)
        return ga.compute(from_, to)

    def _explainSingleIntervalBnB(self, from_, to, thresholds_dict):
        bnb = BranchAndBound(self.traces,
                             self.frontend,
                             thresholds_dict)
        return bnb.compute(from_, to)

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

    def explainsWithGA(self, thresholds_dict,  bandwidth=10):
        n = len(self.splitPoints) - 1
        f = lambda from_, to: self._explainSingleIntervalGA(from_, to, thresholds_dict,  bandwidth)
        return self._computeBestSplits(n, f)

    def explainsWithBnB(self, thresholds_dict):
        n = len(self.splitPoints) - 1
        f = lambda from_, to: self._explainSingleIntervalBnB(from_, to, thresholds_dict)
        return self._computeBestSplits(n, f)
