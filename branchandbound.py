from collections import deque
from metrics import Metrics, Explanation


class Node:
    def __init__(self, explanation, remainingFeatures, mu, parentNode=None):
        self.exp = explanation
        self.remainingFeatures = remainingFeatures
        self.parentNode = parentNode
        self.mu = mu

    def createChild(self, exp, remainingFeatures, mu):
        return Node(exp, remainingFeatures,
                    mu, self)


class BranchAndBound:
    def __init__(self, traces, backends,
                 thresholds, frontend, from_, to):
        self.metrics = Metrics(traces, backends,
                               thresholds, frontend, from_, to)
        self.features = set(backends)
        self.bestExp = None
        self.queue = None

    def updateBestSol(self, exp):
        if self.bestExp is None or self.bestExp.fmeasure <= exp.fmeasure:
            self.bestExp = exp

    def checkRecallBound(self, exp):
        return exp.recall >= self.bestExp.fmeasure / (2 - self.bestExp.fmeasure)

    @staticmethod
    def checkBoundDecrSupp(support, parentSupport,
                           mu, remainingFeatures):
        decrSupp = parentSupport - support
        return decrSupp >= (parentSupport - mu) / len(remainingFeatures)

    def createFirstNode(self, emptyExp):
        return Node(explanation=emptyExp,
                    remainingFeatures=self.features,
                    mu=self.metrics.posCount / emptyExp.precision)

    def createEmptyExp(self):
        emptySet = frozenset()
        metrics = self.metrics.compute(emptySet)
        return Explanation(emptySet, *metrics)

    def createExplanation(self, exp, featureToAdd):
        features = exp.features | {featureToAdd}
        metrics = self.metrics.compute(features)
        return Explanation(features, *metrics)

    def compute(self):
        emptyExp = self.createEmptyExp()
        rootNode = self.createFirstNode(emptyExp)
        self.bestExp = emptyExp
        self.queue = deque([rootNode])
        while self.queue:
            node = self.queue.popleft()
            features = node.remainingFeatures.copy()
            parentSupport = node.exp.support
            for featureToAdd in node.remainingFeatures:
                exp = self.createExplanation(node.exp, featureToAdd)
                if self.checkRecallBound(exp) and self.checkBoundDecrSupp(exp.support, parentSupport,
                                                                          node.mu, features):
                    self.updateBestSol(exp)
                    features.remove(featureToAdd)
                    mu = min(node.mu, self.metrics.posCount / exp.precision)
                    childNode = node.createChild(exp, features.copy(), mu)
                    self.queue.append(childNode)
        return self.bestExp
