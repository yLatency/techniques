from pyspark.sql.functions import col
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
import numpy as np
from abc import ABC, abstractmethod


class Selector(ABC):
    def __init__(self, normalizedTrace, backends, frontend, from_, to):
        self.backends = backends
        self.p = normalizedTrace.filter((col(frontend) > from_) & (col(frontend) <= to)).count()
        self.n = normalizedTrace.count() - self.p
        self.thresholdsDict = {}
        self.tprDict = {}
        self.fprDict = {}
        self._createThresholdsDict(normalizedTrace, backends, frontend, from_, to)

    def _createThresholdsDict(self, normalizedTrace, backends, frontend, from_, to):
        y = [1 if from_ < row[0] <= to else 0
             for row in normalizedTrace.select(frontend).collect()]
        for aBackend in backends:
            scores = [row[0] for row in normalizedTrace.select(aBackend).collect()]
            fpr, tpr, thresholds = roc_curve(y, scores)
            self.thresholdsDict[aBackend] = thresholds[:0:-1]
            self.fprDict[aBackend] = [float(fpr_) for fpr_ in fpr[:0:-1]]
            self.tprDict[aBackend] = [float(tpr_) for tpr_ in tpr[:0:-1]]

    @abstractmethod
    def select(self, k):
        pass


class KMeansSelector(Selector):
    def computeFMeasure(self, tpr, fpr):
        tp = tpr*self.p
        fp = fpr*self.n
        recall = tp/self.p
        precision = tp/(tp+fp) if tp > 0 or fp>0 else 0
        return (2*(precision*recall)/(precision+recall)
                if precision > 0 or recall > 0
                else 0)

    def select(self, k):
        thresholdDict = {}
        for aBackend in self.backends:
            thresholds = self.thresholdsDict[aBackend]
            if k + 1 >= (len(thresholds)):
                thresholdDict[aBackend] = thresholds
                continue
            tpr = self.tprDict[aBackend]
            fpr = self.fprDict[aBackend]
            X = list(zip(tpr, fpr))
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            res = []
            for i in range(k):
                indexes = [j for j in np.where(kmeans.labels_== i)[0]]
                bestIndex = max(indexes, key= lambda j: self.computeFMeasure(tpr[j], fpr[j]))
                res.append(thresholds[bestIndex])
            thresholdDict[aBackend] = [thresholds[0]] + sorted(res)
        return thresholdDict


class KMeansSelector2(Selector):
    def select(self, k):
        thresholdDict = {}
        for aBackend in self.backends:
            thresholds = self.thresholdsDict[aBackend]
            if k + 1 >= (len(thresholds)):
                thresholdDict[aBackend] = thresholds
                continue
            tpr = self.tprDict[aBackend]
            fpr = self.fprDict[aBackend]
            X = list(zip(tpr[1:], fpr[1:]))
            distances = KMeans(n_clusters=k, random_state=0).fit_transform(X)
            indices = [np.argsort(distances[:, i])[0] for i in range(k)]
            thresholdDict[aBackend] = [thresholds[0]] + sorted(thresholds[i + 1] for i in indices)
        return thresholdDict

    
class RandSelector(Selector):
    def select(self, k):
        thresholdDict = {}
        for aBackend in self.backends:
            thresholds = self.thresholdsDict[aBackend]
            if k + 1 >= (len(thresholds)):
                thresholdDict[aBackend] = thresholds
            else:
                selected = np.random.choice(thresholds[1:], size=k, replace=False)
                thresholdDict[aBackend] = sorted([thresholds[0]] + list(selected))
        return thresholdDict
