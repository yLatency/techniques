from gurobipy import GRB, Model
from thresholds import Selector


class _ThresholdModel:
    def __init__(self, thresholds,
                 tpr, fpr, k):
        self.thresholds = thresholds
        self.k = k
        self.tpr = tpr
        self.fpr = fpr
        self.initModel()

    def initModel(self):
        self.m = Model("m")
        self.createVars()
        self.addConstrs()
        self.setObj()

    def createVars(self):
        m, n, k = self.m, len(self.thresholds) - 1, self.k
        self.x = m.addVars(n, k, vtype=GRB.BINARY, name="x")
        self.zt = m.addVars(k, vtype=GRB.CONTINUOUS, name="zt")
        self.zf = m.addVars(k, vtype=GRB.CONTINUOUS, name="zf")
        self.yt = m.addVars(k, vtype=GRB.CONTINUOUS, name="yt")
        self.yf = m.addVars(k, vtype=GRB.CONTINUOUS, name="yf")
        self.ytmax = m.addVar(vtype=GRB.CONTINUOUS, name='ytmax')
        self.yfmax = m.addVar(vtype=GRB.CONTINUOUS, name='yfmax')
        self.ytmin = m.addVar(vtype=GRB.CONTINUOUS, name='ytmin')
        self.yfmin = m.addVar(vtype=GRB.CONTINUOUS, name='yfmin')
        self.st = m.addVar(vtype=GRB.CONTINUOUS, name="st")
        self.sf = m.addVar(vtype=GRB.CONTINUOUS, name="sf")

    def addConstrs(self):
        self.addXConstrs()
        self.addZConstrs()
        self.addYConstrs()
        self.addYMaxMinConstrs()
        self.addSConstrs()

    def addXConstrs(self):
        n = len(self.thresholds) - 1
        self.m.addConstrs((self.x.sum('*', i) == 1
                           for i in range(self.k)), 'c1')
        self.m.addConstrs((self.x.sum(i, '*') <= 1
                           for i in range(n)), 'c2')

    def addZConstrs(self):
        for i in range(self.k):
            coeffTP, coeffFP = self.createCoeffs(i)
            self.m.addConstr(self.x.prod(coeffTP, '*', i) == self.zt[i],
                             'c3.%d' % i)
            self.m.addConstr(self.x.prod(coeffFP, '*', i) == self.zf[i],
                             'c4.%d' % i)
            if i == 0:
                self.m.addConstr(self.tpr[i] >= self.zt[i],
                                 'c5.%d' % i)
                self.m.addConstr(self.fpr[i] >= self.zf[i],
                                 'c6.%d' % i)
            else:
                self.m.addConstr(self.zt[i - 1] >= self.zt[i],
                                 'c5.%d' % i)
                self.m.addConstr(self.zf[i - 1] >= self.zf[i],
                                 'c6.%d' % i)

    def addYConstrs(self):
        self.m.addConstr(self.tpr[0] - self.zt[0] == self.yt[0], 'c7.0')
        self.m.addConstr(self.fpr[0] - self.zf[0] == self.yf[0], 'c8.0')
        for i in range(1, self.k):
            self.m.addConstr(self.zt[i - 1] - self.zt[i] == self.yt[i], 'c7.%d' % i)
            self.m.addConstr(self.zf[i - 1] - self.zf[i] == self.yf[i], 'c8.%d' % i)

    def addYMaxMinConstrs(self):
        self.m.addGenConstrMax(self.ytmax, self.yt, name='c.9')
        self.m.addGenConstrMax(self.yfmax, self.yf, name='c.10')
        self.m.addGenConstrMin(self.ytmin, self.yt, name='c.11')
        self.m.addGenConstrMin(self.yfmin, self.yf, name='c.12')

    def addSConstrs(self):
        self.m.addConstr(self.zt[self.k - 1] - self.tpr[-1] == self.st, 'c13')
        self.m.addConstr(self.zf[self.k - 1] - self.fpr[-1] == self.sf, 'c14')

    def createCoeffs(self, i):
        tpr, fpr = self.tpr[1:], self.fpr[1:]
        coeffTP = {(j, i): tpr_ for j, tpr_ in enumerate(tpr)}
        coeffFP = {(j, i): fpr_ for j, fpr_ in enumerate(fpr)}
        return coeffTP, coeffFP

    def setObj(self):
        obj = (self.sf - self.yfmin) + (self.yfmax - self.yfmin) + (self.ytmax - self.ytmin) + (self.st - self.ytmin)
        self.m.setObjective(obj, GRB.MINIMIZE)

    def compute(self):
        self.m.optimize()
        selectedThresholds = [0]
        for i, j in self.x:
            if self.x[i, j].X == 1:
                selectedThresholds.append(self.thresholds[i + 1])
        return selectedThresholds


class MIPSelector(Selector):
    def select(self, k):
        thresholdsDict = {}
        for aBackend in self.backends:
            thresholds = self.thresholdsDict[aBackend]
            if k + 1 >= (len(thresholds)):
                thresholdsDict[aBackend] = thresholds
                continue
            tpr = self.tprDict[aBackend]
            fpr = self.fprDict[aBackend]
            selected = _ThresholdModel(thresholds, tpr, fpr, k).compute()
            thresholdsDict[aBackend] = selected
        return thresholdsDict
