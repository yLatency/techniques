import random
from deap.tools import cxOnePoint
import numpy as np

class Operator:
    def __init__(self, thresholds_dict, mutprob=0.2):
        self.columns = list(thresholds_dict.keys())
        self.thresholds_dict = thresholds_dict
        self.expl_maxsize = 3
        self.explset_maxsize = 3
        self.mut_prob = mutprob

    def cond(self, col):
        thresholds = self.thresholds_dict[col]
        min_ = random.choice(thresholds[:-1])
        idx = thresholds.index(min_)
        max_ = random.choice(thresholds[idx+1:])
        return col, min_, max_

    def expl(self):
        size = random.randint(1, self.expl_maxsize)
        cols = random.sample(self.columns, k=size)
        return [self.cond(c) for c in cols]

    def expllist(self):
        size = random.randint(1, self.explset_maxsize)
        return [self.expl() for _ in range(size)]

    def cx(self, ind1, ind2):
        if len(ind1) >= 1 or len(ind2) >= 1:
            i = random.randrange(len(ind1))
            j = random.randrange(len(ind2))
            ind1[i], ind2[j] = ind2[j], ind1[i]
        return ind1, ind2

    def mut(self, ind):
        self.mut_expllist(ind)
        for expl in ind:
            if np.random.uniform() < self.mut_prob:
                self.mut_expl(expl)
        return ind,

    def mut_expllist(self, expllist):
        if random.choice([True, False]):
            if len(expllist) > 1:
                expl = random.choice(expllist)
                expllist.remove(expl)
        else:
            expllist.insert(random.randrange(len(expllist)), self.expl())

    def mut_expl(self, expl):
        if random.choice([True, False]):
            if len(expl) > 1:
                cond = random.choice(expl)
                expl.remove(cond)
        else:
            columns = self.columns.copy()
            for col, _, _ in expl:
                columns.remove(col)
            if columns:
                c = random.choice(columns)
                idx = random.randrange(len(expl))
                expl.insert(idx, self.cond(c))
