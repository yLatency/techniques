from functools import reduce
from pyspark.sql.functions import col
from deap import base, creator, tools, algorithms

from thresholds import RandSelector, KMeansSelector


class CacheMaker:
    def __init__(self, traces, backends,
                 frontend, from_, to):
        self.traces = traces
        self.frontend = frontend
        self.backends = backends
        self.from_ = from_
        self.to = to

    def create(self, thr_dict):
        cache = {}
        for b in self.backends:
            tp_intlist = self.create_tp(b, thr_dict[b])
            fp_intlist = self.create_fp(b, thr_dict[b])
            size = len(thr_dict[b])
            for i in range(size):
                cache[b, i] = tp_intlist[i], fp_intlist[i]

        return cache


    def create_tp(self, backend, thresholds):
        pos = (self.traces.filter((col(self.frontend) > self.from_) &
                                    (col(self.frontend) <= self.to)))

        return self.create_bitslists(pos, backend, thresholds)

    def create_bitslists(self, filtered_traces, backend, thresholds):
        sorted_traces = filtered_traces.sortWithinPartitions('traceId')
        list_bitstring = (sorted_traces.rdd.map(lambda row: ['1' if row[backend] >= t else '0' for t in thresholds])
                                           .reduce(lambda x, y: [a + b for a, b in zip(x, y)]))
        return [int(bs, 2) for bs in list_bitstring]

    def create_fp(self, backend, thresholds):
        neg = self.traces.filter((col(self.frontend) <= self.from_) |
                                 (col(self.frontend) > self.to))
        return self.create_bitslists(neg, backend, thresholds)


class FitnessUtils:
    def __init__(self, backends, cache):
        self.backends = backends
        self.cache = cache
        self.p = bin(self.getTPBitString(self.backends[0], 0)).count('1')

    def countOnesInConjunctedBitStrings(self, ind, getter):
        bit = reduce(lambda bits, bt: bits & getter(*bt),
                     zip(self.backends[1:], ind[1:]),
                     getter(self.backends[0], ind[0]))
        return bin(bit).count("1")

    def getTPBitString(self, backend, threshold):
        return self.cache[backend, threshold][0]

    def getFPBitString(self, backend, threshold):
        return self.cache[backend, threshold][1]

    def computeTP(self, ind):
        getter = lambda backend, threshold: self.getTPBitString(backend, threshold)
        return self.countOnesInConjunctedBitStrings(ind, getter)

    def computeFP(self, ind):
        getter = lambda backend, threshold: self.getFPBitString(backend, threshold)
        return self.countOnesInConjunctedBitStrings(ind, getter)

    def computePrecRec(self, ind):
        tp = self.computeTP(ind)
        fp = self.computeFP(ind)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / self.p
        return prec, rec

    def computeFMeasure(self, ind):
        prec, rec = self.computePrecRec(ind)
        return 2 * (prec * rec) / (prec + rec) if prec > 0 or rec > 0 else 0


class GAImpl:
    def __init__(self, backends, thresholdsDict, cache):
        self.backends = backends
        self.thresholdsDict = thresholdsDict
        self.thresholdSizes = {b: len(thresholdsDict[b]) for b in backends}
        self.fitnessUtils = FitnessUtils(backends, cache)
        self.initGA()

    def initGA(self):
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)
        self.toolbox = base.Toolbox()
        self.registerAttributes()
        self.registerIndividual()
        self.registerPop()
        self.registerMateMutateAndSelect()
        self.registerEvaluate()

    def registerAttributes(self):
        for b in self.backends:
            self.toolbox.register(b, lambda x, y: 0, 0, self.thresholdSizes[b] - 1)

    def registerIndividual(self):
        attrs = (self.toolbox.__dict__[b] for b in self.backends)
        self.toolbox.register("individual",
                              tools.initCycle,
                              creator.Individual,
                              tuple(attrs))

    def registerPop(self):
        self.toolbox.register("population",
                              tools.initRepeat,
                              list,
                              self.toolbox.individual)

    def registerMateMutateAndSelect(self):
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate",
                              tools.mutUniformInt,
                              low=[0] * len(self.backends),
                              up=[self.thresholdSizes[b] - 1 for b in self.backends],
                              indpb=1.0 / len(self.backends))
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def registerEvaluate(self):
        self.toolbox.register("evaluate",
                              lambda ind: (self.fitnessUtils.computeFMeasure(ind),))

    def genoToPheno(self, ind):
        return [self.thresholdsDict[b][i] for i, b in zip(ind, self.backends)]

    def compute(self, popSize=100, maxGen=400, mutProb=0.2):
        self.toolbox.pop_size = popSize
        self.toolbox.max_gen = maxGen
        self.toolbox.mut_prob = mutProb
        pop = self.toolbox.population(n=self.toolbox.pop_size)
        pop = self.toolbox.select(pop, len(pop))
        res, _ = algorithms.eaMuPlusLambda(pop, self.toolbox, mu=self.toolbox.pop_size,
                                           lambda_=self.toolbox.pop_size,
                                           cxpb=1 - self.toolbox.mut_prob,
                                           mutpb=self.toolbox.mut_prob,
                                           stats=None,
                                           ngen=self.toolbox.max_gen,
                                           verbose=None)
        return [(self.genoToPheno(ind), self.fitnessUtils.computeFMeasure(ind), *self.fitnessUtils.computePrecRec(ind))
                for ind in res]


class GA:
    KMEANS = 1
    RANDOM = 2

    def __init__(self, traces, backends,
                 frontend, from_, to, mode=1, k=10):
        self.traces = traces
        self.backends = backends
        self.frontend = frontend
        self.from_ = from_
        self.to = to
        self.mode = mode
        self.k = k

    def getSelector(self):
        sel = None

        if self.mode == self.RANDOM:
            sel = RandSelector(self.traces,
                               self.backends,
                               self.frontend,
                               self.from_,
                               self.to)
        else:
            sel = KMeansSelector(self.traces,
                                 self.backends,
                                 self.frontend,
                                 self.from_,
                                 self.to)
        return sel

    def createCache(self, thresholdsDict):
        cacheMaker = CacheMaker(self.traces,
                                self.backends,
                                self.frontend,
                                self.from_,
                                self.to)
        return cacheMaker.create(thresholdsDict)

    def compute(self):
        sel = self.getSelector()
        thresholdsDict = sel.select(self.k)
        cache = self.createCache(thresholdsDict)
        ga = GAImpl(self.backends, thresholdsDict, cache)
        pheno, fmeasure, prec, rec = max(ga.compute(), key=lambda x: x[1])
        return (pheno,
                fmeasure,
                prec,
                rec)
