from functools import reduce
from pyspark.sql.functions import col, when, collect_list, concat_ws
from deap import base, creator, tools, algorithms

from thresholds import Normalizer, RandSelector, KMeansSelector


class CacheMaker:
    def __init__(self, traces, backends,
                 frontend, from_, to):
        self.traces = traces
        self.frontend = frontend
        self.backends = backends
        self.from_ = from_
        self.to = to

    def tpCol(self, index):
        return 'tp-%d' % index

    def fpCol(self, index):
        return 'fp-%d' % index

    def withColsTpFp(self, df, enumThreshold, aBackend):
        index, threshold = enumThreshold
        aboveThresholdCond = col(aBackend) >= threshold
        tpCond = aboveThresholdCond & (col(self.frontend) > self.from_) & (col(self.frontend) <= self.to)
        fpCond = aboveThresholdCond & ((col(self.frontend) <= self.from_) | (col(self.frontend) > self.to))
        whenTrueOneElseZero = lambda cond: when(cond, '1').otherwise('0')
        return (df.withColumn(self.tpCol(index), whenTrueOneElseZero(tpCond))
                .withColumn(self.fpCol(index), whenTrueOneElseZero(fpCond)))

    def createDfWithColsTpFp(self, enumThresholds, aBackend):
        return reduce(lambda df, enumThr: self.withColsTpFp(df, enumThr, aBackend),
                      enumThresholds,
                      self.traces)

    def stringifyCol(self, column):
        colToList = collect_list(col(column))
        listToString = concat_ws("", colToList)
        return listToString.alias(column)

    def genStringifyCols(self, enumThresholds):
        for i, _ in enumThresholds:
            yield self.stringifyCol(self.tpCol(i))
            yield self.stringifyCol(self.fpCol(i))

    def addRowToCache(self, cache, row, aBackend):
        for i in range(len(row) // 2):
            tpBitString = row[self.tpCol(i)]
            fpBitString = row[self.fpCol(i)]
            cache[aBackend, i] = int(tpBitString, 2), int(fpBitString, 2)

    def createBitStringsRow(self, thresholdsDict, aBackend):
        enumThresholds = list(enumerate(thresholdsDict[aBackend]))
        df = self.createDfWithColsTpFp(enumThresholds, aBackend)
        stringifyCols = list(self.genStringifyCols(enumThresholds))
        return (df.groupBy()
                .agg(*stringifyCols)
                .collect()[0])

    def create(self, thresholdsDict):
        cache = {}
        for aBackend in self.backends:
            row = self.createBitStringsRow(thresholdsDict, aBackend)
            self.addRowToCache(cache, row, aBackend)
        return cache


class FitnessUtils:
    def __init__(self, backends, cache):
        self.backends = backends
        self.cache = cache
        self.p = bin(self.getTPBitString(self.backends[0], 0)).count('1')
        if self.p <= 0:
            raise Exception('No positives')

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
        normalizer = Normalizer(backends, traces)
        self.normalizedTrace = normalizer.createNormalizedTrace()
        self.backends = backends
        self.frontend = frontend
        self.from_ = from_
        self.to = to
        self.mode = mode
        self.k = k

    def getSelector(self):
        sel = None

        if self.mode == self.RANDOM:
            sel = RandSelector(self.normalizedTrace,
                               self.backends,
                               self.frontend,
                               self.from_,
                               self.to)
        else:
            sel = KMeansSelector(self.normalizedTrace,
                                 self.backends,
                                 self.frontend,
                                 self.from_,
                                 self.to)
        return sel

    def createCache(self, thresholdsDict):
        cacheMaker = CacheMaker(self.normalizedTrace,
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
        return max(ga.compute(), key=lambda x: x[1])
