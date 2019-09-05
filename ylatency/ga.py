import random
from functools import reduce
from pyspark.sql.functions import col
from deap import base, creator, tools, algorithms
import copy


class CacheMaker:
    def __init__(self, traces, backends,
                 frontend, from_, to):
        self.traces = traces
        self.frontend = frontend
        self.backends = backends
        self.from_ = from_
        self.to = to

    def create(self, thr_dict):
        pos = self.get_positives().count()
        cache = {'p': pos,
                 'n': self.traces.count() - pos}

        for b in self.backends:
            tp_intlist = self.create_tp(b, thr_dict[b])
            fp_intlist = self.create_fp(b, thr_dict[b])
            size = len(thr_dict[b])
            for i in range(size):
                cache[b, i] = tp_intlist[i], fp_intlist[i]

        return cache

    def get_positives(self):
        return (self.traces.filter((col(self.frontend) > self.from_) &
                                   (col(self.frontend) <= self.to)))

    def create_tp(self, backend, thresholds):
        pos = self.get_positives()
        return self.create_bitslists(pos, backend, thresholds)

    def create_bitslists(self, filtered_traces, backend, thresholds):
        sorted_traces = filtered_traces.sort('traceId')
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
        self.p = cache['p']
        self.n = cache['n']

    def countOnesInConjunctedBitStrings(self, ind, getter):
        bit = reduce(lambda bx, by: bx & by,
                     map(getter, ind))
        return bin(bit).count("1")

    def getTPBitString(self, backend_idx, threshold):
        backend = self.backends[backend_idx]
        return self.cache[backend, threshold][0]

    def getFPBitString(self, backend_idx, threshold):
        backend = self.backends[backend_idx]
        return self.cache[backend, threshold][1]

    def computeTP(self, ind):
        tp = None
        if len(ind) == 0:
            tp = self.p
        else:
            getter = lambda bft: self.getTPBitString(bft[0], bft[1]) & ~ self.getTPBitString(bft[0], bft[2])
            tp = self.countOnesInConjunctedBitStrings(ind, getter)
        return tp

    def computeFP(self, ind):
        fp = None
        if len(ind) == 0:
            fp = self.n
        else:
            getter = lambda bft: self.getFPBitString(bft[0], bft[1]) & ~ self.getFPBitString(bft[0], bft[2])
            fp = self.countOnesInConjunctedBitStrings(ind, getter)
        return fp

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
        creator.create("Individual", set, fitness=creator.Fitness)
        self.toolbox = base.Toolbox()
        self.registerAttributes()
        self.registerIndividual()
        self.registerPop()
        self.registerMateMutateAndSelect()
        self.registerEvaluate()

    def rdm_interval(self, thresholds):
        indexes = [i for i, _ in enumerate(thresholds)]
        from_, to = sorted(random.sample(indexes, k=2))
        return from_, to


    def rdm_cond(self):
        cond = None
        while cond is None:
            indexes = [i for i, _ in enumerate(self.backends)]
            bi = random.choice(indexes)
            b = self.backends[bi]
            thresholds = self.thresholdsDict[b]
            if len(thresholds) > 2:
                from_, to = self.rdm_interval(thresholds)
                cond = (bi, from_, to)
        return cond

    def cx(self, ind1, ind2):
        ind1 |= ind2
        ind2 |= ind1
        size = len(ind1)
        if size > 0:
            chosen = random.sample(ind2, k=random.randint(1, size))
            ind1 ^= set(chosen)
            ind2 ^= ind1
        return ind1, ind2

    def mut(self, individual):
        mutkind = random.randrange(3)
        if mutkind == 0:
            self.mutremove(individual)
        elif mutkind == 1:
            self.mutadd(individual)

        elif mutkind == 2:
            self.mutmodify(individual)

        return individual,

    def mutadd(self, individual):
        newcond = self.rdm_cond()
        exist = [cond for cond in individual if cond[0] == newcond[0]]
        if not exist:
            individual.add(newcond)

    def mutremove(self, individual):
        if len(individual) > 0:
            rdmcond = random.sample(individual, 1)[0]
            individual.remove(rdmcond)

    def mutmodify(self, individual):
        if len(individual) > 0:
            rdmcond = random.sample(individual, 1)[0]
            interval = list(rdmcond[1:])
            b = self.backends[rdmcond[0]]
            thrslen = len(self.thresholdsDict[b])
            interval[random.randrange(2)] = random.randrange(thrslen)
            if interval[0] != interval[1]:
                individual.remove(rdmcond)
                newcond = (rdmcond[0], *sorted(interval))
                individual.add(newcond)

    def feasible(self, individual):
        for bi, fi, ti in individual:
            b = self.backends[bi]
            thresholds = self.thresholdsDict[b]
            if fi == 0 and ti == len(thresholds)-1:
                return False

        return True

    def registerAttributes(self):
        self.toolbox.register("attribute", self.rdm_cond)

    def registerIndividual(self):
        SIZE_EXPL = 2
        self.toolbox.register("individual",
                              tools.initRepeat,
                              creator.Individual,
                              self.toolbox.attribute,
                              SIZE_EXPL)

    def registerPop(self):
        self.toolbox.register("population",
                              tools.initRepeat,
                              list,
                              self.toolbox.individual)

    def registerMateMutateAndSelect(self):
        self.toolbox.register("mate", self.cx)
        self.toolbox.register("mutate", self.mut)
        self.toolbox.register("select", tools.selTournament, tournsize=20)

    def registerEvaluate(self):
        evaluate = lambda ind: (self.fitnessUtils.computeFMeasure(ind),)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.decorate("evaluate", tools.DeltaPenalty(self.feasible, 0.0))

    def genoToPheno(self, ind):
        pheno = set()
        for bi, fi, ti in ind:
            b = self.backends[bi]
            from_ = self.thresholdsDict[b][fi]
            to = self.thresholdsDict[b][ti]
            pheno.add((b, from_, to))
        return pheno

    def compute(self, popSize=100, maxGen=400, mutProb=0.2, stats=False):
        if stats:
            stats = tools.Statistics()
            stats.register("pop", copy.deepcopy)
        else:
            stats = None

        self.toolbox.pop_size = popSize
        self.toolbox.max_gen = maxGen
        self.toolbox.mut_prob = mutProb
        pop = self.toolbox.population(n=self.toolbox.pop_size)
        pop = self.toolbox.select(pop, len(pop))
        res, logbook = algorithms.eaMuPlusLambda(pop, self.toolbox, mu=self.toolbox.pop_size,
                                           lambda_=self.toolbox.pop_size,
                                           cxpb=1 - self.toolbox.mut_prob,
                                           mutpb=self.toolbox.mut_prob,
                                           stats=stats,
                                           ngen=self.toolbox.max_gen,
                                           verbose=None)
        return res, logbook

class GA:

    def __init__(self, traces, backends,
                 frontend, thresholds_dict):
        self.traces = traces
        self.backends = backends
        self.frontend = frontend
        self.thresholds_dict = thresholds_dict

    def createCache(self, from_, to):
        cacheMaker = CacheMaker(self.traces,
                                self.backends,
                                self.frontend,
                                from_,
                                to)
        return cacheMaker.create(self.thresholds_dict)

    def compute(self, from_, to):
        cache = self.createCache(from_, to)
        ga = GAImpl(self.backends, self.thresholds_dict, cache)
        res, _ = ga.compute()
        parsed_res = [(ga.genoToPheno(ind),
                       ga.fitnessUtils.computeFMeasure(ind),
                       *ga.fitnessUtils.computePrecRec(ind))
                      for ind in res]

        pheno, fmeasure, prec, rec = max(parsed_res, key=lambda x: (x[1], -len(x[0])))
        return (pheno,
                fmeasure,
                prec,
                rec)
