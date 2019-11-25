import copy

from deap import creator, base, tools, algorithms

from ylatency.graoperators import Operator
from ylatency.grautils import FitnessUtils
from ylatency.thresholds import Hashtable


class GeneticRangeAnalysis:
    def __init__(self, traces, target_col, thresholds_dict, min_, max_):
        columns = list(thresholds_dict.keys())
        ht = Hashtable(traces, columns, target_col, min_, max_)
        self._ops = Operator(thresholds_dict)
        self._fitness = FitnessUtils(columns, ht.positives(thresholds_dict), ht.negatives(thresholds_dict))

    def _initga(self, toolbox):
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)

        toolbox.register("individual", tools.initIterate, creator.Individual, self._ops.expllist)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", self._ops.cx)
        toolbox.register("mutate", self._ops.mut)
        toolbox.register("select", tools.selTournament, tournsize=20)
        toolbox.register("evaluate", lambda ind: (self._fitness.harmonic_mean(ind),))

    def explain(self, npop=100, ngen=400, cp=0.2, mut=0.2, stats=False):
        toolbox = base.Toolbox()
        self._initga(toolbox)
        self._ops.mut_prob = mut

        if stats:
            stats = tools.Statistics()
            stats.register("pop", copy.deepcopy)
        else:
            stats = None

        pop = toolbox.population(n=npop)
        res, logbook = algorithms.eaSimple(pop, toolbox,
                                           cxpb=cp, mutpb=mut, ngen=ngen,
                                           stats=stats, verbose=False)
        return res, logbook