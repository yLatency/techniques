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
        creator.create("Individual", list, fitness=creator.Fitness)
        toolbox.register("individual", tools.initIterate, creator.Individual, self._ops.expllist)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", self._ops.cx)
        toolbox.register("mutate", self._ops.mut)

    def explain(self, mu=100, lambda_=100, ngen=400, cp=0.6, mut=0.4, stats=False):
        toolbox = base.Toolbox()
        creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, -1.0))
        self._initga(toolbox)
        toolbox.register("select", tools.selNSGA2)
        f = self._fitness
        toolbox.register("evaluate", lambda ind: (f.fscore(ind), f.dissimilarity(ind), f.numclusters(ind)))
        toolbox.decorate("evaluate", tools.DeltaPenalty(f.feasible, (0, float('inf'), float('inf'))))

        self._ops.mut_prob = mut

        if stats:
            stats = tools.Statistics()
            stats.register("pop", copy.deepcopy)
        else:
            stats = None

        pop = toolbox.population(n=ngen)
        res, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                                 mu=mu,
                                                 lambda_=lambda_,
                                                 cxpb=cp, mutpb=mut, ngen=ngen,
                                                 stats=stats, verbose=False)
        return res, logbook