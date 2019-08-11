from unittest import TestCase

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from ylatency.ga import CacheMaker, FitnessUtils
import random
from functools import reduce


class TestFitnessUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                     .master('local[2]')
                     .appName('my-local-testing-pyspark-context')
                     .getOrCreate())

    def setUp(self):
        self.traces = self.spark.read.parquet('dummy.parquet')
        self.frontend = 'web-service_HomeControllerHome_avg_dur'
        from_ = self.traces.approxQuantile([self.frontend], [0.5], 0)[0][0]
        to = self.traces.select(self.frontend).rdd.max()[0]
        self.interval = (from_, to)
        self.quantiles = [0.0, 0.4, 0.6, 0.8]

        self.backends = [c for c in self.traces.columns
                         if c.endswith('avg_self_dur')]

        self.pos_traces = self.traces.filter((col(self.frontend) > from_) & (col(self.frontend) <= to))
        self.neg_traces = self.traces.filter((col(self.frontend) <= from_) | (col(self.frontend) > to))

        self.thr_dict = self.create_thr_dict()
        cache = self.create_cache()
        self.fu = FitnessUtils(self.backends, cache)

    def create_thr_dict(self):
        thresholds_lists = self.traces.approxQuantile(self.backends, self.quantiles, 0)
        return {b: thresholds
                for b, thresholds in zip(self.backends, thresholds_lists)}

    def create_cache(self):
        cache_maker = CacheMaker(self.traces,
                                 self.backends,
                                 self.frontend,
                                 *self.interval)

        return cache_maker.create(self.thr_dict)

    def filter_by_ind(self, traces, ind):
        sol = [(b, self.thr_dict[b][i])
               for b, i in zip(self.backends, ind)]

        return reduce(lambda df, bt: df.filter(col(bt[0]) >= bt[1]),
                      sol,
                      traces)

    def test_compute_tp(self):
        random.seed(10)
        n = len(self.quantiles)
        ind = [random.randrange(n)
               for _ in self.backends]

        tp = self.filter_by_ind(self.pos_traces, ind)

        val = self.fu.computeTP(ind)

        self.assertEqual(tp.count(), val)

    def test_compute_fp(self):
        random.seed(10)
        n = len(self.quantiles)
        ind = [random.randrange(n)
               for _ in self.backends]

        fp = self.filter_by_ind(self.neg_traces, ind)

        val = self.fu.computeFP(ind)

        self.assertEqual(fp.count(), val)


    def test_compute_prec_rec(self):
        random.seed(10)
        n = len(self.quantiles)
        ind = [random.randrange(n)
               for _ in self.backends]

        tp = self.filter_by_ind(self.pos_traces, ind)
        fp = self.filter_by_ind(self.neg_traces, ind)
        exp_rec = tp.count() / self.pos_traces.count()
        exp_prec = tp.count() / (tp.count() + fp.count())

        act_prec, act_rec = self.fu.computePrecRec(ind)

        self.assertEqual(exp_prec, act_prec)
        self.assertEqual(exp_rec, act_rec)


    def test_compute_fmeasure(self):
        random.seed(10)
        n = len(self.quantiles)
        ind = [random.randrange(n)
               for _ in self.backends]

        tp = self.filter_by_ind(self.pos_traces, ind)
        fp = self.filter_by_ind(self.neg_traces, ind)
        rec = tp.count() / self.pos_traces.count()
        prec = tp.count() / (tp.count() + fp.count())
        exp_fmeasure = 2 * (prec * rec) / (prec + rec)


        self.assertEqual(exp_fmeasure, self.fu.computeFMeasure(ind))

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
