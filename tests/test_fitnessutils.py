from operator import itemgetter
from unittest import TestCase

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from ylatency.gautils import FitnessUtils
from ylatency.thresholds import CacheMaker
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
        self.quantiles = [0.5]

        self.backends = [c for c in self.traces.columns
                         if c.endswith('avg_self_dur')]

        self.pos_traces = self.traces.filter((col(self.frontend) > from_) & (col(self.frontend) <= to))
        self.neg_traces = self.traces.filter((col(self.frontend) <= from_) | (col(self.frontend) > to))

        self.thr_dict = self.create_thr_dict()
        cache = self.create_cache()
        self.fu = FitnessUtils(self.backends, cache)

    def create_thr_dict(self):
        thresholds_lists = self.traces.approxQuantile(self.backends, self.quantiles, 0)
        thr_dict = {}
        for b, thresholds in zip(self.backends, thresholds_lists):
            min_ = self.traces.select(b).rdd.min()[0]
            max_ = self.traces.select(b).rdd.max()[0] + 1
            thr_dict[b] = [min_]
            thr_dict[b] += thresholds
            thr_dict[b] += [max_]

        return thr_dict

    def create_cache(self):
        cache_maker = CacheMaker(self.traces,
                                 self.backends,
                                 self.frontend,
                                 *self.interval)

        return cache_maker.create(self.thr_dict)

    def decode_ind(self, ind):
        decoded = set()
        for bi, fi, ti in ind:
            b = self.backends[bi]
            f = self.thr_dict[b][fi]
            t = self.thr_dict[b][ti]
            decoded.add((b, f, t))
        return decoded


    def filter_by_ind(self, traces, ind):
        decoded = self.decode_ind(ind)

        filterbyintervals = (lambda df, bft: df.filter(col(bft[0]) >= bft[1])
                                               .filter(col(bft[0]) < bft[2]))
        return reduce(filterbyintervals,
                      decoded,
                      traces)

    def create_ind(self):
        return {(0, 1, 2), (2, 1, 2)}

    def test_compute_tp(self):

        ind = self.create_ind()

        tp = self.filter_by_ind(self.pos_traces, ind)

        val = self.fu.computeTP(ind)

        self.assertEqual(tp.count(), val)

    def test_compute_fp(self):
        ind = self.create_ind()

        fp = self.filter_by_ind(self.neg_traces, ind)

        val = self.fu.computeFP(ind)

        self.assertEqual(fp.count(), val)


    def test_compute_prec_rec(self):
        ind = self.create_ind()

        tp = self.filter_by_ind(self.pos_traces, ind)
        fp = self.filter_by_ind(self.neg_traces, ind)
        exp_rec = tp.count() / self.pos_traces.count()
        exp_prec = tp.count() / (tp.count() + fp.count())

        act_prec, act_rec = self.fu.computePrecRec(ind)

        self.assertEqual(exp_prec, act_prec)
        self.assertEqual(exp_rec, act_rec)


    def test_compute_fmeasure(self):
        ind = self.create_ind()

        tp = self.filter_by_ind(self.pos_traces, ind)
        fp = self.filter_by_ind(self.neg_traces, ind)
        rec = tp.count() / self.pos_traces.count()
        prec = tp.count() / (tp.count() + fp.count())
        exp_fmeasure = 2 * (prec * rec) / (prec + rec)


        self.assertEqual(exp_fmeasure, self.fu.computeFMeasure(ind))

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
