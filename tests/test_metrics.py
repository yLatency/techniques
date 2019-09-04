from unittest import TestCase

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from ks.metrics import Metrics
from functools import reduce

import random


class TestMetrics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                     .master('local[2]')
                     .appName('my-local-testing-pyspark-context')
                     .getOrCreate())

    def setUp(self):
        self.traces = self.spark.read.parquet('dummy.parquet')
        frontend = 'web-service_HomeControllerHome_avg_dur'
        from_ = self.traces.approxQuantile([frontend], [0.5], 0)[0][0]
        to = self.traces.select(frontend).rdd.max()[0]

        self.backends = [c for c in self.traces.columns
                         if c != 'traceId' and c.endswith('avg_self_dur')]

        self.quantiles = [0.0, 0.4, 0.6, 0.8]

        self.thresholds_dict = self.create_thr_dict()

        self.metrics = Metrics(self.traces, self.thresholds_dict,
                               frontend, from_, to)

        self.pos_traces = self.traces.filter((col(frontend) > from_) & (col(frontend) <= to))
        self.neg_traces = self.traces.filter((col(frontend) <= from_) | (col(frontend) > to))

    def create_thr_dict(self):
        thresholds_lists = self.traces.approxQuantile(self.backends, self.quantiles, 0)
        thr_dict = {}
        for b, thresholds in zip(self.backends, thresholds_lists):
            max_ = self.traces.select(b).rdd.max()[0] + 1
            thr_dict[b] = thresholds + [max_]

        return thr_dict

    def compute_fmeasure(self, expl):
        reducefun = (lambda df, bft: df.filter(col(bft[0]) >= bft[1])
                                       .filter(col(bft[0]) < bft[2]))
        tp = reduce(reducefun,
                    expl,
                    self.pos_traces).count()

        fp = reduce(reducefun,
                    expl,
                    self.neg_traces).count()

        rec = tp / self.pos_traces.count()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0

        return 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0

    def test_compute(self):
        random.seed(10)
        for _ in range(10):
            backends = random.sample(self.backends, k=3)
            expl = set()
            for b in backends:
                interval = sorted(random.sample(self.thresholds_dict[b], k=2))
                expl.add((b, *interval))

            fm = self.compute_fmeasure(expl)
            self.assertEqual(fm, self.metrics.compute(expl)[0])

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
