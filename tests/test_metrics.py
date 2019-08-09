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
        self.frontend = 'web-service_HomeControllerHome_avg_dur'
        self.backends = [c for c in self.traces.columns
                         if c != 'traceId' and c.endswith('avg_self_dur')]
        self.from_ = self.traces.approxQuantile([self.frontend], [0.5], 0)[0][0]
        self.to = self.traces.select(self.frontend).rdd.max()[0]
        self.thresholds = [qs[0] for qs in self.traces.approxQuantile(self.backends, [0.5], 0)]

        self.metrics = Metrics(self.traces, self.backends, self.thresholds,
                               self.frontend, self.from_, self.to)

        self.pos_traces = self.traces.filter((col(self.frontend) > self.from_) & (col(self.frontend) <= self.to))
        self.neg_traces = self.traces.filter((col(self.frontend) <= self.from_) | (col(self.frontend) > self.to))

    def compute_fmeasure(self, bt_comb):
        tp = reduce(lambda df, bt: df.filter(col(bt[0]) >= bt[1]),
                    bt_comb,
                    self.pos_traces).count()

        fp = reduce(lambda df, bt: df.filter(col(bt[0]) >= bt[1]),
                    bt_comb,
                    self.neg_traces).count()

        rec = tp / self.pos_traces.count()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0

        return 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0

    def test_compute(self):
        random.seed(10)
        bt_all = list(zip(self.backends, self.thresholds))
        for _ in range(10):
            bt_comb = random.choices(bt_all, k=3)
            backend_comb = {b for b,t in bt_comb}
            fmeasure = self.compute_fmeasure(bt_comb)

            self.assertEqual(fmeasure, self.metrics.compute(backend_comb)[0])

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
