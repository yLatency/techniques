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
        traces = self.spark.read.parquet('dummy.parquet')
        frontend = 'web-service_HomeControllerHome_avg_dur'
        from_ = traces.approxQuantile([frontend], [0.5], 0)[0][0]
        to = traces.select(frontend).rdd.max()[0]

        self.backends = [c for c in traces.columns
                         if c != 'traceId' and c.endswith('avg_self_dur')]
        self.thresholds = [qs[0] for qs in traces.approxQuantile(self.backends, [0.5], 0)]

        self.metrics = Metrics(traces, self.backends, self.thresholds,
                               frontend, from_, to)

        self.pos_traces = traces.filter((col(frontend) > from_) & (col(frontend) <= to))
        self.neg_traces = traces.filter((col(frontend) <= from_) | (col(frontend) > to))

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
            backend_comb = {b for b, t in bt_comb}
            fmeasure = self.compute_fmeasure(bt_comb)

            self.assertEqual(fmeasure, self.metrics.compute(backend_comb)[0])

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
