from unittest import TestCase

from pyspark.sql import SparkSession

from ylatency.gra import GeneticRangeAnalysis
from ylatency.thresholds import MSSelector

class TestGeneticRangeAnalysis(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                                 .master('local[2]')
                                 .appName('my-local-testing-pyspark-context')
                                 .getOrCreate())

    def setUp(self):
        self.traces = self.spark.read.parquet('dummy.parquet')
        self.target_col = 'web-service_HomeControllerHome_avg_dur'
        columns = [c for c in self.traces.columns if c != 'traceId' and c.endswith('avg_self_dur')]
        self.thresholds_dict = MSSelector(self.traces).select_foreach(columns)
        self.min = self.traces.approxQuantile([self.target_col], [0.5], 0)[0][0]
        self.max = self.traces.select(self.target_col).rdd.max()[0]

    def _create_thr_dict(self, quantiles, columns):
        thresholds_lists = self.traces.approxQuantile(columns, quantiles, 0)
        thr_dict = {}
        for b, thresholds in zip(columns, thresholds_lists):
            max_ = self.traces.select(b).rdd.max()[0] + 1
            thr_dict[b] = thresholds + [max_]

        return thr_dict

    def test_explain(self):
        gra = GeneticRangeAnalysis(self.traces, self.target_col, self.thresholds_dict,
                                   self.min, self.max)

        print(gra.explain())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
