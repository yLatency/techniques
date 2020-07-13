from unittest import TestCase

from pyspark.sql import SparkSession

from decaf.decaf import DeCaf

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
        self.columns = [c for c in self.traces.columns if c != 'traceId' and c.endswith('avg_self_dur')]
        self.sla = self.traces.approxQuantile([self.target_col], [0.2], 0)[0][0]

    def test_explain(self):
        decaf = DeCaf(self.traces, self.target_col, self.columns, self.sla)

        print(decaf.explain())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
