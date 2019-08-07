from unittest import TestCase

from pyspark.sql import SparkSession

from ks.branchandbound import BranchAndBound
from tests.enumeration import Enumeration

class TestBranchAndBound(TestCase):
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

    def test_best_solution(self):
        frontendSLA = self.traces.approxQuantile([self.frontend], [0.5], 0)[0][0]
        maxlat = self.traces.select(self.frontend).rdd.max()[0]
        thresholds = [qs[0] for qs in self.traces.approxQuantile(self.backends, [0.5], 0)]


        bestExpBnB = BranchAndBound(self.traces, self.backends, thresholds,
                                   self.frontend, frontendSLA, maxlat).compute()

        bestExpEnum = Enumeration(self.traces, self.backends, thresholds,
                                 self.frontend,  frontendSLA, maxlat).compute()

        self.assertEqual(bestExpEnum.fmeasure, bestExpBnB.fmeasure)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
