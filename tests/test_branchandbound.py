from unittest import TestCase

from pyspark.sql import SparkSession

from ks.branchandbound import BranchAndBound

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
        self.quantiles = [0.0, 0.8]
        self.thresholds_dict = self.create_thr_dict()

    def create_thr_dict(self):
        thresholds_lists = self.traces.approxQuantile(self.backends, self.quantiles, 0)
        thr_dict = {}
        for b, thresholds in zip(self.backends, thresholds_lists):
            max_ = self.traces.select(b).rdd.max()[0] + 1
            thr_dict[b] = thresholds + [max_]

        return thr_dict


    def test_createFeatures(self):
        thr_dict = {'b1': [0, 1, 2], 'b2': [0, 1, 2]}
        expected = {('b1', 0, 1), ('b1', 1, 2),
                    ('b2', 0, 1), ('b2', 1, 2)}

        actual = BranchAndBound.createFeatures(thr_dict)
        self.assertEqual(expected, actual)


    def test_best_solution(self):
        frontendSLA = self.traces.approxQuantile([self.frontend], [0.5], 0)[0][0]
        maxlat = self.traces.select(self.frontend).rdd.max()[0]


        bestExpBnB = BranchAndBound(self.traces, self.frontend, self.thresholds_dict).compute(frontendSLA, maxlat)

        print(bestExpBnB)


    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
