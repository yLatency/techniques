from unittest import TestCase

from pyspark.sql import SparkSession

from ylatency.ga import GA
from ks.branchandbound import BranchAndBound
from ks.range_analysis import RangeAnalysis

class TestRangeAnalysis(TestCase):
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
        self.splitpoints = self.traces.approxQuantile([self.frontend], [0.5, 0.7, 1.0], 0)[0]

    def create_thr_dict(self):
        thresholds_lists = self.traces.approxQuantile(self.backends, self.quantiles, 0)
        thr_dict = {}
        for b, thresholds in zip(self.backends, thresholds_lists):
            max_ = self.traces.select(b).rdd.max()[0] + 1
            thr_dict[b] = thresholds + [max_]

        return thr_dict



    def test_compute_ga(self):


        ga = GA(self.traces, self.backends, self.frontend, self.thresholds_dict)

        ra = RangeAnalysis(ga.compute, self.splitpoints)

        selectedsplitpoints, fscores_sum, detailed_results = ra.explain()

        # returned selected split points do not consider the right endpoint of the interval
        self.assertLess(len(selectedsplitpoints), len(self.splitpoints))

        # sum of fscores
        self.assertIsInstance(fscores_sum, float)

        # test detailed results
        for res in detailed_results:
            # test set of conditions
            self.assertIsInstance(res[0], set)
            for col, from_, to in res[0]:
                self.assertIsInstance(col, str)
                self.assertIsInstance(from_, float)
                self.assertIsInstance(to, float)

            # fmeasure
            self.assertTrue(0 <= res[1] <= 1)

            # precision
            self.assertTrue(0 <= res[2] <= 1)

            # recall
            self.assertTrue(0 <= res[3] <= 1)

        print(selectedsplitpoints, fscores_sum)

    def test_compute_bnb(self):


        bnb = BranchAndBound(self.traces, self.frontend, self.thresholds_dict)

        ra = RangeAnalysis(bnb.compute, self.splitpoints)

        selectedsplitpoints, fscores_sum, detailed_results = ra.explain()

        # returned selected split points do not consider the right endpoint of the interval
        self.assertLess(len(selectedsplitpoints), len(self.splitpoints))

        # sum of fscores
        self.assertIsInstance(fscores_sum, float)

        # test detailed results
        for res in detailed_results:
            # test set of conditions
            self.assertIsInstance(res[0], frozenset)
            for col, from_, to in res[0]:
                self.assertIsInstance(col, str)
                self.assertIsInstance(from_, float)
                self.assertIsInstance(to, float)

            # fmeasure
            self.assertTrue(0 <= res[1] <= 1)

            # precision
            self.assertTrue(0 <= res[2] <= 1)

            # recall
            self.assertTrue(0 <= res[3] <= 1)

        print(selectedsplitpoints, fscores_sum)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
