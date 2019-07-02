from unittest import TestCase

from pyspark.sql import SparkSession
from ga import CacheMaker, FitnessUtils


class TestFitnessUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                                 .master('local[2]')
                                 .appName('my-local-testing-pyspark-context')
                                 .getOrCreate())


    def setUp(self):
        self.backends = ['b1', 'b2']
        self.frontend = 'f'
        self.traces = self.spark.createDataFrame([(1000, 2000, 3000),
                                                  (2000, 3000, 4000)],
                                                   self.backends + [self.frontend])
        # first trace is negative, second one is positive
        self.cachemaker = CacheMaker(self.traces,
                                     self.backends,
                                     self.frontend,
                                     3500,
                                     4001)

    def test_compute_tp_zero(self):
        thr_dict = {'b1': [0], # select one positive
                    'b2': [3001] # select zero positives
                    }
        cache = self.cachemaker.create(thr_dict)
        ind = [0, 0]
        fu = FitnessUtils(self.backends, cache)
        val = fu.computeTP(ind)
        self.assertEqual(0, val)



    def test_compute_tp_one(self):
        thr_dict = {'b1': [0], # select one positive
                    'b2': [3000] # select one positives
                    }
        cache = self.cachemaker.create(thr_dict)
        ind = [0, 0]
        fu = FitnessUtils(self.backends, cache)
        val = fu.computeTP(ind)
        self.assertEqual(1, val)


    def test_compute_fp_zero(self):
        thr_dict = {'b1': [0], # select one negative
                    'b2': [2500] # select zero negatives
                    }
        cache = self.cachemaker.create(thr_dict)
        ind = [0, 0]
        fu = FitnessUtils(self.backends, cache)
        val = fu.computeFP(ind)
        self.assertEqual(0, val)



    def test_compute_fp_one(self):
        thr_dict = {'b1': [0], # select one negative
                    'b2': [2000] # select one negatives
                    }
        cache = self.cachemaker.create(thr_dict)
        ind = [0, 0]
        fu = FitnessUtils(self.backends, cache)
        val = fu.computeFP(ind)
        self.assertEqual(1, val)


    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
