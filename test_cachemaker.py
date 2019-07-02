from unittest import TestCase
from pyspark.sql import SparkSession
from ga import CacheMaker


class TestCacheMaker(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                                 .master('local[2]')
                                 .appName('my-local-testing-pyspark-context')
                                 .getOrCreate())


    def setUp(self):
        self.backend = 'b'
        self.frontend = 'f'
        self.traces = self.spark.createDataFrame([(1000, 3000), (2000, 3000)], [self.backend, self.frontend])

    def test_tp_zerozero(self):
        cm = CacheMaker(self.traces, [self.backend], self.frontend, 2000, 4000)
        threshold = 3000
        bitstring = cm.create_bitstring_tp(self.backend, threshold)
        self.assertEqual('00', bitstring)

    def test_tp_zeroone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 4000)
        threshold = 1500
        bitstring = cm.create_bitstring_tp(self.backend, threshold)
        self.assertEqual('01', bitstring)

    def test_tp_oneone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 4000)
        threshold = 1000
        bitstring = cm.create_bitstring_tp(self.backend, threshold)
        self.assertEqual('11', bitstring)

    def test_fp_zerozero(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 2500)
        threshold = 3000
        bitstring = cm.create_bitstring_fp(self.backend, threshold)
        self.assertEqual('00', bitstring)

    def test_fp_zeroone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 2500)
        threshold = 1500
        bitstring = cm.create_bitstring_fp(self.backend, threshold)
        self.assertEqual('01', bitstring)

    def test_fp_oneone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 2500)
        threshold = 1000
        bitstring = cm.create_bitstring_fp(self.backend, threshold)
        self.assertEqual('11', bitstring)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


