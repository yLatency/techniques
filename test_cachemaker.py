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
        self.traces = self.spark.createDataFrame([('1', 1000, 3000), ('2', 2000, 3000)], ['traceId', self.backend, self.frontend])

    def test_tp_zerozero(self):
        cm = CacheMaker(self.traces, [self.backend], self.frontend, 2000, 4000)
        threshold = 3000
        bitstring = cm.create_bitstr_tp(self.backend, threshold)
        self.assertEqual('00', bitstring)

    def test_tp_zeroone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 4000)
        threshold = 1500
        bitstring = cm.create_bitstr_tp(self.backend, threshold)
        self.assertEqual('01', bitstring)

    def test_tp_oneone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 4000)
        threshold = 1000
        bitstring = cm.create_bitstr_tp(self.backend, threshold)
        self.assertEqual('11', bitstring)

    def test_fp_zerozero(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 2500)
        threshold = 3000
        bitstring = cm.create_bitstr_fp(self.backend, threshold)
        self.assertEqual('00', bitstring)

    def test_fp_zeroone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 2500)
        threshold = 1500
        bitstring = cm.create_bitstr_fp(self.backend, threshold)
        self.assertEqual('01', bitstring)

    def test_fp_oneone(self):
        cm = CacheMaker(self.traces, self.backend, self.frontend, 2000, 2500)
        threshold = 1000
        bitstring = cm.create_bitstr_fp(self.backend, threshold)
        self.assertEqual('11', bitstring)

    def test_create_tp(self):
        cm = CacheMaker(self.traces, [self.backend], self.frontend, 2000, 4000)
        tzerozero= 3000
        tzeroone = 1500
        toneone = 1000
        thresholds = [tzerozero, tzeroone, toneone]

        zerozero = int('00', 2)
        zeroone = int('01', 2)
        oneone = int('11', 2)
        expected =[zerozero, zeroone, oneone]

        list_tp = cm.create_tp(self.backend, thresholds)

        self.assertEqual(expected, list_tp)

    def test_create_fp(self):
        cm = CacheMaker(self.traces, [self.backend], self.frontend,  2000, 2500)
        tzerozero= 3000
        tzeroone = 1500
        toneone = 1000
        thresholds = [tzerozero, tzeroone, toneone]

        zerozero = int('00', 2)
        zeroone = int('01', 2)
        oneone = int('11', 2)
        expected =[zerozero, zeroone, oneone]

        list_tp = cm.create_fp(self.backend, thresholds)

        self.assertEqual(expected, list_tp)


    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


