from unittest import TestCase
from ylatency.grautils import FitnessUtils


class TestFitnessUtils(TestCase):
    def setUp(self):
        pass

    def test_cardinality_zero(self):
        bitstring = int('000', base=2)
        res = FitnessUtils._cardinality(bitstring)
        self.assertEqual(0, res)

    def test_cardinality_morethanzero(self):
        bitstring = int('100', base=2)
        res = FitnessUtils._cardinality(bitstring)
        self.assertEqual(1, res)

    def test_recall_zero(self):
        tp1 = int('0000', base=2)
        tp2 = int('0000', base=2)
        tp3 = int('0000', base=2)
        tp4 = int('0000', base=2)
        tplist = [tp1, tp2, tp3, tp4]
        num_pos = 4

        res = FitnessUtils._recall(tplist, num_pos)
        self.assertEqual(0, res)

    def test_recall_half(self):
        tp1 = int('1000', base=2)
        tp2 = int('1000', base=2)
        tp3 = int('0100', base=2)
        tp4 = int('0000', base=2)
        tplist = [tp1, tp2, tp3, tp4]
        num_pos = 4

        res = FitnessUtils._recall(tplist, num_pos)
        self.assertEqual(0.5, res)

    def test_recall_one(self):
        tp1 = int('1000', base=2)
        tp2 = int('0100', base=2)
        tp3 = int('0010', base=2)
        tp4 = int('0001', base=2)
        tplist = [tp1, tp2, tp3, tp4]
        num_pos = 4

        res = FitnessUtils._recall(tplist, num_pos)
        self.assertEqual(1, res)

    def test_precision_zero(self):
        tp1 = int('00', base=2)
        tp2 = int('00', base=2)

        fp1 = int('10', base=2)
        fp2 = int('01', base=2)

        tplist = [tp1, tp2]
        fplist = [fp1, fp2]

        res = FitnessUtils._precision(tplist, fplist)
        self.assertEqual(0, res)

    def test_precision_half(self):
        tp1 = int('10', base=2)
        tp2 = int('01', base=2)

        fp1 = int('10', base=2)
        fp2 = int('01', base=2)

        tplist = [tp1, tp2]
        fplist = [fp1, fp2]

        res = FitnessUtils._precision(tplist, fplist)
        self.assertEqual(0.5, res)

    def test_precision_one(self):
        tp1 = int('10', base=2)
        tp2 = int('01', base=2)

        fp1 = int('00', base=2)
        fp2 = int('00', base=2)

        tplist = [tp1, tp2]
        fplist = [fp1, fp2]

        res = FitnessUtils._precision(tplist, fplist)
        self.assertEqual(1, res)

    def test_disjointess_zero(self):
        tp1 = int('10', base=2)
        tp2 = int('10', base=2)
        tp3 = int('10', base=2)

        fp1 = int('10', base=2)
        fp2 = int('10', base=2)
        fp3 = int('10', base=2)

        tplist = [tp1, tp2, tp3]
        fplist = [fp1, fp2, fp3]

        res = FitnessUtils._disjointness(tplist, fplist)
        self.assertEqual(0, res)

    def test_disjointess_half(self):
        tp1 = int('10', base=2)
        tp2 = int('10', base=2)

        fp1 = int('10', base=2)
        fp2 = int('01', base=2)

        tplist = [tp1, tp2]
        fplist = [fp1, fp2]

        res = FitnessUtils._disjointness(tplist, fplist)
        self.assertEqual(0.5, res)

    def test_disjointess_one(self):
        tp1 = int('10', base=2)
        tp2 = int('01', base=2)
        tp3 = int('00', base=2)

        fp1 = int('10', base=2)
        fp2 = int('01', base=2)
        fp3 = int('00', base=2)

        tplist = [tp1, tp2, tp3]
        fplist = [fp1, fp2, fp3]

        res = FitnessUtils._disjointness(tplist, fplist)
        self.assertEqual(1, res)

    def test_satisfy_expl(self):
        hashtable = {('col1', 0): int('111', base=2),
                     ('col1', 1): int('101', base=2),
                     ('col2', 0): int('111', base=2),
                     ('col2', 1): int('001', base=2)}
        cond1 = ('col1', 0, 1)
        cond2 = ('col2', 0, 1)
        expl = [cond1, cond2]

        expected = int('010', base=2)
        actual = FitnessUtils._satisfy_expl(expl, hashtable)

        self.assertEqual(expected, actual)


    def test_satisfy_cond(self):
        hashtable = {('col1', 0): int('111', base=2),
                     ('col1', 1): int('101', base=2)}

        cond = ('col1', 0, 1)
        expected = int('010', base=2)
        actual = FitnessUtils._satisfy_cond(cond, hashtable)

        self.assertEqual(expected, actual)
