from unittest import TestCase
from ylatency.graoperators import Operator
import random
import numpy as np

class TestOperator(TestCase):
    def setUp(self):
        self.thresholds_dict = {'a': [0, 1], 'b': [5, 6], 'c': [1, 5]}
        self.op = Operator(self.thresholds_dict)
        random.seed(33)
        np.random.seed(33)


    def test_cond(self):
        cond = self.op.cond('a')

        self.assertIsInstance(cond, tuple)
        self.assertEqual(3, len(cond))

        col, min_, max_ = cond
        self.assertEqual('a', col)
        self.assertIn(min_, self.thresholds_dict['a'])
        self.assertIn(max_, self.thresholds_dict['a'])
        self.assertLess(min_, max_)


    def test_expl(self):
        expl = self.op.expl()
        self._test_expl(expl)

    def _test_expl(self, expl):
        self.assertIsInstance(expl, list)
        self.assertGreater(len(expl), 0)
        for cond in expl:
            self.assertIsInstance(cond, tuple)
            self.assertEqual(3, len(cond))
            self.assertIn(cond[1], self.thresholds_dict[cond[0]])
            self.assertIn(cond[2], self.thresholds_dict[cond[0]])
            self.assertLess(cond[1], cond[2])

    def test_expllist(self):
        expllist = self.op.expllist()
        self._test_expllist(expllist)

    def _test_expllist(self, expllist):
        self.assertIsInstance(expllist, list)
        self.assertGreater(len(expllist), 0)
        for expl in expllist:
            self._test_expl(expl)

    def test_cx(self):
        el1 = self.op.expllist()
        el2 = self.op.expllist()
        el1, el2 = self.op.cx(el1, el2)
        self._test_expllist(el1)
        self._test_expllist(el2)

    def test_mut(self):
        expllist = self.op.expllist()
        self.op.mut(expllist)
        self._test_expllist(expllist)
