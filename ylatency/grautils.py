from functools import reduce
from statistics import pvariance, mean


class FitnessUtils:
    def __init__(self, backends, pos_hashtable, neg_hashtable):
        self.backends = backends
        self.pos_hashtable = pos_hashtable
        self.neg_hashtable = neg_hashtable

    def _tplist(self, expllist):
        return [self._satisfy_expl(expl, self.pos_hashtable) for expl in expllist]

    def _fplist(self, expllist):
        return [self._satisfy_expl(expl, self.neg_hashtable) for expl in expllist]

    # number of bit equal one
    @classmethod
    def _cardinality(cls, bitstring):
        return bin(bitstring).count('1')

    @classmethod
    def _satisfy_expl(cls, expl, hashtable):
        bs_list = [cls._satisfy_cond(cond, hashtable) for cond in expl]
        return reduce(lambda x, y: x & y, bs_list)

    @classmethod
    def _satisfy_cond(cls, cond, hashtable):
        col, min_, max_ = cond
        bs_min = hashtable[col, min_]
        bs_max = hashtable[col, max_]
        return bs_min & ~ bs_max


    @classmethod
    def _recall(cls, tplist, num_pos):
        bitstring = reduce(lambda x, y: x | y, tplist)
        num_tp = cls._cardinality(bitstring)
        return num_tp / num_pos

    @classmethod
    def _precision(cls, tplist, fplist):
        bs_tp = reduce(lambda x, y: x | y, tplist)
        bs_fp = reduce(lambda x, y: x | y, fplist)
        num_tp = cls._cardinality(bs_tp)
        support = num_tp + cls._cardinality(bs_fp)
        return num_tp / support if support > 0 else 0

    @classmethod
    def _disjointness(cls, tplist, fplist):
        tp_card = [cls._cardinality(tp) for tp in tplist]
        fp_card = [cls._cardinality(fp) for fp in fplist]
        xtp = reduce(lambda x, y: x ^ y, tplist)
        xfp = reduce(lambda x, y: x ^ y, fplist)
        sum_card = (sum(tp_card) + sum(fp_card))
        res = 0
        if sum_card:
            res = (cls._cardinality(xtp) + cls._cardinality(xfp)) / sum_card
        return res

    @classmethod
    def _dissimilarity(cls, tplist, fplist, target_col_pos, target_col_neg):
        dissimilarity = 0
        reverse_bitstring = lambda bs: bin(bs).replace('0b', '')[::-1]
        for tp, fp in zip(tplist, fplist):
            reversed_tp = reverse_bitstring(tp)
            reversed_fp = reverse_bitstring(fp)
            values_tp = [val for bit, val in zip(reversed_tp, target_col_pos[::-1]) if bit == '1']
            values_fp = [val for bit, val in zip(reversed_fp, target_col_neg[::-1]) if bit == '1']
            values = values_tp + values_fp
            if values:
                mean_ = mean(values)
                variability = sum((v - mean_)**2 for v in values)
                dissimilarity += variability
        return dissimilarity


    @classmethod
    def _sizesofclusters(cls, tplist, fplist, target_col_pos, target_col_neg):
        sizes=[]
        reverse_bitstring = lambda bs: bin(bs).replace('0b', '')[::-1]
        for tp, fp in zip(tplist, fplist):
            reversed_tp = reverse_bitstring(tp)
            reversed_fp = reverse_bitstring(fp)
            values_tp = [val for bit, val in zip(reversed_tp, target_col_pos[::-1]) if bit == '1']
            values_fp = [val for bit, val in zip(reversed_fp, target_col_neg[::-1]) if bit == '1']
            values = values_tp + values_fp
            sizes.append(len(values))
        return sizes


    def recall(self, expllist):
        tplist = self._tplist(expllist)
        num_pos = self.pos_hashtable['cardinality']
        return self._recall(tplist, num_pos)

    def precision(self, expllist):
        tplist = self._tplist(expllist)
        fplist = self._fplist(expllist)
        return self._precision(tplist, fplist)

    def disjointness(self, expllist):
        tplist = self._tplist(expllist)
        fplist = self._fplist(expllist)
        return self._disjointness(tplist, fplist)

    def harmonic_mean(self, expllist):
        prec = self.precision(expllist)
        rec = self.recall(expllist)
        disj = self.disjointness(expllist)

        den = prec*rec + prec*disj + rec*disj
        num = 3*prec*rec*disj
        return num/den if den else 0

    def fscore(self, expllist):
        prec = self.precision(expllist)
        rec = self.recall(expllist)
        den = prec + rec
        if den != 0:
            score = (2 * prec * rec)/den
        else:
            score = 0
        return score

    def numclusters(self, expllist):
        return len(expllist)

    def dissimilarity(self, expllist):
        tplist = self._tplist(expllist)
        fplist = self._fplist(expllist)
        return self._dissimilarity(tplist, fplist, self.pos_hashtable['target'], self.neg_hashtable['target'])


    def sizesofclusters(self, expllist):
        tplist = self._tplist(expllist)
        fplist = self._fplist(expllist)
        return self._sizesofclusters(tplist, fplist, self.pos_hashtable['target'], self.neg_hashtable['target'])

    def feasible(self, expllist):
        disj = self.disjointness(expllist)
        num_pos = self.pos_hashtable['cardinality']
        return disj == 1 and min(self.sizesofclusters(expllist)) >= num_pos * 0.05
