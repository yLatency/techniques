from functools import reduce


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
        col, idx_min, idx_max = cond
        bs_min = hashtable[col, idx_min]
        bs_max = hashtable[col, idx_max]
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
        xtp_card = cls._cardinality(xtp)
        xfp_card = cls._cardinality(xfp)
        return (xtp_card + xfp_card) / (sum(tp_card) + sum(fp_card))

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
