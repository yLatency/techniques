from functools import reduce


class FitnessUtils:
    def __init__(self, backends, pos_hashtable, neg_hashtable):
        self.backends = backends
        self.pos_hashtable = pos_hashtable
        self.neg_hashtable = neg_hashtable

    def _tplist(self, explset):
        sorted_explset = sorted(explset, key=str)
        return [self.satisfy_expl(expl, self.pos_hashtable) for expl in sorted_explset]

    def _fplist(self, explset):
        sorted_explset = sorted(explset, key=str)
        return [self.satisfy_expl(expl, self.neg_hashtable) for expl in sorted_explset]

    # number of bit equal one
    @classmethod
    def _cardinality(cls, bitstring):
        return bin(bitstring).count('1')

    def satisfy_expl(self, expl, hashtable):
        pass

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

    def recall(self, explset):
        tplist = self._tplist(explset)
        num_pos = self.pos_hashtable['cardinality']
        return self._recall(tplist, num_pos)

    def precision(self, explset):
        tplist = self._tplist(explset)
        fplist = self._fplist(explset)
        return self._precision(tplist, fplist)

    def disjointness(self, explset):
        tplist = self._tplist(explset)
        fplist = self._fplist(explset)
        return self._disjointness(tplist, fplist)

    def harmonic_mean(self, explset):
        prec = self.precision(explset)
        rec = self.recall(explset)
        disj = self.disjointness(explset)

        den = prec*rec + prec*disj + rec*disj
        num = 3*prec*rec*disj
        return num/den if den else 0
