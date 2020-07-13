from operator import itemgetter

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class DeCaf2:
    def __init__(self, traces, frontend, rpcs):
        df = traces.toPandas()
        self.rpcs = rpcs
        self.regr = RandomForestRegressor(n_estimators=200, min_samples_leaf=0.01, bootstrap=False, max_features=0.6)
        self.regr.fit(df[rpcs], df[frontend])

    def explain(self, k=10):
        predicates = []

        for estimator in self.regr.estimators_:
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            feature = estimator.tree_.feature
            threshold = estimator.tree_.threshold
            value = estimator.tree_.value
            samples = estimator.tree_.n_node_samples
            stack = [(0, [])]
            while len(stack) > 0:
                nodeid, scopepred = stack.pop()
                child_left, child_right = children_left[nodeid], children_right[nodeid]
                if child_left != child_right:
                    score = samples[child_left] * value[child_left][0][0] - samples[child_right] * value[child_right][0][0]
                    if score >= 0:
                        pred_ = (self.rpcs[feature[nodeid]], 0, threshold[nodeid] + 1)
                    else:
                        pred_ = (self.rpcs[feature[nodeid]], threshold[nodeid] + 1, 10 ** 6)
                    pred = scopepred + [pred_]
                    predicates.append((pred, abs(score)))
                    stack.append((child_left, pred))
                    stack.append((child_right, pred))

            return [p for p, score in sorted(predicates, key=itemgetter(1), reverse=True)[:k]]

class DeCaf:
    def __init__(self, traces, frontend, rpcs, sla):
        df = traces.toPandas()
        anomalytraces = traces[traces[frontend] > sla]
        min_samples_leaf = int(anomalytraces.count() * 0.05)

        X = df[rpcs]
        y = [1 if y_ > sla else 0 for y_ in df[frontend]]
        self.rpcs = rpcs
        self.regr = RandomForestClassifier(n_estimators=50,  min_samples_leaf=min_samples_leaf, bootstrap=False, max_features=0.6)
        self.regr.fit(X, y)

    def explain(self, k=10):
        predicates = []

        for estimator in self.regr.estimators_:
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            feature = estimator.tree_.feature
            threshold = estimator.tree_.threshold
            value = estimator.tree_.value
            stack = [(0, [])]
            while len(stack) > 0:
                nodeid, scopepred = stack.pop()
                child_left, child_right = children_left[nodeid], children_right[nodeid]
                if child_left != child_right:
                    score = value[child_left][0][1]/(value[child_left][0][0] + value[child_left][0][1]) - \
                            value[child_right][0][1]/(value[child_right][0][0] + value[child_right][0][1])
                    if score >= 0:
                        pred_ = (self.rpcs[feature[nodeid]], 0, threshold[nodeid] + 1)
                    else:
                        pred_ = (self.rpcs[feature[nodeid]], threshold[nodeid] + 1, 10 ** 6)
                    pred = scopepred + [pred_]
                    predicates.append((pred, abs(score)))
                    stack.append((child_left, pred))
                    stack.append((child_right, pred))

            return [p for p, score in sorted(predicates, key=itemgetter(1), reverse=True)[:k]]