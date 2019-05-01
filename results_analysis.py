from reshape import loadExperimentSpans
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import shutil
import csv
from datetime import datetime
import ast
from functools import reduce


# For a given latency interval compute f-measure for each injected pattern
# and return true positives, selected traces and computed f-measure
def scores_every_pattern(traces_between_interval,
                         sel_traces,
                         num_patterns):
    for rc in range(num_patterns):
        p_traces = traces_between_interval.filter(col('experiment') == rc)
        tp_traces = sel_traces.filter(col('experiment') == rc)
        p = p_traces.count()
        tp = tp_traces.count()
        prec = tp / sel_traces.count() if sel_traces.count() else 0
        rec = tp / p if p else 0
        fmeasure = 2 * prec * rec / (prec + rec) if prec + rec else 0
        yield tp_traces, sel_traces, fmeasure


# Compute best scoring (f-measure) for each latency interval
# and return true positives and selected traces
def best_scoring_explanation(explanations, splitPoints):
    for i, sp in enumerate(splitPoints[:-1]):
        explanation = explanations[i]
        from_, to = sp, splitPoints[i + 1]
        traces_between_interval = expTraces.filter((from_ < col(frontend)) & (col(frontend) <= to))
        sel_traces = traces_between_interval
        for b, t in zip(backends, explanation):
            sel_traces = sel_traces.filter(col(b) >= t)
        it = scores_every_pattern(traces_between_interval, sel_traces, num_patterns)
        tp_traces, sel_traces, _ = max(it, key=lambda x: x[2])
        yield tp_traces, sel_traces


def getTraces(file):
    spark = (SparkSession.builder
             .master("local[*]")
             .appName("My app")
             .config("spark.executor.memory", "40G")
             .config('spark.sql.catalogImplementation', 'hive')
             .config("spark.driver.memory", "30G")
             .config("spark.driver.extraClassPath", "/home/luca/lib/elasticsearch-spark-20_2.11-6.4.1.jar")
             .getOrCreate())

    traces = (spark.read.option('mergeSchema', 'true')
              .parquet(file))

    frontend = 'web-service_HomeControllerHome_avg_dur'
    backends = [c for c in traces.columns
                if c != 'traceId' and c.endswith('avg_self_dur')]

    return spark, traces, frontend, backends

def union(dfx, dfy):
    return dfx.union(dfy)


spark = None
try:
    with open('experiments.csv', mode='r') as exp_file:
        exp_reader = csv.reader(exp_file, delimiter=';')
        for exp_row in exp_reader:
            num_patterns, from_ts, to_ts = [int(x) for x in exp_row]
            from_ = datetime.fromtimestamp(from_ts)
            to = datetime.fromtimestamp(to_ts)
            spark, traces, frontend, backends = getTraces('dataset/%s/%s_%s.parquet' % tuple(exp_row))
            expSpans = loadExperimentSpans(from_, to, spark)
            expTraces = traces.join(expSpans, on='traceId')
            with open('results/%s/%s_%s.csv' % tuple(exp_row), mode='r') as res_file:
                res_reader = csv.reader(res_file, delimiter=';')
                split_points = [float(s) for s in next(res_reader)]
                exec_time = float(next(res_reader)[0])
                explanations = [ast.literal_eval(r[0]) for r in res_reader]
                dfs = [expTraces.filter(col('experiment') == i) for i in range(num_patterns)]
                P = reduce(union, dfs)
                best_tp_sel = list(best_scoring_explanation(explanations, split_points))
                G = reduce(union, (tp for tp, _ in best_tp_sel))
                total_sel = sum(sel.count() for _, sel in best_tp_sel)
                rec = G.count() / P.count()
                prec = G.count() / total_sel
                f_measure = 2 * prec * rec / (prec + rec)
                print(f_measure, prec, rec)
finally:
    if spark:
        spark.stop()
        shutil.rmtree('metastore_db')
