import time

from pyspark.sql import SparkSession
from range_analysis import RangeAnalysis
import shutil
import csv


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


spark = None
try:
    with open('experiments.csv', mode='r') as exp_file:
        qsp = [0.7, 0.8, 0.9]
        reader = csv.reader(exp_file, delimiter=';')
        for row in reader:
            num_patterns, from_ts, to_ts = row
            spark, traces, frontend, backends = getTraces('dataset/%s/%s_%s.parquet' % tuple(row))
            max_latency = traces.select(frontend).rdd.max()[0]
            splitPoints = traces.approxQuantile([frontend], qsp, 0)[0] + [max_latency]

            t1 = time.perf_counter()
            ra = RangeAnalysis(traces, backends, frontend, splitPoints)
            selectedSplits, fmeasure, explanations = ra.explainsWithGA(5)
            t2 = time.perf_counter()

            with open('results/%s/%s_%s.csv' % tuple(row), mode='w+') as res_file:
                writer = csv.writer(res_file, delimiter=';')
                writer.writerow(splitPoints)
                writer.writerow([str(t2-t1)])
                for e in explanations:
                    writer.writerow(e)
finally:
    if spark:
        spark.stop()
        shutil.rmtree('metastore_db')
