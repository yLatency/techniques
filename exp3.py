import time

import shutil
from pyspark.sql import SparkSession

from ks.range_analysis import RangeAnalysis
import csv

spark = None


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

    return traces, frontend, backends

try:

    files = ['mock/data1.parquet', 'mock/data2.parquet']

    with open('results.csv', mode='w') as resfile:
        for file in files:
            traces, frontend, backends = getTraces(file)
            Q = [0.75, 0.9, 0.95]
            qsp = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
            splitPoints = traces.approxQuantile([frontend], qsp, 0)[0] + [traces.select(frontend).rdd.max()[0]]
            writer = csv.writer(resfile, delimiter=';')
            writer.writerow(['dataset:', file])
            writer.writerow(['split points(ms):', splitPoints])
            writer.writerow(['split points(quantiles -without max-)):', qsp])
            writer.writerow([])

            ra = RangeAnalysis(traces, backends, frontend, splitPoints)

            k = 5
            t1 = time.perf_counter()
            selectedSplits, fmeasure, explanations = ra.explainsWithGA(k=k)
            t2 = time.perf_counter()
            writer.writerow(['k:', k])
            writer.writerow(['GA splits:', selectedSplits])
            writer.writerow(['GA f-score:', fmeasure])
            writer.writerow(['GA explanations:', explanations])
            writer.writerow(['GA execution time:', t2 - t1])
            writer.writerow([])

            for q in Q:
                writer.writerow(['backends threshold', q])
                thresholds = [qs[0] for qs in traces.approxQuantile(backends, [q], 0)]
                t1 = time.perf_counter()
                selectedSplits, fmeasure, explanations = ra.explainsWithBnB(thresholds)
                t2 = time.perf_counter()
                writer.writerow(['BnB splits:', selectedSplits])
                writer.writerow(['BnB f-score:', fmeasure])
                writer.writerow(['BnB explanations:', explanations])
                writer.writerow(['BnB execution time:', t2 - t1])
                writer.writerows([[]])
            writer.writerows([[],[],[],[]])
        shutil.rmtree('metastore_db')

finally:
    if spark is not None:
        spark.stop()