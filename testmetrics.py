from datetime import date

from pyspark.sql import SparkSession

from metrics import Metrics
from pyspark.sql.functions import col

import random

file = 'mock/data.parquet'
spark = None

try:
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
    frontendSLA = traces.approxQuantile([frontend], [0.99], 0)[0][0]
    thresholds = [qs[0] for qs in traces.approxQuantile(backends, [0.99], 0)]

    random.seed(33)
    utils = Metrics(traces, backends, thresholds,
                    frontend, frontendSLA)

    for _ in range(10):
        random.seed(0)
        backendComb = random.choices(backends, k=3)

        thresholds = [q[0] for q in traces.approxQuantile(backendComb, [0.99], 0)]

        p = traces.filter(col(frontend) > frontendSLA).count()

        tp = (traces.filter(col(frontend) > frontendSLA)
              .filter(col(backendComb[0]) >= thresholds[0])
              .filter(col(backendComb[1]) >= thresholds[1])
              .filter(col(backendComb[2]) >= thresholds[2])).count()

        fp = (traces.filter(col(frontend) <= frontendSLA)
              .filter(col(backendComb[0]) >= thresholds[0])
              .filter(col(backendComb[1]) >= thresholds[1])
              .filter(col(backendComb[2]) >= thresholds[2])).count()

        rec = tp / p
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        fmeas = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0
        print(fmeas == utils.compute(backendComb)[0])

finally:
    if spark is not None:
        spark.stop()