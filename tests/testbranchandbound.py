import shutil

from pyspark.sql import SparkSession

from ks.branchandbound import BranchAndBound
from tests.enumeration import Enumeration

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
    frontendSLA = traces.approxQuantile([frontend], [0.999], 0)[0][0]
    thresholds = [qs[0] for qs in traces.approxQuantile(backends, [0.9], 0)]

    bestExpBnB = BranchAndBound(traces, backends, thresholds,
                                frontend, frontendSLA).compute()
    print(bestExpBnB.features, bestExpBnB.fmeasure)

    bestExpEnum = Enumeration(traces, backends, thresholds,
                              frontend, frontendSLA).compute()

    print(bestExpEnum.features, bestExpEnum.fmeasure)

    print('Are F-Measure equal? ', bestExpEnum.fmeasure == bestExpBnB.fmeasure)
    print('Delta:', bestExpBnB.fmeasure - bestExpEnum.fmeasure)

finally:
    if spark is not None:
        spark.stop()
        shutil.rmtree('metastore_db')
