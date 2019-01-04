from datetime import date
from reshape import loadSpansByDay, createEndpointTraces
from thresholds import KMeansSelector, Normalizer
from search import CacheMaker, GA
from pyspark.sql import SparkSession
import shutil

file = 'data/test.parquet'


def writeSpans(spark, file):
    day = date(2018, 11, 14)
    spans = loadSpansByDay(day, spark)
    endpointTraces = createEndpointTraces(spans)
    endpointTraces.write.parquet(file)


def execThrSel(spark, file):
    traces = (spark.read.option('mergeSchema', 'true')
              .parquet(file))
    frontend = 'web-service_HomeControllerHome_avg_dur'
    backends = [c for c in traces.columns
                if c != 'traceId' and c.endswith('avg_self_dur')]
    frontendSLA = traces.approxQuantile([frontend], [0.99], 0)[0][0]

    k = 3
    normalizer = Normalizer(backends, traces)
    normalizedTrace = normalizer.createNormalizedTrace()
    thresholdsDict = KMeansSelector(normalizedTrace, backends,
                                   frontend, frontendSLA).select(k)

    cache = CacheMaker(normalizedTrace, backends,
                       frontend, frontendSLA).create(thresholdsDict)
    ga = GA(backends, thresholdsDict, cache)

    res = ga.compute()

    for thresholds, fmeasure in res:
        for b,t in zip(backends, thresholds):
            print(b, ': ', normalizer.denormalizesThreshold(t, b))
        print('F-Measure: ', fmeasure, '\n\n')




try:
    spark = (SparkSession.builder
             .master("local[*]")
             .appName("My app")
             .config("spark.executor.memory", "40G")
             .config('spark.sql.catalogImplementation', 'hive')
             .config("spark.driver.memory", "30G")
             .config("spark.driver.extraClassPath", "/home/luca/lib/elasticsearch-spark-20_2.11-6.4.1.jar")
             .getOrCreate())

    day = date(2018, 11, 14)
    spans = loadSpansByDay(day, spark)
    endpointTraces = createEndpointTraces(spans)
    endpointTraces.write.parquet(file)

    execThrSel(spark, file)

finally:
    spark.stop()
    shutil.rmtree('data/')
    shutil.rmtree('metastore_db')
