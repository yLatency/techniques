from datetime import date

import shutil
from pyspark.sql import SparkSession

from reshape import loadSpansByDay, createEndpointTraces


def writeSpans(spark, file):
    day = date(2018, 11, 14)
    spans = loadSpansByDay(day, spark)
    endpointTraces = createEndpointTraces(spans)
    endpointTraces.write.parquet(file)


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
    day = date(2018, 11, 14)
    spans = loadSpansByDay(day, spark)
    endpointTraces = createEndpointTraces(spans)
    endpointTraces.write.parquet(file)

finally:
    if spark is not None:
        spark.stop()
        shutil.rmtree('metastore_db')
