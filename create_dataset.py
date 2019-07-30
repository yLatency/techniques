import os

from pyspark.sql import SparkSession

from tracelib.reshape import loadSpansByInterval, createEndpointTraces
from datetime import datetime
import csv
import shutil

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

    with open('experiments.csv', mode='r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            num_patterns, from_ts, to_ts = row
            filename = 'dataset/%s/%s_%s.parquet' % (num_patterns, from_ts, to_ts)
            if not os.path.exists(filename):
                from_ = datetime.fromtimestamp(int(from_ts))
                to = datetime.fromtimestamp(int(to_ts))
                spans = loadSpansByInterval(from_, to, spark)
                endpointTraces = createEndpointTraces(spans)
                endpointTraces.write.parquet(filename)
finally:
    if spark is not None:
        spark.stop()
        shutil.rmtree('metastore_db')
