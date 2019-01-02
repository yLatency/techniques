from pyspark import SparkConf
from pyspark.sql import functions as f, SparkSession, DataFrame
from datetime import date, datetime, timedelta, time
from functools import reduce

conf = SparkConf()

spark = (SparkSession.builder
         .master("local[*]")
         .appName("My app")
         .config("spark.executor.memory", "40G")
         .config('spark.sql.catalogImplementation', 'hive')
         .config("spark.driver.memory", "30G")
         .config("spark.driver.extraClassPath", "/home/luca/lib/elasticsearch-spark-20_2.11-6.4.1.jar")
         .getOrCreate())


def loadSpansByInterval(from_, to):
    fromTimestamp = int(from_.timestamp() * 1000000)
    toTimestamp = int(to.timestamp() * 1000000)
    return (spark.read.format("es")
            .option("es.resource", "zipkin*")
            .load()
            .select('traceId',
                    f.concat_ws('_', *['localEndpoint.serviceName', 'name']).alias('endpoint'),
                    'duration',
                    'id',
                    'kind',
                    'timestamp',
                    'parentId')
            .filter(f.col('timestamp').between(fromTimestamp, toTimestamp)))


def loadSpansByDay(day):
    return reduce(DataFrame.union,
                  [loadSpansByInterval(from_, to) for from_, to in HoursOfTheDay(day)])


class HoursOfTheDay:
    def __init__(self, date):
        self.datetime = datetime.combine(date, time.min)
        self.hour = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.hour > 23:
            raise StopIteration
        else:
            from_ = self.datetime + timedelta(hours=self.hour)
            to = self.datetime + timedelta(hours=self.hour + 1) - timedelta(microseconds=1)
            self.hour += 1
            return from_, to


def createEndpointTraces(spans):
    serverSpansWithClientsDuration = craeteServerSpansWithClientsDuration(spans)
    serverSpansWithSelfDuration = (serverSpansWithClientsDuration
                                   .withColumn('self_duration',
                                               f.col('duration') - f.col('clients_duration')))
    avgDurPerTraceEndpoint = createAvgDurPerTraceEndpointPairs(serverSpansWithSelfDuration)
    return (avgDurPerTraceEndpoint
            .groupBy('traceId')
            .pivot('endpoint')
            .agg(f.first('avg_self_duration').alias('avg_self_dur'),
                 f.first('avg_duration').alias('avg_dur'))
            .dropna())


def createAvgDurPerTraceEndpointPairs(serverSpansWithSelfDuration):
    return (serverSpansWithSelfDuration
            .groupBy('traceId', 'endpoint')
            .agg(f.avg('duration').alias('avg_duration'),
                 f.avg('self_duration').alias('avg_self_duration')))


def filterServerSpans(spans):
    return (spans.filter(spans.kind == 'SERVER')
            .drop('parentId', 'kind'))


def filterClientSpans(spans):
    return (spans.filter(spans.kind == 'CLIENT')
            .drop('kind'))


def createClientsDuration(serverSpans, clientSpans):
    return (clientSpans.groupBy('parentId')
            .agg(f.sum('duration').alias('clients_duration')))


def craeteServerSpansWithClientsDuration(spans):
    serverSpans = filterServerSpans(spans)
    clientSpans = filterClientSpans(spans)
    clientsDuration = createClientsDuration(serverSpans, clientSpans)
    return (serverSpans.join(clientsDuration,
                             serverSpans.id == clientsDuration.parentId,
                             'left_outer')
            .drop('parentId')
            .na.fill(0))