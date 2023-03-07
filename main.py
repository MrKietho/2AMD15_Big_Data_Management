from pyspark import SparkConf, SparkContext, RDD
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, array
from pyspark.sql.functions import udf, split
from pyspark.sql.types import ArrayType, DoubleType, FloatType
import numpy as np

def get_spark_context(on_server) -> SparkContext:
    spark_conf = SparkConf().setAppName("2AMD15")
    if not on_server:
        spark_conf = spark_conf.setMaster("local[*]")
    spark_context = SparkContext.getOrCreate(spark_conf)

    spark_context.setLogLevel("ERROR") # Remove

    if on_server:
        # TODO: You may want to change ERROR to WARN to receive more info. For larger data sets, to not set the
        # log level to anything below WARN, Spark will print too much information.
        spark_context.setLogLevel("ERROR")

    return spark_context


def q1a(spark_context: SparkContext, on_server: bool) -> DataFrame:
    vectors_file_path = "/vectors.csv" if on_server else "vectors.csv"

    spark_session = SparkSession(spark_context)

    def parse_vector(vector_string):
        return list(map(float, vector_string.strip().split(';')))

    parse_vector_udf = udf(parse_vector, ArrayType(DoubleType()))

    df = (spark_session.read.format("csv")
          .option("header", "false")
          .load("vectors.csv")
          .withColumn("vectors", parse_vector_udf("_c1"))
          .select("_c0", "vectors"))

    return df


def q1b(spark_context: SparkContext, on_server: bool) -> RDD:
    vectors_file_path = "/vectors.csv" if on_server else "vectors.csv"

    # create rdd
    rdd1 = spark_context.textFile(vectors_file_path).map(lambda line: tuple(line.split(';')))

    return rdd1

# Define a custom aggregation function that sums the values of three arrays
def my_agg(arr1, arr2, arr3):
    for i in range(len(arr1)):
      arr1[i] = arr1[i] + arr2[i] + arr3[i]
    return float(((1 / len(arr1)) * sum(x*x for x in arr1)) - np.power(sum(x/len(arr1) for x in arr1), 2))

# Register the UDF with Spark
my_agg_udf = udf(my_agg, FloatType())


def q2(spark_context: SparkContext, data_frame: DataFrame, tau: float):
    spark_session = SparkSession(spark_context)

    data_frame.show()

    # SQL query to select all possible triples of vectors and calculate their aggregate variance
    result = (
        data_frame.alias("v1")
        .crossJoin(data_frame.alias("v2"))
        .crossJoin(data_frame.alias("v3"))
        .filter("v1._c0 < v2._c0 AND v2._c0 < v3._c0")
        .limit(20)
        .select("v1._c0", "v2._c0", "v3._c0", 
                my_agg_udf(col("v1.vectors"), col("v2.vectors"), col("v3.vectors")).alias("var_agg"))
        # .groupBy("v1._c0", "v2._c0", "v3._c0")
        .filter(f"var_agg <= {tau}")
    )

    result.show()

    return result


def q3(spark_context: SparkContext, rdd: RDD):
    # TODO: Imlement Q3 here
    return


def q4(spark_context: SparkContext, rdd: RDD):
    # TODO: Imlement Q4 here
    return


if __name__ == '__main__':

    on_server = False  # TODO: Set this to true if and only if deploying to the server
    spark_context = get_spark_context(on_server)

    data_frame = q1a(spark_context, on_server)

    rdd = q1b(spark_context, on_server)

    result = q2(spark_context, data_frame, 410)

    result.show()

    #q3(spark_context, rdd)

    #q4(spark_context, rdd)

    spark_context.stop()
