from pyspark import SparkConf, SparkContext, RDD
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, array
from pyspark.sql.functions import udf, split,  array, expr, pow, size, posexplode, transform
from pyspark.sql.types import ArrayType, DoubleType, FloatType
import numpy as np
import time

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

    df = (spark_session.read.format("csv")
          .option("header", "false")
          .load(vectors_file_path)
          .withColumn("vectors", split("_c1", ";"))
          .select("_c0", "vectors")
          )

    return df


def q1b(spark_context: SparkContext, on_server: bool) -> RDD:
    vectors_file_path = "/vectors.csv" if on_server else "vectors.csv"

    # create rdd
    rdd1 = spark_context.textFile(vectors_file_path) \
        .map(lambda line: (line.split(',')[0], list(map(int, line.split(',')[1].split(';')))))

    return rdd1

# Define a custom aggregation function that sums the values of three arrays
def my_agg(arr1):
    start = time.time()
    value = float(((1 / len(arr1)) * np.sum(x * x for x in arr1)) - np.power(np.mean(arr1), 2))
    end = time.time()
    print(end-start)
    return value


# Register the UDF with Spark
my_agg_udf = udf(my_agg, FloatType())

def q2(spark_context: SparkContext, data_frame: DataFrame, tau: float):
    spark_session = SparkSession(spark_context)

    data_frame.cache()

    # Compute average, sum of squares and variance for all vectors
    result0 = (
        data_frame.withColumn("avg", expr("aggregate(vectors, cast(0 as double), (acc, x) -> acc + x)/10000"))
        .withColumn("sumsum", expr("aggregate(vectors, cast(0 as double), (acc, x) -> acc + x*x)"))
        .withColumn("variance", expr("sumsum/10000 - (avg*avg)"))
        .select("_c0", "vectors", "avg", "variance")
        .cache()
    )

    # Make dataframe with all unique combinations of vectors
    result1 = (
        result0.alias("v1")
        .crossJoin(result0.alias("v2"))
        .filter("v1._c0 < v2._c0")
    )

    # Compute the covariance for all the combinations of vectors
    result2 = (
        result1.withColumn("cov_arr", expr("transform(v1.vectors, (x, i) -> (v1.vectors[i] - v1.avg) * (v2.vectors[i] - v2.avg))"))
        .withColumn("sum", expr("aggregate(cov_arr, cast(0 as double), (acc, x) -> acc + x)"))
        .withColumn("cov", expr("sum/10000"))
        .selectExpr("v1._c0 AS X", "v1.variance AS X_var", "v2._c0 AS Y", "v2.variance as Y_var", "cov")
        .cache()
    )

    # Create triples by combining pairs XY and YZ
    # Then we have var X, Y, Z and cov XY, YZ (only miss the covariance for XZ)
    result3 = (
        result2.alias("v1")
        .join(result2.alias("v3"), col("v1.X") == col("v3.X"), "inner")
        .selectExpr("v1.X AS X", "v1.X_var AS X_var", "v1.Y AS Y", "v1.Y_var AS Y_var", "v1.cov AS cov1", "v3.Y AS Z", "v3.Y_var AS Z_var", "v3.cov AS cov2")
    )

    # Add the covariance for XZ for triple XYZ to each row
    result4 = (
         result3.alias("V1")
         .join(result2.alias("v3"), (col("v1.Y") == col("v3.X")) & (col("v1.Z") == col("v3.Y")), "inner")
         .select("v1.X", "v1.X_var", "v1.Y", "v1.Y_var", "v1.cov1", "v1.Z", "v1.Z_var", "v1.cov2", "v3.cov")

     )

    # Compute the variance for XYZ by using var(XYZ) = var(x)+var(y)+var(z)+2(cov(xy)+cov(yz)+cov(xz)
    result5 = (
        result4.withColumn("variance", expr("X_var + Y_var + Z_var + 2*(cov1 + cov2 + cov)"))
        .select("X", "Y", "Z", "variance")
        .cache()
    )

    # Filter the triples variance on the right tau
    result6 = result5.filter(f"variance  <= {tau}")

    result6.explain()

    return result6


def q3(spark_context: SparkContext, rdd: RDD, tau: float):
    n = 4

    # Compute the mean of each vector
    def mean(v):
        return sum(v) / n

    # Compute the variance of each vector
    def variance(v, m):
        return sum([(x - m) ** 2 for x in v]) / len(v)

    def cov(tuple1, tuple2):
        vec1 = tuple1[0]
        vec2 = tuple2[0]
        avg1 = tuple1[2]
        avg2 = tuple2[2]

        covariance = 0
        for t, x in enumerate(vec1):
            covariance += (vec1[t] - avg1) * (vec2[t] - avg2)
        return covariance/n

    def variance_triple(var1, var2, var3, cov1, cov2, cov3):
        return var1 + var2 + var3 + 2*(cov1 + cov2 + cov3)

    # Compute rdd with variance and mean for all vectors
    mean_rdd = rdd.map(lambda x: (x[0], (x[1], mean(x[1]))))
    variance_rdd = mean_rdd.map(lambda x: (x[0], (x[1][0], variance(x[1][0], x[1][1]), x[1][1]))).cache()

    # Join the RDD with itself to create all pairs of vectors
    pairs_rdd = variance_rdd.cartesian(variance_rdd).filter(lambda x: x[0][0] < x[1][0])

    #Add the covariance to the rdd -- ((X), (Y, varX, varY, covXY))
    cov_rdd = pairs_rdd.map(lambda x: ((x[0][0]), (x[1][0], x[0][1][1], x[1][1][1], cov(x[0][1], x[1][1]))))


    # Create all tripples -- ((X), (Y, varX, varY, covXY), (Z, varX, varZ, covXZ))
    triple_rdd = cov_rdd.join(cov_rdd).filter(lambda x: x[1][0][0] < x[1][1][0])

    # Make sure all cov are in the triples
    triple_cov = cov_rdd.map(lambda x: ((x[0], x[1][0]), x[1][3]))
    triple_rdd2 = triple_rdd.map(lambda x: ((x[1][0][0], x[1][1][0]), ((x[0], x[1][0][1], x[1][0][2], x[1][1][2]), (x[1][0][3], x[1][1][3]))))

    triple_complete_rdd = triple_rdd2.join(triple_cov)

    # Compute variance
    final1_result = triple_complete_rdd.map(lambda x: ((x[0][0], x[0][1], x[1][0][0][0]), (variance_triple(x[1][0][0][1], x[1][0][0][2], x[1][0][0][3], x[1][0][1][0], x[1][0][1][1], x[1][1]))))

    # Filter for tau
    final_result = final1_result.filter(lambda x: x[1] <= tau)

    for element in final_result.collect():
        print(element)

    return


def q4(spark_context: SparkContext, rdd: RDD):
    # TODO: Imlement Q4 here
    return


if __name__ == '__main__':

    on_server = False  # TODO: Set this to true if and only if deploying to the server
    spark_context = get_spark_context(on_server)

    data_frame = q1a(spark_context, on_server)
    result = q2(spark_context, data_frame, 410)
    result.show()

    #rdd = q1b(spark_context, on_server)
    #q3(spark_context, rdd, 410)

    #q4(spark_context, rdd)

    spark_context.stop()
