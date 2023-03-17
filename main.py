from pyspark import SparkConf, SparkContext, RDD
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, array
from pyspark.sql.functions import udf, split,  array, expr, pow, size, posexplode, transform
from pyspark.sql.types import ArrayType, DoubleType, FloatType
import numpy as np
import time
from itertools import combinations

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

    # First join 2 vectors
    result1 = (
        data_frame.alias("v1")
        .crossJoin(data_frame.alias("v2"))
        .filter("v1._c0 < v2._c0")
    )

    # Aggregate the 2 vectors
    result2 = result1.selectExpr(
            "v1._c0",
            "v2._c0",
            "transform(v1.vectors, (x, i) -> x + v2.vectors[i]) AS combined_vectors")

    # Join combined vectors with another vector
    result3 = (
        result2
        .crossJoin(data_frame.alias("v3"))
        .filter("v2._c0 < v3._c0 ")
    )

    # Aggregate the 3 vectors together
    result4 = result3.selectExpr(
            "v1._c0",
            "v2._c0",
            "v3._c0",
            "transform(combined_vectors, (x, i) -> x + v3.vectors[i]) AS combined_vectors")


    # Create column with sum and squared sum, then create variance column from that
    # Lastly select triples with variance <= tau
    result5 = (
        result4
        .withColumn("sum", expr("aggregate(combined_vectors, cast(0 as double), (acc, x) -> acc + x)"))
        .withColumn("sumsum", expr("aggregate(combined_vectors, cast(0 as double), (acc, x) -> acc + x*x)"))
        .withColumn("variance", expr("sumsum/10000 - (sum/10000)*(sum/10000)"))
        .select("v1._c0", "v2._c0", "v3._c0", "variance")
        .cache()
        .filter(f"variance  <= {tau}")
    )

    result5.explain()
    return result5


def q2a(spark_context: SparkContext, data_frame: DataFrame, tau: float):
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
    
    mean = lambda x: sum([i for i in x]) / len(x)
    variance = lambda x: sum([pow(i - mean(x), 2) for i in x]) / (len(x) - 1)
    covariance = lambda x, y: sum( [ (x[i] - mean(x)) * (y[i] - mean(y)) for i in range(len(x)) ] ) / (len(x) - 1)
    agg_variance = lambda x, y, z: covariances[(x, x)] + covariances[(y, y)] + covariances[(z, z)] + 2 * covariances[(x, y)] + 2 * covariances[(x, z)] + 2 * covariances[(y, z)]

    vectors = rdd.collectAsMap()

    vector_ids = list(vectors.keys())

    # Calculate covariance matrix for all pairs of vectors
    covariance_values = np.cov([vectors[x] for x in vector_ids])

    # Create a dictionary of covariances
    covariances = {(vector_ids[i], vector_ids[j]): covariance_values[i][j] for i in range(len(vector_ids)) for j in range(len(vector_ids))}

    # Create a list of all triplets of vectors
    vector_triples = list(combinations(vector_ids, 3))

    # Calculate aggregate variance for each triplet and filter triplets with aggregate variance less than or equal to threshold
    result = spark_context.parallelize(vector_triples).map(lambda x: ((x[0], x[1], x[2]), agg_variance(x[0], x[1], x[2]))).filter(lambda x: x[1] <= tau).collect()
    
    print(result)
    
    return
    


def q4(spark_context: SparkContext, rdd: RDD):
    # TODO: Imlement Q4 here
    return


if __name__ == '__main__':

    on_server = True  # TODO: Set this to true if and only if deploying to the server
    spark_context = get_spark_context(on_server)

    # data_frame = q1a(spark_context, on_server)
    # result = q2a(spark_context, data_frame, 410)
    # result.show()

    rdd = q1b(spark_context, on_server)
    q3(spark_context, rdd, 410)

    #q4(spark_context, rdd)

    spark_context.stop()
