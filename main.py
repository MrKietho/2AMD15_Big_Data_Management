from pyspark import SparkConf, SparkContext, RDD
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, array
from pyspark.sql.functions import udf, split,  array, expr, pow, size, posexplode, transform
from pyspark.sql.types import ArrayType, DoubleType, FloatType
import numpy as np
import time
import math
import random

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
        .select("_c0", "avg", "vectors", "variance")
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
         result1 .withColumn("cov_arr", expr("transform(v1.vectors, (x, i) -> v1.vectors[i] * v2.vectors[i])"))
         .withColumn("sum", expr("aggregate(cov_arr, cast(0 as double), (acc, x) -> acc + x)"))
         .withColumn("cov", expr("((sum/10000) - (v1.avg*v2.avg))"))
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



def q4(spark_context: SparkContext, rdd: RDD, tau: float, epsilon: float, delta: float):
    n = 10000
    depth = int(np.ceil(math.log(1/delta)))
    width = int(np.ceil(math.e/epsilon))


    primes = [3079, 6151, 12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869, 3145739, 6291469, 12582917,
              25165843, 50331653, 100663319, 201326611, 402653189, 805306457, 1610612741]

    def variance(sketch1, sketch2, sketch3):
        sketch = np.add(np.add(sketch1, sketch2), sketch3)
        arr_x2 = []
        x = np.sum(sketch)

        for i in range(depth):
            x += sum(sketch[i])
            arr_x2.append(sum(np.square(sketch[i])))

        exp_x2 = min(arr_x2) / n
        exp_x = x / (n*depth)

        return exp_x2 - (exp_x**2)

    def vector_to_cms(vector):
        # Initialize the count min sketch with zeros.
        count_min_sketch = np.zeros((depth, width))


        # For each non-zero element in the vector, update the count min sketch.
        for i in range(len(vector)):
            value = vector[i]

            for j in range(depth):
                # hash_value = hash(str(i) + 'A' + str(j)) % primes[j] % width
                hash_value = (((i+2394)*23893)+111) % primes[j] % width
                count_min_sketch[j][hash_value] += value

        return count_min_sketch

    # Broadcast original RDD, so that we can access the sketches later at every node
    sketch_rdd = rdd.map(lambda x: (x[0], vector_to_cms(x[1])))
    sketch_broadcast = spark_context.broadcast(sketch_rdd.collectAsMap())

    # Compute rdd with ID
    id_rdd = rdd.map(lambda x: (x[0])).cache()

    # Create all unique pairs by computing cartesian product on id rdd
    id_pairs = id_rdd.cartesian(id_rdd).filter(lambda x: x[0] < x[1])

    # Create all triples by combining id_pairs with id_rdd
    id_triple = id_pairs.cartesian(id_rdd).filter(lambda x: x[0][1] < x[1])

    # Compute the variance for all triples by summing the sketches and then compute the variance
    variance_rdd = id_triple.map(lambda x: ((x[0][0], x[0][1], x[1]), variance(sketch_broadcast.value.get(x[0][0]), sketch_broadcast.value.get(x[0][1]), sketch_broadcast.value.get(x[1])))).cache()

    # Filter for the right tau
    tau_variance_rdd = variance_rdd.filter(lambda x: x[1] <= tau)

    for element in tau_variance_rdd.collect():
        print(element)

    return


if __name__ == '__main__':

    on_server = False  # TODO: Set this to true if and only if deploying to the server
    spark_context = get_spark_context(on_server)

    # data_frame = q1a(spark_context, on_server)
    # result = q2(spark_context, data_frame, 410)
    # result.show()

    # rdd = q1b(spark_context, on_server)
    # q3(spark_context, rdd, 410)

    rdd = q1b(spark_context, on_server)
    q4(spark_context, rdd, 32410, 0.001, 0.1)

    spark_context.stop()
