from pyspark import SparkConf, SparkContext, RDD
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import variance
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
import numpy as np

def get_spark_context(on_server) -> SparkContext:
    spark_conf = SparkConf().setAppName("2AMD15")
    if not on_server:
        spark_conf = spark_conf.setMaster("local[*]")
    spark_context = SparkContext.getOrCreate(spark_conf)

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


def q2(spark_context: SparkContext, data_frame: DataFrame, tau: float):
    spark_session = SparkSession(spark_context)

    data_frame.show()

    # SQL query to select all possible triples of vectors and calculate their aggregate variance
    result = (
        data_frame.alias("v1").crossJoin(data_frame.alias("v2"))
        .crossJoin(data_frame.alias("v3"))
        .filter("v1._c0 < v2._c0 AND v2._c0 < v3._c0")
        .select("v1._c0", "v2._c0", "v3._c0",
                variance(sum("v1.vectors", "v2.vectors", "v3.vectors")).alias("var_agg"))
        .groupBy("v1._c0", "v2._c0", "v3._c0")
        .having(f"var_agg <= {tau}")
    )

    result.show()

    return result


def q3(spark_context: SparkContext, rdd: RDD):
    # TODO: Imlement Q3 here
    return


# def q4(spark_context: SparkContext, rdd: RDD):
def q4(spark_context: SparkContext):
    from pyspark.ml.feature import CountVectorizer

    # Initialize Spark
    conf = SparkConf().setAppName("Count-Min Sketch").setMaster("local")
    sc = SparkContext.getOrCreate(conf)

    # TODO change 250 vectors
    # Create a set of 250 random vectors
    vectors = sc.parallelize([np.random.rand(10) for i in range(250)])

    # Convert each vector to a hash value using MurmurHash
    hashes = vectors.map(lambda v: hash(str(v)))

    # Create a Count-Min Sketch with 10 hash functions and 1000 hash table sizes
    sketch = CountVectorizer(inputCol="hashes", outputCol="counts", vocabSize=1000, minDF=2)

    hashes_df = hashes.toDF("hashes")

    # Fit the sketch to the hashes
    model = sketch.fit(hashes_df)

    # # Transform the hashes to counts using the sketch
    # counts = model.transform(hashes_df)

    # # Compute the approximate variance for each vector
    # mean = vectors.mean()
    # sum_squares = counts.rdd.map(lambda row: np.sum(np.square(vectors.filter(lambda v: hash(str(v)) == row["hashes"]).collect()[0] - mean)) * row["counts"]).sum()
    # approx_variances = counts.rdd.map(lambda row: (row["hashes"], np.sum(np.square(vectors.filter(lambda v: hash(str(v)) == row["hashes"]).collect()[0] - mean)) * row["counts"] / sum_squares)).collect()

    # # Compute the approximate aggregate variance for each triple of vectors
    # triples = vectors.cartesian(vectors).cartesian(vectors).filter(lambda triple: triple[0][0] != triple[0][1] and triple[0][0] != triple[1] and triple[0][1] != triple[1] and triple[0][0] < triple[0][1] and triple[0][1] < triple[1]).map(lambda triple: (triple, (triple[0][0], triple[0][1], triple[1], np.sum(triple[0][0]) + np.sum(triple[0][1]) + np.sum(triple[1]), len(triple[0][0]) + len(triple[0][1]) + len(triple[1]), np.sum([approx_variance[1] for approx_variance in approx_variances if approx_variance[0] in [hash(str(v)) for v in triple[0] + (triple[1],)]]))))
    # lower_variances = triples.filter(lambda triple: triple[1][5] < 0.1).map(lambda triple: triple[1][0:3]).collect()
    # higher_variances = triples.filter(lambda triple: triple[1][5] > 0.9).map(lambda triple: triple[1][0:3]).collect()

    # print("Triples with lower variances:")
    # for triple in lower_variances:
    #     print(triple)

    # print("Triples with higher variances:")
    # for triple in higher_variances:
    #     print(triple)
    return


if __name__ == '__main__':

    on_server = False  # TODO: Set this to true if and only if deploying to the server
    spark_context = get_spark_context(on_server)

    data_frame = q1a(spark_context, on_server)

    rdd = q1b(spark_context, on_server)

    # result = q2(spark_context, data_frame, 40)

    # result.show()

    #q3(spark_context, rdd)

    #q4(spark_context, rdd)
    #q4(spark_context)

    spark_context.stop()
