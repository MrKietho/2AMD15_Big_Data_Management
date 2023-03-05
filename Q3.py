from pyspark import SparkConf, SparkContext

"""
    1. Build covariance matrix
    2. Use Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z) + 2 * Cov(X, Y) + 2 * Cov(X, Z) + 2 * Cov(Y, Z)
       to find the triplets of vectors with threshold aggregate variance
"""

sc = SparkContext(conf=SparkConf().setAppName("Q3").setMaster("local[*]").set("spark.executor.memory", "8g").set("spark.driver.memory", "8g").set("spark.executor.heartbeatInterval", "600").set("spark.driver.cores", "4").set("spark.executor.cores", "4"))

threshold = 410

path = "vectors.csv"
vector_file = sc.textFile(path)
vectors = vector_file.map(lambda x: x.split(",")).map(lambda x: (x[0], [int(i) for i in x[1].split(";")])).collectAsMap()

pairs = vector_file.map(lambda x: x.split(",")[0])
#TODO: Find a faster way to generate pairs, this is unnecessarily slow probably because of sting comparisons
pairs = pairs.cartesian(pairs).filter(lambda x: x[0] <= x[1]).map(lambda x: ((x[0], x[1]), (vectors[x[0]], vectors[x[1]])))

mean = lambda x: sum([i for i in x]) / len(x)
covariance = lambda x, y: sum( [ (x[i] - mean(x)) * (y[i] - mean(y)) for i in range(len(x)) ] ) / (len(x) - 1)

covariances = pairs.mapValues(lambda x: covariance(x[0], x[1])).take(10)

#TODO find triplets with threshold aggregate variance using Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z) + 2 * Cov(X, Y) + 2 * Cov(X, Z) + 2 * Cov(Y, Z)

print(covariances)


