from pyspark import SparkConf, SparkContext
from itertools import combinations
import numpy as np

"""
    Solves Question 3 of the assignment by using the following steps:
    1. Read the vectors from the file
    1. Build covariance matrix of all pairs of vectors
    3. Generate all triplets of vectors
    4. Calculate the aggregate variance from of each triplet using Var(X + Y + Z) = Cov(X, X) + Cov(Y, Y) + Cov(Z, Z) + 2 * Cov(X, Y) + 2 * Cov(X, Z) + 2 * Cov(Y, Z)
    5. Filter triplets with aggregate variance less than or equal to threshold
"""

# Methods to calculate covariance and aggregate variance
mean = lambda x: sum([i for i in x]) / len(x)
variance = lambda x: sum([pow(i - mean(x), 2) for i in x]) / (len(x) - 1)
covariance = lambda x, y: sum( [ (x[i] - mean(x)) * (y[i] - mean(y)) for i in range(len(x)) ] ) / (len(x) - 1)
agg_variance = lambda x, y, z: covariances[(x, x)] + covariances[(y, y)] + covariances[(z, z)] + 2 * covariances[(x, y)] + 2 * covariances[(x, z)] + 2 * covariances[(y, z)]

sc = SparkContext(conf=SparkConf().setAppName("Q3").setMaster("local[*]"))

threshold = 410

# Load vector file
path = "vectors.csv"
vector_file = sc.textFile(path)

# Create a dictionary of vectors (ID, Vector)
vectors = vector_file.map(lambda x: x.split(",")).map(lambda x: (x[0], [int(i) for i in x[1].split(";")])).collectAsMap()

vector_ids = list(vectors.keys())

# Create a dictionary of vector id pairs
vector_pairs = list(combinations(vector_ids, 2)) + [(x, x) for x in vector_ids]

# Calculate covariance matrix for all pairs of vectors
# covariance_values = np.cov([vectors[x] for x in vector_ids])
covariances = sc.parallelize(vector_pairs).map(lambda x: (x, covariance(vectors[x[0]], vectors[x[1]]))).collectAsMap()

# Create a dictionary of covariances
# covariances = {(vector_ids[i], vector_ids[j]): covariance_values[i][j] for i in range(len(vector_ids)) for j in range(len(vector_ids))}

# Create a list of all triplets of vectors
vector_triples = list(combinations(vector_ids, 3))
# vector_triples = [((x[0], x[1], x[2]), [vectors[x[0]][i] + vectors[x[1]][i] + vectors[x[2]][i] for i in range(len(vectors[x[0]]))]) for x in vector_triples]

# Calculate aggregate variance for each triplet and filter triplets with aggregate variance less than or equal to threshold
result = sc.parallelize(vector_triples).map(lambda x: ((x[0], x[1], x[2]), agg_variance(x[0], x[1], x[2]))).filter(lambda x: x[1] <= threshold).collect()

# result = sc.parallelize(vector_triples).map(lambda x: (x, [vectors[x[0]][i] + vectors[x[1]][i] + vectors[x[2]][i] for i in range(len(vectors[x[0]]))])).mapValues(lambda x: variance(x)).filter(lambda x: x[1] <= threshold).collect()
# result = sc.parallelize(vector_triples).mapValues(lambda x: variance(x)).filter(lambda x: x[1] <= threshold).collect()

print(result)

sc.stop()