'''
The purpose of this first homework is to get acquainted with Spark and with its use to implement MapReduce algorithms. In preparation for the homework, you must set up your environment following the instructions given in Moodle Exam, in the same section as this page. After the set up is complete, test it using the WordCountExample program (Java or Python), and familiarize with the Spark methods it uses. The Introduction to Programming in Spark may turn out useful to this purpose.
Undirected graph with trianglesTRIANGLE COUNTING. In the homework you must implement and test in Spark two MapReduce algorithms to count the number of distinct triangles in an undirected graph G=(V,E), where a triangle is defined by 3 vertices u,v,w in V, such that (u,v),(v,w),(w,u) are in E. In the right image, you find an example of a 5-node graph with 3 triangles. Triangle counting is a popular primitive in social network analysis, where it is used to detect communities and measure the cohesiveness of those communities. It has also been used is different other scenarios, for instance: detecting web spam (the distributions of local triangle frequency of spam hosts significantly differ from those of the non-spam hosts), uncovering the hidden thematic structure in the World Wide Web (connected regions of the web which are dense in triangles represents a common topic), query plan optimization in databases (triangle counting can be used for estimating the size of some joins).

Both algorithms use an integer parameter C≥1, which is used to partition the data.

ALGORITHM 1: Define a hash function hC which maps each vertex u in V into a color hC(u) in [0,C−1].
To this purpose, we advise you to use the hash function hC(u)=((a⋅u+b)modp)modC where p=8191 (which is prime), a is a random integer in [1,p−1], and b is a random integer in [0,p−1].

Round 1:

Create C subsets of edges, where, for 0≤i<C, the i-th subset, E(i) consist of all edges (u,v) of E such that hC(u)=hC(v)=i. Note that if the two endpoints of an edge have different colors, the edge does not belong to any E(i) and will be ignored by the algorithm.

Compute the number t(i) triangles formed by edges of E(i), separately for each 0≤i<C.

Round 2:
 
Compute and return tfinal=C2∑0≤i<Ct(i) as final estimate of the number of triangles in G.

In the homework you must develop an implementation of this algorithm as a method/function MR_ApproxTCwithNodeColors. (More details below.)
ALGORITHM 2:
Round 1:

Partition the edges at random into C subsets E(0),E(1),...E(C−1). Note that, unlike the previous algorithm, now every edge ends up in some E(i).
Compute the number t(i) of triangles formed by edges of E(i), separately for each 0≤i<C.

Round 2: Compute and return  tfinal=C2∑0≤i<Ct(i) as final estimate of the number of triangles in G.
In the homework you must develop an implementation of this algorithm as a method/function MR_ApproxTCwithSparkPartitions that, in Round 1, uses the partitions provided by Spark, which you can access through method mapPartitions. (More details below.)

SEQUENTIAL CODE for TRIANGLE COUNTING. Both algorithms above require (in Round 1) to compute the number of triangles formed by edges in the subsets E(i)'s. To this purpose you can use the methods/functions provided in the following files: CountTriangles.java (Java) and CountTriangles.py (Python).

DATA FORMAT. To implement the algorithms assume that the vertices (set V
) are represented as 32-bit integers (i.e., type Integer in Java), and that the graph G is given in input as the set of edges E stored in a file. Each row of the file contains one edge stored as two integers (the edge's endpoints) separated by comma (','). Each edge of E appears exactly once in the file and E

does not contain multiple copies of the same edge.

TASK for HW1:

1) Write the method/function MR_ApproxTCwithNodeColors which implements ALGORITHM 1. Specifically, MR_ApproxTCwithNodeColors must take as input an RDD of edges and a number of colors C
and must return an estimate tfinal of the number of triangles formed by the input edges computed through transformations of the input RDD, as specified by the algorithm. It is important that the local space required by the algorithm be proportional to the size of the largest subset E(i) (hence, you cannot download the whole graph into a local data structure). Hint: define the hash function hC

inside MR_ApproxTCwithNodeColors, but before starting processing the RDD, so that all transformations of the RDD will use the same hash function, but different runs of MR_ApproxTCwithNodeColors will use different hash functions (i.e., defined by different values of a and b).

2) Write the method/function MR_ApproxTCwithSparkPartitions which implements ALGORITHM 2. Specifically, MR_ApproxTCwithSparkPartitions must take as input an RDD of edges and must return an estimate tfinal

of the number of triangles formed by the input edges computed through transformations of the input RDD, as specified by the algorithm. It is important that the local space required by the algorithm be proportional to the size of the largest subset Spark partition.

3) Write a program GxxxHW1.java (for Java users) or GxxxHW1.py (for Python users), where xxx is your 3-digit group number (e.g., 004 or 045), which receives in input, as command-line arguments, 2 integers C
and R, and a path to the file storing the input graph, and does the following:

    Reads parameters C and R
Reads the input graph into an RDD of strings (called rawData) and transform it into an RDD of edges (called edges), represented as pairs of integers, partitioned into C
partitions, and cached.
Prints: the name of the file, the number of edges of the graph, C
, and R
.
Runs R
times MR_ApproxTCwithNodeColors to get R independent estimates tfinal
of the number of triangles in the input graph.
Prints: the median of the R
estimates returned by MR_ApproxTCwithNodeColors and the average running time of MR_ApproxTCwithNodeColors over the R
runs.
Runs MR_ApproxTCwithSparkPartitions to get an estimate tfinal

    of the number of triangles in the input graph.
    Prints: the estimate returned by MR_ApproxTCwithSparkPartitions and its running time.

File OutputFBsmallC2R5.txt shows you how to format your output. Make sure that your program complies with this format.

4) Test your program using the datasets that we provide in the same section as this page, together with the outputs of some runs of our program on the datasets. Note that while using C=1
should give the exact count of triangles (unique for each graph), using C>1 provides only an approximate count. So, for C=1 your counts should be the same as ours, but for C>1 they may differ. Fill the table given in this word file HW1-Table.docx with the required values obtained when testing your program. '''



from pyspark import SparkContext, SparkConf
from CountTriangles import CountTriangles
import sys
import os
import random as rand

def h_c(u, C):
    hash_value = ((a * u + b) % p) % C
    return hash_value

def MR_ApproxTCwithNodeColors(edges, C):
	"""
	First Algorithm.
		
	Args:
		edges: RDD of edges.
		C: number of colors.
		
	Returns:
		int: number of triangles.
	"""
	#	ROUND 1:
	#Create a new RDD where each element is a key-value pair, with: 
 	#		key: result of the hash function applied to the first node of the edge.
	# 		value: original element itself
	#then filter out the None values that are created when the two nodes of an edge have different colors.
	colored_edges = edges.flatMap(lambda x: [(h_c(x[0], C), x)] if (h_c(x[0], C) == h_c(x[1], C)) else []) # <-- MAP PHASE (R1)

	#group the edges by color, thus by the same key, and return pairs (key, list(edge)).
	E_i = colored_edges.groupByKey() # <-- SHUFFLE+GROUPING (R1)
	
	#count the number of triangles in each subset E_i.
	#Apply CountTriangle() to each list v in the RDD and return a new RDD with (k, number)
	T_i = E_i.mapValues(CountTriangles)  # <-- REDUCE PHASE (R1)
	
 	#we use map instead of flatMap because CountTriangles returns a list of one element, and also
	# we don't need to use reduceByKey because we don't need to sum the values of the same key.
	#TODO da controllare
	
	#	ROUND 2:
	#T_final = C^2 ∑ T_i(i) as final estimate of the number of triangles in G.
	
	#reduce(lambda x, y: (x[0], x[1] + y[1]))
	#t_2 = T_i.reduceByKey(lambda x, y: (0, x[1] + y[1]))
	#t_2 = T_i.reduce(lambda x, y: x+y )
	
	#T_i.reduceByKey(lambda x, y: x + y).map(lambda x: x[1]).sum()
 
	#print(t_2.collect())
	#print(str(t_2))
 
	#T_final = C**2 * T_i.reduce(lambda x, y: (x[0], x[1] + y[1]))
	T_final = C**2 * T_i.map(lambda x: x[1]).reduce(lambda x, y: x + y) # <-- REDUCE PHASE (R2)
	print("Number of triangles (alg 1): " + str(T_final))


def MR_ApproxTCwithSparkPartitions(edges):
	"""
	Second Algorithm.
		
	Args:
		edges: RDD of edges.
		
	Returns:
		int: number of triangles.
	"""
	print("prova")


def main():

	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 4, "Usage: python3 G090HW1.py <C> <R> <file_name>"

	# SPARK SETUP
	conf = SparkConf().setAppName('G090HW1')

	sc = SparkContext(conf=conf)


	# INPUT READING

	# 1. Reads parameters C and R
	C = sys.argv[1]
	assert C.isdigit() and int(C) >= 1, "K must be an integer and greater than 0"
	C = int(C)

	R = sys.argv[2]
	assert R.isdigit() and int(R) >= 1, "K must be an integer and greater than 0"
	R = int(R)

	# 2. Reads the input graph into an RDD of strings (called rawData) and transform it into an RDD 
  	#	of edges (called edges), represented as pairs of integers, partitioned into C partitions, and cached
	data_path = sys.argv[3]
	assert os.path.isfile(data_path), "File or folder not found"
	
	rawData = sc.textFile(data_path, minPartitions = C).cache()#.cache lo mettiamo anche qui ??
	edges = rawData.map(lambda x: tuple(map(int, x.split(',')))).repartition(numPartitions = C).cache()
 
	#TODO # create an RDD from the list of edges ( sto punto sopra non so se sia giusto)
	#rdd = sc.parallelize(edges) IN CERTI CODICI VEDO QUESTO, NO SO SE SERVE
 
 	# 3. Prints: the name of the file, the number of edges of the graph, C, and R.
	print("File name: ", data_path)
	print("Number of edges: ", edges.count())
	print("C: ", C)
	print("R: ", R)
 
	#print(edges.collect()) to print the edges in the RDD
 
	# 4. Runs R times MR_ApproxTCwithNodeColors to get R independent estimates tfinal of the number of triangles in the input graph.
	for i in range(R):
		MR_ApproxTCwithNodeColors(edges, C)

	# 5. Prints: the median of the R estimates returned by MR_ApproxTCwithNodeColors and the average running time of MR_ApproxTCwithNodeColors over the R runs.
 
	# 6. Runs MR_ApproxTCwithSparkPartitions to get an estimate tfinal of the number of triangles in the input graph.

	# 7. Prints: the estimate returned by MR_ApproxTCwithSparkPartitions and its running time.
 
	sc.stop()

 
if __name__ == "__main__":
	p = 8191 # prime number
	a = rand.randint(1, p-1)
	b = rand.randint(0, p-1)
	main()
