from pyspark import SparkContext, SparkConf
from CountTriangles import CountTriangles
import sys
import os
import random as rand
import time

def h_c(u, a, b, C):
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
	a = rand.randint(1, p-1)
	b = rand.randint(0, p-1)
	#	ROUND 1:
	#Create a new RDD where each element is a key-value pair, with: 
 	#		key: result of the hash function applied to the first node of the edge.
	# 		value: original element itself
	#then filter out the None values that are created when the two nodes of an edge have different colors.
	colored_edges = edges.flatMap(lambda x: [(h_c(x[0], a, b, C), x)] if (h_c(x[0], a, b, C) == h_c(x[1], a, b, C)) else []) # <-- MAP PHASE (R1)

	#group the edges by color, thus by the same key, and return pairs (key, list(edge)).
	E_i = colored_edges.groupByKey() # <-- SHUFFLE+GROUPING (R1)
	
	#count the number of triangles in each subset E_i.
	#Apply CountTriangle() to each list v in the RDD and return a new RDD with (k, number)
	T_i = E_i.mapValues(CountTriangles)  # <-- REDUCE PHASE (R1)
	
	#	ROUND 2:
	#T_final = C^2 ∑ T_i(i) as final estimate of the number of triangles in G.
	return C**2 * T_i.map(lambda x: x[1]).reduce(lambda x, y: x + y) # <-- REDUCE PHASE (R2)
	


def MR_ApproxTCwithSparkPartitions(edges, C):
	"""
	Second Algorithm.
		
	Args:
		edges: RDD of edges.
		
	Returns:
		int: number of triangles.
	"""
 	#	ROUND 1:
	#Partition the edges at random into C subsets E(0),E(1),...E(C−1)
	#We don't need a map phase because the RDD is already into a C partitions.
	
	#Compute the partial counts T(0),T(1),...,T(C−1) of triangles in each subset E(i)
	T_i = edges.mapPartitions(lambda partition: [CountTriangles(partition)]) # <-- REDUCE PHASE (R1)

	#	ROUND 2:
 	#Compute the total count
	return C**2 * T_i.reduce(lambda x, y: x + y)# <-- REDUCE PHASE (R2)
	



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
	
	#Reads the input graph into an RDD of strings (called rawData)
	rawData = sc.textFile(data_path, minPartitions = C)
	#and transform it into an RDD of edges (called edges), represented as pairs of integers, partitioned into C partitions, and cached.
	edges = rawData.map(lambda x: tuple(map(int, x.split(',')))).repartition(numPartitions = C).cache()
  
 	# 3. Prints: the name of the file, the number of edges of the graph, C, and R.
	print("Dataset = ", data_path)
	print("Number of Edges = ", edges.count())
	print("Number of Colors = ", C)
	print("Number of Repetitions = ", R)
	#print(edges.collect()) to print the edges in the RDD
 
	# 4. Runs R times MR_ApproxTCwithNodeColors to get R independent estimates tfinal of the number of triangles in the input graph.
	print("Approximation through node coloring")
	sum = 0
	time_sum = 0
	for i in range(R):	
		start_time = time.time()
		sum += MR_ApproxTCwithNodeColors(edges, C)
		time_sum += time.time() - start_time

	# 5. Prints: the median of the R estimates returned by MR_ApproxTCwithNodeColors and the average running time of MR_ApproxTCwithNodeColors over the R runs.
	print("- Number of triangles (median over "+ str(R) +" runs) = " + str(int(sum/R)))
	print("- Running time (average over "+ str(R) +" runs) = " + str(int(time_sum*1000/R)) + " ms")
	
		
	# 6. Runs MR_ApproxTCwithSparkPartitions to get an estimate tfinal of the number of triangles in the input graph.
	print("Approximation through Spark partitions")
	start_time = time.time()
	tot = MR_ApproxTCwithSparkPartitions(edges, C)
	time_sum = time.time() - start_time
  
	# 7. Prints: the estimate returned by MR_ApproxTCwithSparkPartitions and its running time.
	print("- Number of triangles = " + str(tot))
	print("- Running time = " + str(int(time_sum*1000)) + " ms")
	
	sc.stop()



 
if __name__ == "__main__":
	p = 8191 # prime number
	main()
