import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import scala.Tuple3;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;


public class G090HW2 {
  public static void main(String[] args) throws FileNotFoundException {
    if (args.length < 4) {
      throw new IllegalArgumentException("Please provide <C> <R> <F> <file_path> as arguments");
    }

    // Acquiring parameters
    int C = Integer.parseInt(args[0]);
    int R = Integer.parseInt(args[1]);
    int F = Integer.parseInt(args[2]);
    String file_path = args[3];

    //checking integer parameters
    if (C <= 0 || R <= 0 || F < 0 || F > 1) {
      throw new IllegalArgumentException("Invalid argument, C and R must be positive, F must be between 0 and 1");
    }

    // Checking if file_path is a valid one
    File f = new File(file_path);
    if (!f.exists() || f.isDirectory()) {
      throw new FileNotFoundException("Invalid argument, file not found");
    }

    // Initializing spark context
    SparkConf conf = new SparkConf(true).setAppName("G090HW2");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("WARN");

    // Setting up JavaRDD
    JavaRDD<String> rawData = sc.textFile(file_path);
    JavaPairRDD<Integer, Integer> edges;

    edges = rawData.flatMapToPair(
      (document) -> {
        String[] tokens = document.split(",");
        ArrayList<Tuple2<Integer, Integer>> pairs = new ArrayList<>();

        for (int i = 0; i < tokens.length; i+=2) {
          int vertex1 = Integer.parseInt(tokens[i]);
          int vertex2 = Integer.parseInt(tokens[i+1]);
          pairs.add(new Tuple2<>(vertex1, vertex2));
        }

        return pairs.iterator();
      });
    edges.repartition(C).cache();

    //TODO CHANGE

    // Deciding a and b parameters
    int p = 8191;
    Random rnd = new Random();
    long a = rnd.nextInt(p - 1) + 1;
    long b = rnd.nextInt(p);

    // Printing Data
    System.out.printf("Dataset = %s\n", file_path);
    System.out.printf("Number of Edges = %d\n", edges.count());
    System.out.printf("Number of Colors = %d\n", C);
    System.out.printf("Number of repetitions = %d\n", R);
    //System.out.printf("Parameters (p,a,b) -> (%d,%d,%d)\n", p, a, b);


    if(F == 0){
      // Approximation through node coloring
      System.out.println("Approximation through node coloring");
      ArrayList<Long> results = new ArrayList<>();
      ArrayList<Long> timestamps = new ArrayList<>();
      for (int i = 0; i < R; i++) {
          long startTime = System.currentTimeMillis();
          results.add(MR_ApproxTCwithNodeColors(edges, C, a, b, p));
          long endTime = System.currentTimeMillis();
          timestamps.add(endTime - startTime);
      }
      Collections.sort(results);

      // Computing median value
      long median = -1;
      if (results.size() % 2 == 0) {
          int left_index = (results.size() / 2) - 1;
          long left_el = results.get(left_index);
          long right_el = results.get(left_index + 1);
          median = (left_el + right_el ) / 2;
      } else {
          median = results.get((results.size() - 1) / 2);
      }

      // Computing average time
      long avg = 0l;
      for (long time : timestamps) {
          avg += time;
      }
      avg = avg / timestamps.size();
      System.out.printf("- Number of triangles (median over %d runs) = %d\n", R, median);
      System.out.printf("- Running time (average over %d runs) = %dms\n", R, avg);

    }else{
      edges.repartition(C).cache();
      ArrayList<Long> timestamps = new ArrayList<>();
      long res = 0;
      for (int i = 0; i < R; i++) {
        long startTime = System.currentTimeMillis();
        res = MR_ExactTC(edges, C, a, b, p);
        long endTime = System.currentTimeMillis();
        timestamps.add(endTime - startTime);
      }

      // Computing average time
      long avg = 0l;
      for (long time : timestamps) {
        avg += time;
      }
      avg = avg / timestamps.size();
      System.out.printf("Number of triangles exact: %d\n", res);
      System.out.printf("- Running time (average over %d runs) = %dms\n", R, avg);
    }
  }

  public static Long MR_ExactTC(JavaPairRDD<Integer, Integer> edges, int C, long a, long b, long p) {
    JavaPairRDD<Integer, Long> round1 = edges
        .flatMapToPair(
            (pair) -> {
              ArrayList<Tuple2<Tuple3<Integer, Integer, Integer>, Tuple2<Integer, Integer>>> tuples = new ArrayList<>();
              for (int i = 0; i < C; i++) {
                int hash1 = (int)((a * pair._1() + b) % p) % C;
                int hash2 = (int)((a * pair._2() + b) % p) % C;
                int hash3 = i;

                ArrayList<Integer> hash = new ArrayList<>();
                hash.add(hash1);
                hash.add(hash2);
                hash.add(hash3);
                Collections.sort(hash);

                Tuple3<Integer, Integer, Integer> key = new Tuple3<>(hash.get(0), hash.get(1), hash.get(2));

                tuples.add(new Tuple2<>(key, pair));
              }

              return tuples.iterator();
            }
        )
        .groupByKey()
        .flatMapToPair(
            (key_value) -> {
              ArrayList<Tuple2<Integer, Integer>> list = new ArrayList<>();
              for (Tuple2<Integer, Integer> el : key_value._2) {
                list.add(el);
              }
              long triangles = CountTriangles2(list, key_value._1, a, b, p, C);

              ArrayList<Tuple2<Integer, Long>> partial_result = new ArrayList<>();
              partial_result.add(new Tuple2<>(0, triangles));
              return partial_result.iterator();
            }
        ).cache();

    JavaPairRDD<Integer, Long> round2 = round1
        .groupByKey()
        .mapValues(
            (list) -> {
              long sum = 0;
              for (long num : list) {
                sum += num;
              }
              return sum;
            }
        ).cache();

    long result = round2.cache().collect().get(0)._2;

    return result;
  }

  public static Long MR_ApproxTCwithNodeColors(JavaPairRDD<Integer, Integer> edges, int C, long a, long b, long p) {

    JavaPairRDD<Integer, Long> round1 = edges.flatMapToPair(
            (pair) -> {
              ArrayList<Tuple2<Long, Tuple2<Integer, Integer>>> list = new ArrayList<>();

              long hash1 = ((a * pair._1() + b) % p) % C;
              long hash2 = ((a * pair._2() + b) % p) % C;

              if (hash1 == hash2) {
                Tuple2<Long, Tuple2<Integer, Integer>> tuple = new Tuple2<>(hash1, pair);
                list.add(tuple);
              }

              return list.iterator();
            })
        .groupByKey()
        .flatMapToPair(
            (pair) -> {
              ArrayList<Tuple2<Integer, Integer>> list = new ArrayList<>();
              for (Tuple2<Integer, Integer> tuple : pair._2) {
                list.add(tuple);
              }

              ArrayList<Tuple2<Integer, Long>> final_count = new ArrayList<>();
              final_count.add(new Tuple2<>(0, CountTriangles(list)));

              return final_count.iterator();
            }
        ).cache();

    JavaPairRDD<Integer, Long> round2 = round1
        .groupByKey()
        .mapValues(
            (list) -> {
              long sum = 0;
              for (long el : list) {
                sum += el;
              }
              return Double.valueOf(Math.pow(C, 2) * sum).longValue();
            }
        ).cache();

    long result = round2.cache().collect().get(0)._2;

    return result;
  }

  /**
   * Counts the number of triangles TODO.
   * @param edgeSet The set of edges.
   * @return The number of triangles.
   */
  public static Long CountTriangles(ArrayList<Tuple2<Integer, Integer>> edgeSet) {
    if (edgeSet.size()<3) return 0L;
    HashMap<Integer, HashMap<Integer,Boolean>> adjacencyLists = new HashMap<>();
    for (int i = 0; i < edgeSet.size(); i++) {
      Tuple2<Integer,Integer> edge = edgeSet.get(i);
      int u = edge._1();
      int v = edge._2();
      HashMap<Integer,Boolean> uAdj = adjacencyLists.get(u);
      HashMap<Integer,Boolean> vAdj = adjacencyLists.get(v);
      if (uAdj == null) {uAdj = new HashMap<>();}
      uAdj.put(v,true);
      adjacencyLists.put(u,uAdj);
      if (vAdj == null) {vAdj = new HashMap<>();}
      vAdj.put(u,true);
      adjacencyLists.put(v,vAdj);
    }
    Long numTriangles = 0L;
    for (int u : adjacencyLists.keySet()) {
      HashMap<Integer,Boolean> uAdj = adjacencyLists.get(u);
      for (int v : uAdj.keySet()) {
        if (v>u) {
          HashMap<Integer,Boolean> vAdj = adjacencyLists.get(v);
          for (int w : vAdj.keySet()) {
            if (w>v && (uAdj.get(w)!=null)) numTriangles++;
          }
        }
      }
    }
    return numTriangles;
  }

  /**
   * Counts the number of triangles in a given set of edges using the exact algorithm with color filtering.
   * @param edgeSet The set of edges.
   * @param key The color combination.
   * @param a The parameter a.
   * @param b The parameter b.
   * @param p The parameter p.
   * @param C The number of colors.
   * @return The number of triangles.
   */
  public static Long CountTriangles2(ArrayList<Tuple2<Integer, Integer>> edgeSet, Tuple3<Integer, Integer, Integer> key, long a, long b, long p, int C) {
    if (edgeSet.size()<3) return 0L;
    HashMap<Integer, HashMap<Integer,Boolean>> adjacencyLists = new HashMap<>();
    HashMap<Integer, Integer> vertexColors = new HashMap<>();
    for (int i = 0; i < edgeSet.size(); i++) {
      Tuple2<Integer,Integer> edge = edgeSet.get(i);
      int u = edge._1();
      int v = edge._2();
      if (vertexColors.get(u) == null) {vertexColors.put(u, (int) ((a*u+b)%p)%C);}
      if (vertexColors.get(v) == null) {vertexColors.put(v, (int) ((a*v+b)%p)%C);}
      HashMap<Integer,Boolean> uAdj = adjacencyLists.get(u);
      HashMap<Integer,Boolean> vAdj = adjacencyLists.get(v);
      if (uAdj == null) {uAdj = new HashMap<>();}
      uAdj.put(v,true);
      adjacencyLists.put(u,uAdj);
      if (vAdj == null) {vAdj = new HashMap<>();}
      vAdj.put(u,true);
      adjacencyLists.put(v,vAdj);
    }
    Long numTriangles = 0L;
    for (int u : adjacencyLists.keySet()) {
      HashMap<Integer,Boolean> uAdj = adjacencyLists.get(u);
      for (int v : uAdj.keySet()) {
        if (v>u) {
          HashMap<Integer,Boolean> vAdj = adjacencyLists.get(v);
          for (int w : vAdj.keySet()) {
            if (w>v && (uAdj.get(w)!=null)) {
              ArrayList<Integer> tcol = new ArrayList<>();
              tcol.add(vertexColors.get(u));
              tcol.add(vertexColors.get(v));
              tcol.add(vertexColors.get(w));
              Collections.sort(tcol);
              boolean condition = (tcol.get(0).equals(key._1())) && (tcol.get(1).equals(key._2())) && (tcol.get(2).equals(key._3()));
              if (condition) {numTriangles++;}
            }
          }
        }
      }
    }
    return numTriangles;
  }

}
