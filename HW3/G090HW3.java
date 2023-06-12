package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.util.*;
import java.util.concurrent.Semaphore;

public class G090HW3 {

    // After how many items should we stop?
    public static final int THRESHOLD = 10000000;

    /*public int hash_function(int a, int b, int p, int C, int x){
        return ((a * x + b) % p) % C;
    }*/
    /*
     @param a a value
      @param b b value
      @param p p value
      @param C C value
      @param x integer element
      @param normal if true, return a normalized value in {-1,1}, otherwise return a value in [0,C-1]
     */
    public static int hash_function(long a, long b, int p, int C, int x, boolean normal) {
        double result = ((a * x + b) % p);

        if (normal) {
            double normalizedResult = 2.0 * result / (p - 1) - 1.0;
            if (normalizedResult >= 0) {
                return 1;
            } else {
                return -1;
            }
        } else {
            return (int) (result % C);
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 6) {
            throw new IllegalArgumentException("USAGE: <D> <W> <left> <right> <K> <portExp>");
        }


        // IMPORTANT: when running locally, it is *fundamental* that the
        // `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
        // there will be no processor running the streaming computation and your
        // code will crash with an out of memory (because the input keeps accumulating).
        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]") // remove this line if running on the cluster
                .setAppName("DistinctExample");

        // Here, with the duration you can control how large to make your batches.
        // Beware that the data generator we are using is very fast, so the suggestion
        // is to use batches of less than a second, otherwise you might exhaust the
        // JVM memory.
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        // TECHNICAL DETAIL:
        // The streaming spark context and our code and the tasks that are spawned all
        // work concurrently. To ensure a clean shut down we use this semaphore.
        // The main thread will first acquire the only permit available and then try
        // to acquire another one right after spinning up the streaming computation.
        // The second tentative at acquiring the semaphore will make the main thread
        // wait on the call. Then, in the `foreachRDD` call, when the stopping condition
        // is met we release the semaphore, basically giving "green light" to the main
        // thread to shut down the computation.
        // We cannot call `sc.stop()` directly in `foreachRDD` because it might lead
        // to deadlocks.
        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        int D = Integer.parseInt(args[0]);
        int W = Integer.parseInt(args[1]);
        int left = Integer.parseInt(args[2]);
        int right = Integer.parseInt(args[3]);
        int K = Integer.parseInt(args[4]);
        int portExp = Integer.parseInt(args[5]);
        // Display the input format
        System.out.println("D = " + D + " W = " + W + " [left, right] = [" + left + ", " + right + "] K = " + K + " Port = " + portExp);


        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long[] streamLength = new long[1]; // Stream length (an array to be passed by reference)
        streamLength[0]=0L;
        HashMap<Long, Long> histogram = new HashMap<>(); // Hash Table for the distinct elements ( with their true counts)

        // HASH FUNCTIONS:
        // We use the same hash functions with different parameters (a, b)
        // p=8191, C depends on W (C = W+1 to get the correct range of columns)
        //a end b = [1 , p-1]
        int p = 8191;
        int C = W;
        Random rnd = new Random();
        long a = rnd.nextInt(p - 1) + 1;
        long b = rnd.nextInt(p);
        //parameters for g(x)
        long g_a = rnd.nextInt(p - 1) + 1;
        long g_b = rnd.nextInt(p);

        // Pair a,b for the hash functions
        Tuple2<Long, Long> a_b_param = new Tuple2<>(0l, 0l);


        //now to use Hj(x) = hash_function(a,b,p,C,x,false)
        //now to use Gj(x) = hash_function(g_a,g_b,p,C,x,true)

        //MATRIX C (DxW)
        //List<Tuple2<Tuple2<Integer, Integer>, Long>> matrixC = new ArrayList<>();
        long[][] matrixC = new long[D][W];
        //Init the matrix
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < W; j++) {
                matrixC[i][j] = 0;
            }
        }


        // CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                // For each batch, to the following.
                // BEWARE: the `foreachRDD` method has "at least once semantics", meaning
                // that the same data might be processed multiple times in case of failure.
                .foreachRDD((batch, time) -> {
                    // this is working on the batch at time `time`.
                    if (streamLength[0] < THRESHOLD) {
                        long batchSize = batch.count();
                        streamLength[0] += batchSize;

                        //Exact values
                        List<Tuple2<Long, Long>> batchItems = batch
                                .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L))
                                .groupByKey()
                                .mapValues((list) -> {//   REDUCE PHASE
                                    long sum = 0;
                                    for (long num : list) {
                                        sum += num;
                                    }
                                    return sum;
                                })
                                .collect();


                        // Update the number of exact distinct elements
                        for (Tuple2<Long, Long> pair : batchItems) {
                            if(pair._1 > left && pair._1 < right){//in ΣR
                                if (!histogram.containsKey(pair._1)) {
                                    histogram.put(pair._1, pair._2);
                                }else{//update
                                    histogram.replace(pair._1, histogram.get(pair._1), histogram.get(pair._1) + pair._2);
                                }
                            }
                        }
                        // If we wanted, here we could run some additional code on the global histogram
                        if (batchSize > 0) {
                            System.out.println("Batch size at time [" + time + "] is: " + batchSize);
                        }

                        // Update the matrix C foreach element of the batchitems
                        // foreach element of the batch
                        for (Tuple2<Long, Long> pair : batchItems) {
                            //foreach row of the matrix
                            for(int row = 0; row<D; row++){
                                //C[row, Hj(x)] += Gj(x) * number of occurrences of x in the batch
                                int hjx=(int) hash_function(a,b, p, C, pair._1.intValue(), false);
                                long gjx = hash_function(g_a, g_b, p, C, pair._1.intValue(), true);

                                matrixC[row][hjx] += gjx * pair._2;
                            }
                        }


                        if (streamLength[0] >= THRESHOLD) {
                            stoppingSemaphore.release();
                        }
                    }
                });

        // MANAGING STREAMING SPARK CONTEXT
        System.out.println("Starting streaming engine");
        sc.start();
        System.out.println("Waiting for shutdown condition");
        stoppingSemaphore.acquire();
        System.out.println("Stopping the streaming engine");
        // NOTE: You will see some data being processed even after the
        // shutdown command has been issued: This is because we are asking
        // to stop "gracefully", meaning that any outstanding work
        // will be done.
        sc.stop(false, false);
        System.out.println("Streaming engine stopped");

        // COMPUTE AND PRINT FINAL STATISTICS
        System.out.println("Number of items processed = " + streamLength[0]);

        // TRUE SECOND MOMENT
        double F2 = 0;
        for (Long individual_freq : histogram.values()){//frequency of each element
            F2 += Math.pow(individual_freq, 2);
        }
        // Normalization of the true second moment
        F2 = F2 / Math.pow(histogram.size(), 2);
        System.out.println("True second moment = " + F2);

        // APPROXIMATE SECOND MOMENT
        double[] F2_approx = new double[D];
        double median = 0;
        for (int i = 0; i < D; i++){
            for (int j = 0; j < W; j++){
                F2_approx[i] += Math.pow(matrixC[i][j], 2);
            }
        }

        if (F2_approx.length % 2 == 0)
            median = (F2_approx[F2_approx.length / 2] + F2_approx[F2_approx.length / 2 - 1]) / 2;
        else
            median = F2_approx[F2_approx.length / 2];

        median = median / Math.pow(histogram.size(), 2);
        System.out.println("Approximate second moment = " + median);


        //NON HO IDEA DI COSA SIA QUESTO

        /*
        // AVERAGE RELATIVE ERROR
        //Sorting true frequencies
        ArrayList<Long> sorted_trueFreq=(ArrayList<Long>)histogram.values();
        Collections.sort(sorted_trueFreq,Collections.reverseOrder());

        //K-th largest frequency of the items of SigmaR
        Long k_th_freq=sorted_trueFreq.get(K);

        //fu <-- histogram
        //phi(K)<--k_th_freq
        //fu_approx <-- median of the fu,j's
        double[] relative_errors=new double[K];
        int k=0;
        for (Map.Entry<Long,Long> e : histogram.entrySet()) {
            if (e.getValue() > k_th_freq) {
                ArrayList<Tuple2<Long, Long>> f = data.get(e.getKey());
                double[] temp_freq = new double[D];
                int count = 0;
                for (Tuple2<Long, Long> v : f) {
                    int col = v._1().intValue();
                    temp_freq[count] = matrixC[count][col] * v._2();
                }
                double freq_median=0;
                double relative_error=0;
                if (temp_freq.length % 2 == 0)
                    freq_median = (temp_freq[temp_freq.length / 2] + temp_freq[temp_freq.length / 2 - 1]) / 2;
                else
                    freq_median = temp_freq[temp_freq.length / 2];

                freq_median = freq_median / Math.pow(histogram.size(), 2);

                relative_errors[k++]=(e.getValue()-freq_median)/e.getValue();
            }
        }

        double m=0;
        for(double c:relative_errors){
            m+=c;
        }
        double average=m/relative_errors.length;
        System.out.println("Average Relative Error: "+ average);
        */

        long max = 0L;
        for (Long key : histogram.keySet()) {
            if (key > max) {max = key;}
        }
        System.out.println("Largest item = " + max);
    }
}

