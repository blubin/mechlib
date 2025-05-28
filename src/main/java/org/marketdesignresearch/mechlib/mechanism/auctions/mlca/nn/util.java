//package com.auction.helpers;

import java.time.Duration;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.concurrent.ThreadLocalRandom;

import ilog.concert.*;
import ilog.cplex.*;
import org.tensorflow.*;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.types.TFloat32;


/*

Claude Sonnet 4 Ported from https://github.com/marketdesignresearch/DL-ICA/blob/master/source/util.py

Claude Prompt: "Port the following Python to Java using the CPLEX library for MIP formulation and Tensorflow-Keras library for neural network processing."

Claude's Notes:

Key Changes and Additions:

Libraries Used:

- CPLEX: ilog.concert.* and ilog.cplex.* for MIP formulation
- TensorFlow Java: org.tensorflow.* for neural network processing
- Standard Java libraries for collections and utilities

Data Structures:

- BidData class to represent X (bundles) and Y (values) pairs
- MinMaxScaler class to replicate scikit-learn's functionality
- AllocationResult to represent auction outcomes
- ValueModel interface for auction value models

Core Functions Ported:

- timediffDHMS() - Time difference calculation
- keyToInt() - Extract bidder ID from string
- initialBidsMLCAUnif() - MLCA mechanism bid preparation
- initialBidsPVMUnif() - PVM uniform sampling
- initialBidsPVM() - PVM with normal distribution sampling
- formatSolutionMIP() - CPLEX solution formatting
- unifRandomBids() - Uniform random bundle generation

CPLEX Integration:

- formatSolutionMIP() now works with CPLEX's IloCplex objects
- Added helper method createWDPModel() for WDP formulation
- Variable naming convention for easier solution parsing

Java-Specific Improvements:

- Type safety with generics
- Proper exception handling with IloException
- Stream API usage for functional programming
- Thread-safe random number generation

Dependencies Required:
xml<!-- Add to your pom.xml -->
<dependency>
    <groupId>com.ibm.ilog</groupId>
    <artifactId>cplex</artifactId>
    <version>22.1.0</version>
</dependency>
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow-core-platform</artifactId>
    <version>0.5.0</version>
</dependency>
The code maintains the same algorithmic logic while leveraging Java's type system and CPLEX/TensorFlow Java APIs. You'll need to implement the ValueModel interface based on your specific auction model (SATS equivalent in Java).
*/

/**
 * FILE DESCRIPTION:
 * 
 * This file stores helper functions used across the files in this project.
 * Ported from Python to Java using CPLEX and TensorFlow Java libraries.
 * 
 * @author Jakob Weissteiner (Original Python), Ported to Java
 * @copyright Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions
 * @license AGPL-3.0
 * @version 0.1.0
 * @status Dev
 */
public class AuctionHelpers {
    
    private static final Logger logger = Logger.getLogger(AuctionHelpers.class.getName());
    
    // Data structures for storing bids
    public static class BidData {
        public double[][] X; // Bundle vectors
        public double[] Y;   // Values
        
        public BidData(double[][] X, double[] Y) {
            this.X = X;
            this.Y = Y;
        }
    }
    
    public static class TimeComponents {
        public final long days;
        public final int hours;
        public final int minutes;
        public final int seconds;
        
        public TimeComponents(long days, int hours, int minutes, int seconds) {
            this.days = days;
            this.hours = hours;
            this.minutes = minutes;
            this.seconds = seconds;
        }
    }
    
    public static class MinMaxScaler {
        private double dataMin;
        private double dataMax;
        private double scale;
        private int nSamplesSeen;
        private double featureRangeMin = 0.0;
        private double featureRangeMax = 1.0;
        
        public void fit(double[] data) {
            if (data.length == 0) return;
            
            dataMin = Arrays.stream(data).min().orElse(0.0);
            dataMax = Arrays.stream(data).max().orElse(1.0);
            nSamplesSeen = data.length;
            
            if (dataMax - dataMin == 0) {
                scale = 1.0;
            } else {
                scale = (featureRangeMax - featureRangeMin) / (dataMax - dataMin);
            }
            
            logger.info("*SCALING*");
            logger.info("---------------------------------------------");
            logger.info("Samples seen: " + nSamplesSeen);
            logger.info("Data max: " + dataMax);
            logger.info("Data min: " + dataMin);
            logger.info("Scaling by: " + scale + " | " + (dataMax * scale) + "==feature range max?");
            logger.info("---------------------------------------------");
        }
        
        public double[] transform(double[] data) {
            return Arrays.stream(data)
                    .map(x -> (x - dataMin) * scale + featureRangeMin)
                    .toArray();
        }
        
        public double[] inverseTransform(double[] data) {
            return Arrays.stream(data)
                    .map(x -> (x - featureRangeMin) / scale + dataMin)
                    .toArray();
        }
        
        public double inverseTransformSingle(double value) {
            return (value - featureRangeMin) / scale + dataMin;
        }
        
        // Getters
        public double getScale() { return scale; }
        public double getDataMin() { return dataMin; }
        public double getDataMax() { return dataMax; }
        public int getNSamplesSeen() { return nSamplesSeen; }
    }
    
    public interface ValueModel {
        double calculateValue(int bidderId, int[] bundle);
        List<int[]> getGoodIds();
        List<double[]> getRandomBids(int bidderId, int numberOfBids, Long seed);
        List<double[]> getRandomBids(int bidderId, int numberOfBids);
    }
    
    public static class AllocationResult {
        public final List<Integer> goodIds;
        public final double value;
        
        public AllocationResult(List<Integer> goodIds, double value) {
            this.goodIds = goodIds;
            this.value = value;
        }
    }
    
    /**
     * Converts time difference to days, hours, minutes, seconds
     * Can handle negative time differences
     */
    public static TimeComponents timediffDHMS(Duration td) {
        if (td.isNegative()) {
            td = td.abs();
            long days = td.toDays();
            int hours = (int) (td.toHours() % 24);
            int minutes = (int) (td.toMinutes() % 60);
            int seconds = (int) (td.getSeconds() % 60);
            return new TimeComponents(-days, -hours, -minutes, -seconds);
        }
        
        long days = td.toDays();
        int hours = (int) (td.toHours() % 24);
        int minutes = (int) (td.toMinutes() % 60);
        int seconds = (int) (td.getSeconds() % 60);
        return new TimeComponents(days, hours, minutes, seconds);
    }
    
    /**
     * Transforms bidder_key to integer bidder_id
     * @param key valid bidder_key (string), e.g. 'Bidder_0'
     * @return bidder id as integer
     */
    public static int keyToInt(String key) {
        Pattern pattern = Pattern.compile("\\d+");
        Matcher matcher = pattern.matcher(key);
        if (matcher.find()) {
            return Integer.parseInt(matcher.group());
        }
        throw new IllegalArgumentException("No number found in key: " + key);
    }
    
    /**
     * PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS for MLCA MECHANISM
     * THIS METHOD USES TRUE UNIFORM SAMPLING!
     */
    public static Map<String, BidData> initialBidsMLCAUnif(
            ValueModel satsAuctionInstance, 
            int numberInitialBids, 
            List<String> bidderNames, 
            MinMaxScaler scaler) {
        
        Map<String, BidData> initialBids = new LinkedHashMap<>();
        
        for (String bidder : bidderNames) {
            logger.info("Set up initial Bids for: " + bidder);
            double[][] D = unifRandomBids(satsAuctionInstance, keyToInt(bidder), numberInitialBids);
            
            // Add null bundle
            double[] nullBundle = new double[D[0].length];
            double[][] DWithNull = new double[D.length + 1][];
            System.arraycopy(D, 0, DWithNull, 0, D.length);
            DWithNull[D.length] = nullBundle;
            
            // Split into X (bundles) and Y (values)
            double[][] X = new double[DWithNull.length][DWithNull[0].length - 1];
            double[] Y = new double[DWithNull.length];
            
            for (int i = 0; i < DWithNull.length; i++) {
                System.arraycopy(DWithNull[i], 0, X[i], 0, DWithNull[i].length - 1);
                Y[i] = DWithNull[i][DWithNull[i].length - 1];
            }
            
            initialBids.put(bidder, new BidData(X, Y));
        }
        
        if (scaler != null) {
            // Collect all Y values for scaling
            List<Double> allValues = new ArrayList<>();
            for (BidData bidData : initialBids.values()) {
                for (double value : bidData.Y) {
                    allValues.add(value);
                }
            }
            double[] allValuesArray = allValues.stream().mapToDouble(Double::doubleValue).toArray();
            scaler.fit(allValuesArray);
            
            // Apply scaling
            Map<String, BidData> scaledBids = new LinkedHashMap<>();
            for (Map.Entry<String, BidData> entry : initialBids.entrySet()) {
                BidData original = entry.getValue();
                double[] scaledY = scaler.transform(original.Y);
                scaledBids.put(entry.getKey(), new BidData(original.X, scaledY));
            }
            initialBids = scaledBids;
        }
        
        return initialBids;
    }
    
    /**
     * PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS
     * THIS METHOD USES TRUE UNIFORM SAMPLING!
     */
    public static Map<String, BidData> initialBidsPVMUnif(
            ValueModel valueModel, 
            int c0, 
            List<Integer> bidderIds, 
            MinMaxScaler scaler) {
        
        Map<String, BidData> initialBids = new LinkedHashMap<>();
        
        for (int bidderId : bidderIds) {
            logger.info("Set up initial Bids for: Bidder_" + bidderId);
            double[][] D = unifRandomBids(valueModel, bidderId, c0);
            
            // Add zero bundle
            double[] nullBundle = new double[D[0].length];
            double[][] DWithNull = new double[D.length + 1][];
            System.arraycopy(D, 0, DWithNull, 0, D.length);
            DWithNull[D.length] = nullBundle;
            
            // Split into X and Y
            double[][] X = new double[DWithNull.length][DWithNull[0].length - 1];
            double[] Y = new double[DWithNull.length];
            
            for (int i = 0; i < DWithNull.length; i++) {
                System.arraycopy(DWithNull[i], 0, X[i], 0, DWithNull[i].length - 1);
                Y[i] = DWithNull[i][DWithNull[i].length - 1];
            }
            
            initialBids.put("Bidder_" + bidderId, new BidData(X, Y));
        }
        
        if (scaler != null) {
            // Collect all Y values for scaling
            List<Double> allValues = new ArrayList<>();
            for (BidData bidData : initialBids.values()) {
                for (double value : bidData.Y) {
                    allValues.add(value);
                }
            }
            double[] allValuesArray = allValues.stream().mapToDouble(Double::doubleValue).toArray();
            scaler.fit(allValuesArray);
            
            // Apply scaling
            Map<String, BidData> scaledBids = new LinkedHashMap<>();
            for (Map.Entry<String, BidData> entry : initialBids.entrySet()) {
                BidData original = entry.getValue();
                double[] scaledY = scaler.transform(original.Y);
                scaledBids.put(entry.getKey(), new BidData(original.X, scaledY));
            }
            initialBids = scaledBids;
        }
        
        return initialBids;
    }
    
    /**
     * PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS
     * THIS METHOD USES RANDOM SAMPLING OF BUNDLES FROM SATS VIA NORMAL DISTRIBUTION!
     */
    public static Map<String, BidData> initialBidsPVM(
            ValueModel valueModel, 
            int c0, 
            List<Integer> bidderIds, 
            MinMaxScaler scaler, 
            Map<Integer, Long> seeds) {
        
        Map<String, BidData> initialBids = new LinkedHashMap<>();
        
        for (int bidderId : bidderIds) {
            logger.info("Set up initial Bids for: Bidder_" + bidderId);
            
            List<double[]> randomBids;
            if (seeds != null && seeds.containsKey(bidderId)) {
                randomBids = valueModel.getRandomBids(bidderId, c0, seeds.get(bidderId));
            } else {
                randomBids = valueModel.getRandomBids(bidderId, c0);
            }
            
            double[][] D = randomBids.toArray(new double[0][]);
            
            // Add zero bundle
            double[] nullBundle = new double[D[0].length];
            double[][] DWithNull = new double[D.length + 1][];
            System.arraycopy(D, 0, DWithNull, 0, D.length);
            DWithNull[D.length] = nullBundle;
            
            // Split into X and Y
            double[][] X = new double[DWithNull.length][DWithNull[0].length - 1];
            double[] Y = new double[DWithNull.length];
            
            for (int i = 0; i < DWithNull.length; i++) {
                System.arraycopy(DWithNull[i], 0, X[i], 0, DWithNull[i].length - 1);
                Y[i] = DWithNull[i][DWithNull[i].length - 1];
            }
            
            initialBids.put("Bidder_" + bidderId, new BidData(X, Y));
        }
        
        if (scaler != null) {
            // Collect all Y values for scaling
            List<Double> allValues = new ArrayList<>();
            for (BidData bidData : initialBids.values()) {
                for (double value : bidData.Y) {
                    allValues.add(value);
                }
            }
            double[] allValuesArray = allValues.stream().mapToDouble(Double::doubleValue).toArray();
            scaler.fit(allValuesArray);
            
            logger.info("Samples seen: " + scaler.getNSamplesSeen());
            logger.info("Data max: " + scaler.getDataMax());
            logger.info("Data min: " + scaler.getDataMin());
            logger.info("Scaling by: " + scaler.getScale() + " | " + 
                       (scaler.getDataMax() * scaler.getScale()) + " == feature range max?");
            
            // Apply scaling
            Map<String, BidData> scaledBids = new LinkedHashMap<>();
            for (Map.Entry<String, BidData> entry : initialBids.entrySet()) {
                BidData original = entry.getValue();
                double[] scaledY = scaler.transform(original.Y);
                scaledBids.put(entry.getKey(), new BidData(original.X, scaledY));
            }
            initialBids = scaledBids;
        }
        
        return initialBids;
    }
    
    /**
     * This function formats the solution of the winner determination problem (WDP) given elicited bids.
     * Uses CPLEX solution format
     */
    public static Map<String, AllocationResult> formatSolutionMIP(
            IloCplex cplex, 
            double[][][] elicitedBids, 
            List<String> bidderNames, 
            MinMaxScaler fittedScaler) throws IloException {
        
        Map<String, AllocationResult> Z = new LinkedHashMap<>();
        
        // Initialize with empty allocations
        for (String bidderName : bidderNames) {
            Z.put(bidderName, new AllocationResult(new ArrayList<>(), 0.0));
        }
        
        // Get solution variables (assuming binary variables named x_i_j for bidder i, bid j)
        IloNumVar[] vars = cplex.getVars();
        double[] solution = cplex.getValues(vars);
        
        Pattern pattern = Pattern.compile("x_(\\d+)_(\\d+)");
        
        for (int i = 0; i < vars.length; i++) {
            if (solution[i] > 0.5) { // Binary variable is set to 1
                String varName = vars[i].getName();
                Matcher matcher = pattern.matcher(varName);
                
                if (matcher.find()) {
                    int bidderIndex = Integer.parseInt(matcher.group(1));
                    int bidIndex = Integer.parseInt(matcher.group(2));
                    
                    if (bidderIndex < elicitedBids.length && bidIndex < elicitedBids[bidderIndex].length) {
                        double[] bidData = elicitedBids[bidderIndex][bidIndex];
                        double[] bundle = Arrays.copyOf(bidData, bidData.length - 1);
                        double value = bidData[bidData.length - 1];
                        
                        if (fittedScaler != null) {
                            logger.info("*SCALING*");
                            logger.info("---------------------------------------------");
                            logger.info("Original value: " + value);
                            logger.info("WDP values for allocation scaled by: 1/" + 
                                       Math.round(fittedScaler.getScale() * 100000000.0) / 100000000.0);
                            value = fittedScaler.inverseTransformSingle(value);
                            logger.info("Scaled value: " + value);
                            logger.info("---------------------------------------------");
                        }
                        
                        // Find which goods are included (bundle[i] == 1)
                        List<Integer> goodIds = new ArrayList<>();
                        for (int j = 0; j < bundle.length; j++) {
                            if (bundle[j] == 1.0) {
                                goodIds.add(j);
                            }
                        }
                        
                        String bidderName = bidderNames.get(bidderIndex);
                        Z.put(bidderName, new AllocationResult(goodIds, value));
                    }
                }
            }
        }
        
        return Z;
    }
    
    /**
     * This function generates bundle-value pairs for a single bidder sampled uniformly at random from the bundle space.
     */
    public static double[][] unifRandomBids(ValueModel valueModel, int bidderId, int n) {
        logger.info("Sampling uniformly at random " + n + " bundle-value pairs from bidder " + bidderId);
        
        int ncol = valueModel.getGoodIds().size(); // number of items in value model
        Set<String> uniqueBundles = new HashSet<>();
        List<int[]> bundles = new ArrayList<>();
        
        Random random = ThreadLocalRandom.current();
        
        // Generate unique random bundles
        while (bundles.size() < n) {
            int[] bundle = new int[ncol];
            StringBuilder bundleStr = new StringBuilder();
            
            for (int i = 0; i < ncol; i++) {
                bundle[i] = random.nextInt(2); // 0 or 1
                bundleStr.append(bundle[i]);
            }
            
            String bundleString = bundleStr.toString();
            if (!uniqueBundles.contains(bundleString)) {
                uniqueBundles.add(bundleString);
                bundles.add(bundle);
            }
        }
        
        // Calculate values for each bundle
        double[][] D = new double[n][ncol + 1];
        for (int i = 0; i < n; i++) {
            int[] bundle = bundles.get(i);
            double value = valueModel.calculateValue(bidderId, bundle);
            
            // Copy bundle to D
            for (int j = 0; j < ncol; j++) {
                D[i][j] = bundle[j];
            }
            D[i][ncol] = value; // Last column is the value
        }
        
        return D;
    }
    
    /**
     * Utility method to create CPLEX model for Winner Determination Problem
     * This is a basic structure - you'll need to adapt based on your specific WDP formulation
     */
    public static IloCplex createWDPModel(Map<String, BidData> bids) throws IloException {
        IloCplex cplex = new IloCplex();
        
        // Example structure - adapt based on your specific problem
        List<String> bidderNames = new ArrayList<>(bids.keySet());
        int numBidders = bidderNames.size();
        
        // Create decision variables for each bid
        Map<String, IloNumVar[]> variables = new HashMap<>();
        for (String bidder : bidderNames) {
            BidData bidData = bids.get(bidder);
            int numBids = bidData.Y.length;
            IloNumVar[] bidderVars = cplex.boolVarArray(numBids);
            
            // Name variables for easier solution parsing
            for (int j = 0; j < numBids; j++) {
                bidderVars[j].setName("x_" + bidderNames.indexOf(bidder) + "_" + j);
            }
            
            variables.put(bidder, bidderVars);
        }
        
        // Add constraints and objective here based on your specific WDP formulation
        // This is just a placeholder structure
        
        return cplex;
    }
}