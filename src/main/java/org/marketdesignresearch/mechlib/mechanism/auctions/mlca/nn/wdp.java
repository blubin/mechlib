import ilog.concert.*;
import ilog.cplex.*;
import java.util.*;
import java.util.logging.Logger;
import java.util.logging.Level;


/*

Claude Sonnet 4 Ported from https://github.com/marketdesignresearch/DL-ICA/blob/master/source/wdp.py

Claud's Notes:

Key Differences from Python Version:

- CPLEX Library: Uses ilog.concert.* and ilog.cplex.* instead of docplex.mp.model
- Data Structure: Uses a 3D array double[bidder][bid][item+value] instead of Python's list of numpy arrays
- Variable Management: Uses a HashMap<String, IloNumVar> to store decision variables with string keys
- Memory Management: Added a dispose() method to properly clean up CPLEX resources
- Error Handling: Uses Java exceptions and logging instead of Python's logging module

Main Features Preserved:

- Constructor: Takes bids as 3D array and initializes all necessary variables
- initializeMip(): Creates binary variables, allocation constraints, intersection constraints, and objective function
- solveMip(): Solves the optimization problem and extracts the optimal allocation
- summary(): Provides comprehensive output of solve status and results
- printOptimalAllocation(): Displays the allocation matrix in a formatted table

Additional Java-Specific Features:

- Resource Management: Proper cleanup with dispose()
- Getter Methods: getOptimalAllocation(), getObjectiveValue(), isSolved()
- Example Usage: Included in the main() method for testing
- Better Error Handling: Try-catch blocks for CPLEX exceptions

Usage:
java // Create WDP instance
double[][][] bids = {  your bid data  };
WDP wdp = new WDP(bids);

// Solve the problem
wdp.initializeMip(true);
wdp.solveMip(true);
wdp.summary();

// Clean up
wdp.dispose();

*/

/**
 * FILE DESCRIPTION:
 * 
 * This file implements the class WDP (Winner Determination Problem). This class is used for solving 
 * a winner determination problem given a finite sample of submitted XOR bids.
 * WDP has the following functionalities:
 *     0.CONSTRUCTOR: WDP(double[][][] bids)
 *        bids = 3D array representing elicited bundle-value pairs from each bidder. 
 *        bids[i][k][j] where i=bidder, k=bid, j=item (last column is value)
 *     1.METHOD: initializeMip(boolean verbose)
 *         verbose = boolean, level of verbosity when initializing the MIP for the logger.
 *         This method initializes the winner determination problem as a MIP.
 *     2.METHOD: solveMip(boolean verbose)
 *         This method solves the MIP of the winner determination problem and sets the optimal allocation.
 *     3.METHOD: logSolveDetails()
 *         This method logs Solution details.
 *     4.METHOD: toString()
 *         String representation of the WDP instance.
 *     5.METHOD: printOptimalAllocation()
 *         This method prints the optimal allocation x_star in a nice way.
 * 
 * @author Jakob Weissteiner (Original Python), Ported to Java
 * @copyright Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken
 * @license AGPL-3.0
 * @version 0.1.0
 * @status Dev
 */
public class WDP {
    
    private static final Logger logger = Logger.getLogger(WDP.class.getName());
    
    // Instance variables
    private double[][][] bids;  // 3D array: [bidder][bid][item+value]
    private int N;              // number of bidders
    private int M;              // number of items
    private IloCplex cplex;     // CPLEX model
    private int[] K;            // number of elicited bids per bidder
    private Map<String, IloNumVar> z; // decision variables z(i,k)
    private double[][] xStar;   // optimal allocation matrix
    private boolean solved;     // whether the problem has been solved
    
    /**
     * Constructor for WDP
     * @param bids 3D array representing elicited bundle-value pairs from each bidder
     *             bids[i][k][j] where i=bidder, k=bid, j=item (last column is value)
     */
    public WDP(double[][][] bids) {
        this.bids = bids;
        this.N = bids.length;  // number of bidders
        this.M = bids[0][0].length - 1;  // number of items (last column is value)
        this.K = new int[N];
        this.z = new HashMap<>();
        this.solved = false;
        
        // Calculate number of bids per bidder
        for (int i = 0; i < N; i++) {
            this.K[i] = bids[i].length;
        }
        
        // Initialize optimal allocation matrix
        this.xStar = new double[N][M];
        
        try {
            this.cplex = new IloCplex();
            this.cplex.setParam(IloCplex.Param.Simplex.Display, 0); // Reduce output
        } catch (IloException e) {
            throw new RuntimeException("Failed to initialize CPLEX: " + e.getMessage(), e);
        }
        
        logger.info("WDP initialized with " + N + " bidders and " + M + " items");
    }
    
    /**
     * Initialize the MIP formulation
     * @param verbose whether to log detailed information
     */
    public void initializeMip(boolean verbose) {
        try {
            // Add decision variables and allocation constraints for each bidder
            for (int i = 0; i < N; i++) {
                // Create binary decision variables z(i,k)
                IloNumVar[] bidderVars = new IloNumVar[K[i]];
                for (int k = 0; k < K[i]; k++) {
                    String varName = "z(" + i + "," + k + ")";
                    IloNumVar var = cplex.boolVar(varName);
                    z.put(varName, var);
                    bidderVars[k] = var;
                }
                
                // Add allocation constraint: sum of z(i,k) <= 1 for each bidder i
                IloLinearNumExpr allocationConstraint = cplex.linearNumExpr();
                for (int k = 0; k < K[i]; k++) {
                    allocationConstraint.addTerm(1.0, bidderVars[k]);
                }
                cplex.addLe(allocationConstraint, 1.0, "CT_Allocation_Bidder_" + i);
            }
            
            // Add intersection constraints for each item
            for (int m = 0; m < M; m++) {
                IloLinearNumExpr intersectionConstraint = cplex.linearNumExpr();
                
                for (int i = 0; i < N; i++) {
                    for (int k = 0; k < K[i]; k++) {
                        String varName = "z(" + i + "," + k + ")";
                        IloNumVar var = z.get(varName);
                        double coefficient = bids[i][k][m]; // bundle indicator for item m
                        intersectionConstraint.addTerm(coefficient, var);
                    }
                }
                
                cplex.addLe(intersectionConstraint, 1.0, "CT_Intersection_Item_" + m);
            }
            
            // Add objective function: maximize total value
            IloLinearNumExpr objective = cplex.linearNumExpr();
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K[i]; k++) {
                    String varName = "z(" + i + "," + k + ")";
                    IloNumVar var = z.get(varName);
                    double value = bids[i][k][M]; // value is in the last column
                    objective.addTerm(value, var);
                }
            }
            
            cplex.addMaximize(objective);
            
            if (verbose) {
                logger.info("MIP initialized with " + cplex.getNcols() + " variables and " + 
                           cplex.getNrows() + " constraints");
            }
            
        } catch (IloException e) {
            throw new RuntimeException("Failed to initialize MIP: " + e.getMessage(), e);
        }
    }
    
    /**
     * Solve the MIP
     * @param verbose whether to log detailed solve information
     */
    public void solveMip(boolean verbose) {
        try {
            boolean solved = cplex.solve();
            
            if (verbose) {
                logSolveDetails();
            }
            
            if (solved) {
                // Set the optimal allocation
                for (int i = 0; i < N; i++) {
                    Arrays.fill(xStar[i], 0.0); // Reset allocation
                    
                    for (int k = 0; k < K[i]; k++) {
                        String varName = "z(" + i + "," + k + ")";
                        IloNumVar var = z.get(varName);
                        double solutionValue = cplex.getValue(var);
                        
                        if (solutionValue > 0.5) { // Binary variable is 1
                            for (int m = 0; m < M; m++) {
                                xStar[i][m] = solutionValue * bids[i][k][m];
                            }
                        }
                    }
                }
                this.solved = true;
                logger.info("WDP solved successfully with objective value: " + cplex.getObjValue());
            } else {
                logger.warning("WDP could not be solved");
                this.solved = false;
            }
            
        } catch (IloException e) {
            throw new RuntimeException("Failed to solve MIP: " + e.getMessage(), e);
        }
    }
    
    /**
     * Log detailed solve information
     */
    public void logSolveDetails() {
        try {
            logger.info("Status: " + cplex.getStatus());
            logger.info("Time: " + String.format("%.2f", cplex.getCplexTime()) + " sec");
            
            if (cplex.getStatus() == IloCplex.Status.Optimal || 
                cplex.getStatus() == IloCplex.Status.Feasible) {
                logger.info("Objective Value: " + cplex.getObjValue());
                
                if (cplex.getStatus() == IloCplex.Status.Feasible) {
                    logger.info("MIP Relative Gap: " + String.format("%.5f", cplex.getMIPRelativeGap()) + "%");
                }
            }
            
        } catch (IloException e) {
            logger.log(Level.WARNING, "Could not retrieve solve details: " + e.getMessage(), e);
        }
    }
    
    /**
     * Print a comprehensive summary of the WDP solution
     */
    public void summary() {
        System.out.println("################################ OBJECTIVE ################################");
        try {
            if (solved) {
                System.out.println("Objective Value: " + cplex.getObjValue() + "\n");
            } else {
                System.out.println("Not yet solved!\n");
            }
        } catch (IloException e) {
            System.out.println("Error retrieving objective value: " + e.getMessage() + "\n");
        }
        
        System.out.println("############################# SOLVE STATUS ################################");
        try {
            System.out.println("Status: " + cplex.getStatus());
            System.out.println("Time: " + String.format("%.2f", cplex.getCplexTime()) + " sec");
            System.out.println("Variables: " + cplex.getNcols());
            System.out.println("Constraints: " + cplex.getNrows() + "\n");
        } catch (IloException e) {
            System.out.println("Error retrieving solve status: " + e.getMessage() + "\n");
        }
        
        System.out.println("########################### ALLOCATED BIDDERS ############################");
        try {
            if (solved) {
                for (int i = 0; i < N; i++) {
                    for (int k = 0; k < K[i]; k++) {
                        String varName = "z(" + i + "," + k + ")";
                        IloNumVar var = z.get(varName);
                        double value = cplex.getValue(var);
                        if (value > 0.5) {
                            System.out.println("z(" + i + "," + k + ")=" + Math.round(value));
                        }
                    }
                }
            } else {
                System.out.println("Not yet solved!");
            }
        } catch (IloException e) {
            System.out.println("Error retrieving variable values: " + e.getMessage());
        }
        
        System.out.println("########################### OPT ALLOCATION ###############################");
        printOptimalAllocation();
    }
    
    /**
     * Print the optimal allocation in a formatted table
     */
    public void printOptimalAllocation() {
        if (!solved) {
            System.out.println("Problem not yet solved!");
            return;
        }
        
        // Print header
        System.out.print("Bidder\t");
        for (int j = 0; j < M; j++) {
            System.out.print("Item_" + (j + 1) + "\t");
        }
        System.out.println();
        
        // Print allocation matrix
        for (int i = 0; i < N; i++) {
            System.out.print(i + "\t");
            for (int j = 0; j < M; j++) {
                System.out.print(String.format("%.1f", xStar[i][j]) + "\t");
            }
            System.out.println();
        }
        
        // Print items allocated totals
        System.out.println("\nItems allocated:");
        for (int j = 0; j < M; j++) {
            double total = 0.0;
            for (int i = 0; i < N; i++) {
                total += xStar[i][j];
            }
            System.out.println("Item_" + (j + 1) + ": " + String.format("%.1f", total));
        }
    }
    
    /**
     * Get the optimal allocation matrix
     * @return 2D array representing the optimal allocation
     */
    public double[][] getOptimalAllocation() {
        return xStar.clone();
    }
    
    /**
     * Get the objective value
     * @return objective value if solved, -1 otherwise
     */
    public double getObjectiveValue() {
        try {
            if (solved) {
                return cplex.getObjValue();
            }
        } catch (IloException e) {
            logger.log(Level.WARNING, "Could not retrieve objective value: " + e.getMessage(), e);
        }
        return -1;
    }
    
    /**
     * Check if the problem has been solved
     * @return true if solved, false otherwise
     */
    public boolean isSolved() {
        return solved;
    }
    
    /**
     * Clean up CPLEX resources
     */
    public void dispose() {
        if (cplex != null) {
            cplex.end();
        }
    }
    
    @Override
    public String toString() {
        return "WDP{" +
                "bidders=" + N +
                ", items=" + M +
                ", solved=" + solved +
                '}';
    }
    
    /**
     * Example usage and testing
     */
    public static void main(String[] args) {
        // Example: 2 bidders, 3 items
        // Bidder 0: 2 bids, Bidder 1: 1 bid
        double[][][] exampleBids = {
            { // Bidder 0
                {1.0, 0.0, 0.0, 10.0}, // Bundle {item1} with value 10
                {0.0, 1.0, 1.0, 15.0}  // Bundle {item2, item3} with value 15
            },
            { // Bidder 1
                {1.0, 1.0, 0.0, 12.0}  // Bundle {item1, item2} with value 12
            }
        };
        
        WDP wdp = new WDP(exampleBids);
        
        try {
            wdp.initializeMip(true);
            wdp.solveMip(true);
            wdp.summary();
        } finally {
            wdp.dispose();
        }
        
        System.out.println("WDP Class demonstration completed");
    }
}