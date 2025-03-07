import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.common.primitives.Pair;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.GridLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * FILE DESCRIPTION:
 *
 * This file implements the class NN (Neural Network) that has the following functionalities:
 * 0. CONSTRUCTOR: NN(ModelParameters modelParameters, INDArray xTrain, INDArray yTrain, NormalizerMinMaxScaler scaler)
 *    modelParameters = the parameters specifying the neural network
 *    *(xTrain, yTrain) is the training set of bundle-value pairs*.
 *    xTrain = The bundles of items
 *    yTrain = The corresponding values for the bundles from a specific bidder.
 *    scaler = A scaler instance for rescaling errors to the original scale
 * 
 * 1. METHOD: initializeModel(String regularizationType)
 *    regularizationType = The regularization for the affine transformation between layers.
 *    'l1': L1-Regularization, 'l2': L2-Regularization, 'l1_l2': Combination of both L1 and L2-Regularization.
 *    This method initializes the model attribute by defining the architecture and parameters of the neural network.
 * 
 * 2. METHOD: fit(int epochs, int batchSize, INDArray xValid, INDArray yValid, INDArray sampleWeight)
 *    epochs = Number of epochs the neural network is trained
 *    batchSize = Batch size used in training
 *    xValid = Test set of bundles
 *    yValid = Values for xValid
 *    sampleWeight = weights vector for datapoints of bundle-value pairs
 *    This method fits a neural network to data and returns loss numbers.
 * 
 * 3. METHOD: lossInfo(int batchSize, boolean plot, String scale)
 *    batchSize = Batch size used in training
 *    plot = boolean parameter if plots for the goodness of fit should be executed
 *    scale = either null or 'log' defining the scaling of the y-axis for the plots
 *    This method calculates losses and plots a goodness of fit plot.
 */

public class NN {
    private static final Logger logger = Logger.getLogger(NN.class.getName());
    
    private int m; // number of items
    private INDArray xTrain; // training set of bundles
    private INDArray yTrain; // bidder's values for the bundles in xTrain
    private INDArray xValid; // test/validation set of bundles
    private INDArray yValid; // bidder's values for the bundles in xValid
    private ModelParameters modelParameters; // neural network parameters
    private MultiLayerNetwork model; // DL4J model, i.e., the neural network
    private NormalizerMinMaxScaler scaler; // the scaler used for scaling yTrain values
    private List<Pair<Double, Double>> history; // training and validation loss history
    private double[] loss; // final loss values
    
    /**
     * Constructor for the NN class
     */
    public NN(ModelParameters modelParameters, INDArray xTrain, INDArray yTrain, NormalizerMinMaxScaler scaler) {
        this.m = xTrain.columns(); // number of items
        this.xTrain = xTrain; // training set of bundles
        this.yTrain = yTrain; // bidder's values for the bundles in xTrain
        this.xValid = null; // test/validation set of bundles
        this.yValid = null; // bidder's values for the bundles in xValid
        this.modelParameters = modelParameters; // neural network parameters
        this.model = null; // DL4J model
        this.scaler = scaler; // the scaler used for scaling yTrain values
        this.history = new ArrayList<>(); // training history
        this.loss = null; // final loss values
    }
    
    /**
     * Initialize the neural network model
     */
    public void initializeModel(String regularizationType) {
        double r = modelParameters.getRegularizationParameter();
        double lr = modelParameters.getLearningRate();
        int[] dim = modelParameters.getDimensions();
        boolean dropout = modelParameters.isUseDropout();
        double dp = modelParameters.getDropoutRate();
        
        int numberOfHiddenLayers = dim.length;
        
        // Configuration for the neural network
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(lr, 0.9, 0.999, 1e-8));
        
        // Set regularization
        if ("l2".equals(regularizationType) || regularizationType == null) {
            builder.l2(r);
            logger.info("l2 regularization");
        } else if ("l1".equals(regularizationType)) {
            builder.l1(r);
            logger.info("l1 regularization");
        } else if ("l1_l2".equals(regularizationType)) {
            builder.l1(r).l2(r);
            logger.info("l1&l2 regularization");
        }
        
        // Create the neural network configuration
        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        
        // First hidden layer
        listBuilder.layer(0, new DenseLayer.Builder()
                .nIn(this.xTrain.columns())
                .nOut(dim[0])
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .dropOut(dropout ? dp : 0.0)
                .build());
        
        // Remaining hidden layers
        for (int k = 1; k < numberOfHiddenLayers; k++) {
            listBuilder.layer(k, new DenseLayer.Builder()
                    .nIn(dim[k-1])
                    .nOut(dim[k])
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .dropOut(dropout ? dp : 0.0)
                    .build());
        }
        
        // Output layer
        listBuilder.layer(numberOfHiddenLayers, new OutputLayer.Builder(LossFunctions.LossFunction.MAE)
                .nIn(dim[numberOfHiddenLayers-1])
                .nOut(1)
                .activation(Activation.RELU) // ReLU for non-negative output
                .weightInit(WeightInit.XAVIER)
                .build());
        
        // Finalize the configuration
        listBuilder.pretrain(false).backprop(true);
        
        // Build and initialize the model
        this.model = new MultiLayerNetwork(listBuilder.build());
        this.model.init();
        this.model.setListeners(new ScoreIterationListener(10)); // Print score every 10 iterations
        
        logger.info("Neural Net initialized");
    }
    
    /**
     * Fit the neural network to the data
     */
    public double[] fit(int epochs, int batchSize, INDArray xValid, INDArray yValid, INDArray sampleWeight) {
        // Set test set if provided
        this.xValid = xValid;
        this.yValid = yValid;
        
        // Create a DataSet from the training data
        DataSet trainingData = new DataSet(this.xTrain, this.yTrain);
        
        // If sample weights are provided, apply them
        if (sampleWeight != null) {
            // DL4J handles weights differently from Keras, would need additional processing
        }
        
        // Train the model
        this.history.clear();
        for (int i = 0; i < epochs; i++) {
            trainingData.shuffle();
            for (int j = 0; j < trainingData.numExamples() / batchSize + 1; j++) {
                int start = j * batchSize;
                int end = Math.min(start + batchSize, trainingData.numExamples());
                if (start < end) {
                    DataSet batch = trainingData.getRange(start, end);
                    this.model.fit(batch);
                }
            }
            
            // Calculate and store current loss
            double trainLoss = this.model.score(trainingData);
            double validLoss = (this.xValid != null && this.yValid != null) ? 
                    this.model.score(new DataSet(this.xValid, this.yValid)) : 0.0;
            this.history.add(new Pair<>(trainLoss, validLoss));
        }
        
        // Get final loss information
        this.loss = lossInfo(batchSize, false);
        return this.loss;
    }
    
    /**
     * Calculate loss information and optionally plot results
     */
    public double[] lossInfo(int batchSize, boolean plot) {
        return lossInfo(batchSize, plot, null);
    }
    
    /**
     * Calculate loss information and optionally plot results with y-axis scaling
     */
    public double[] lossInfo(int batchSize, boolean plot, String scale) {
        logger.info("Model Parameters: " + this.modelParameters);
        
        Double tr = null;
        Double trOrig = null;
        Double val = null;
        Double valOrig = null;
        
        // Calculate errors
        if (this.scaler != null) {
            // Calculate errors on the training set
            DataSet trainSet = new DataSet(this.xTrain, this.yTrain);
            tr = this.model.score(trainSet);
            
            // Convert error back to original scale
            INDArray scaledErrorArray = Nd4j.create(new double[][]{{tr}});
            INDArray originalErrorArray = this.scaler.revertFeatures(scaledErrorArray);
            trOrig = originalErrorArray.getDouble(0, 0);
            
            // Calculate errors on the test set if available
            if (this.xValid != null && this.yValid != null) {
                DataSet validSet = new DataSet(this.xValid, this.yValid);
                val = this.model.score(validSet);
                
                // Convert validation error back to original scale
                scaledErrorArray = Nd4j.create(new double[][]{{val}});
                originalErrorArray = this.scaler.revertFeatures(scaledErrorArray);
                valOrig = originalErrorArray.getDouble(0, 0);
            }
        } else {
            // Data has not been scaled
            DataSet trainSet = new DataSet(this.xTrain, this.yTrain);
            trOrig = this.model.score(trainSet);
            
            if (this.xValid != null && this.yValid != null) {
                DataSet validSet = new DataSet(this.xValid, this.yValid);
                valOrig = this.model.score(validSet);
            }
        }
        
        // Log errors
        if (tr != null) {
            logger.info("Train Error Scaled " + tr);
        }
        if (val != null) {
            logger.info("Validation Error Scaled " + val);
        }
        if (trOrig != null) {
            logger.info("Train Error Orig. " + trOrig);
        }
        if (valOrig != null) {
            logger.info("Validation Error Orig " + valOrig);
        }
        
        // Plot results if requested
        if (plot) {
            // Predict values for training and test sets
            INDArray yHatTrain = this.model.output(this.xTrain);
            INDArray yHatValid = null;
            if (this.xValid != null && this.yValid != null) {
                yHatValid = this.model.output(this.xValid);
            }
            
            // Create plots
            JFrame frame = new JFrame("Neural Network Results");
            frame.setSize(1200, 600);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLayout(new GridLayout(1, 2));
            
            // Plot 1: Training vs Test Loss
            XYSeriesCollection lossDataset = new XYSeriesCollection();
            XYSeries trainLossSeries = new XYSeries("Train Loss");
            XYSeries validLossSeries = new XYSeries("Test Loss");
            
            for (int i = 0; i < this.history.size(); i++) {
                trainLossSeries.add(i, this.history.get(i).getFirst());
                if (this.xValid != null && this.yValid != null) {
                    validLossSeries.add(i, this.history.get(i).getSecond());
                }
            }
            
            lossDataset.addSeries(trainLossSeries);
            if (this.xValid != null && this.yValid != null) {
                lossDataset.addSeries(validLossSeries);
            }
            
            JFreeChart lossChart = ChartFactory.createXYLineChart(
                    "Training vs. Test Loss DNN",
                    "Number of Epochs",
                    "Mean Absolute Error",
                    lossDataset,
                    PlotOrientation.VERTICAL,
                    true,
                    true,
                    false
            );
            
            // Plot 2: Predicted vs True Values
            XYSeriesCollection predictionDataset = new XYSeriesCollection();
            XYSeries trainPredSeries = new XYSeries("Training Points");
            XYSeries validPredSeries = new XYSeries("Test Points");
            XYSeries idealLine = new XYSeries("Ideal");
            
            // Add training predictions
            for (int i = 0; i < this.xTrain.rows(); i++) {
                trainPredSeries.add(yHatTrain.getDouble(i, 0), this.yTrain.getDouble(i, 0));
            }
            
            // Add validation predictions if available
            if (yHatValid != null) {
                for (int i = 0; i < this.xValid.rows(); i++) {
                    validPredSeries.add(yHatValid.getDouble(i, 0), this.yValid.getDouble(i, 0));
                }
            }
            
            // Add ideal line (y = x)
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            
            for (int i = 0; i < this.yTrain.rows(); i++) {
                double value = this.yTrain.getDouble(i, 0);
                if (value < min) min = value;
                if (value > max) max = value;
            }
            
            if (this.yValid != null) {
                for (int i = 0; i < this.yValid.rows(); i++) {
                    double value = this.yValid.getDouble(i, 0);
                    if (value < min) min = value;
                    if (value > max) max = value;
                }
            }
            
            idealLine.add(min, min);
            idealLine.add(max, max);
            
            predictionDataset.addSeries(trainPredSeries);
            if (yHatValid != null) {
                predictionDataset.addSeries(validPredSeries);
            }
            predictionDataset.addSeries(idealLine);
            
            JFreeChart predictionChart = ChartFactory.createScatterPlot(
                    "Prediction Accuracy",
                    "Predicted Values",
                    "True Values",
                    predictionDataset,
                    PlotOrientation.VERTICAL,
                    true,
                    true,
                    false
            );
            
            // Add charts to frame
            frame.add(new ChartPanel(lossChart));
            frame.add(new ChartPanel(predictionChart));
            
            frame.setVisible(true);
        }
        
        // Return loss values
        double[] lossValues = new double[4];
        lossValues[0] = tr != null ? tr : 0.0;
        lossValues[1] = val != null ? val : 0.0;
        lossValues[2] = trOrig != null ? trOrig : 0.0;
        lossValues[3] = valOrig != null ? valOrig : 0.0;
        
        return lossValues;
    }
    
    // Helper class to store model parameters
    public static class ModelParameters {
        private double regularizationParameter;
        private double learningRate;
        private int[] dimensions;
        private boolean useDropout;
        private double dropoutRate;
        
        public ModelParameters(double regularizationParameter, double learningRate, 
                              int[] dimensions, boolean useDropout, double dropoutRate) {
            this.regularizationParameter = regularizationParameter;
            this.learningRate = learningRate;
            this.dimensions = dimensions;
            this.useDropout = useDropout;
            this.dropoutRate = dropoutRate;
        }
        
        public double getRegularizationParameter() {
            return regularizationParameter;
        }
        
        public double getLearningRate() {
            return learningRate;
        }
        
        public int[] getDimensions() {
            return dimensions;
        }
        
        public boolean isUseDropout() {
            return useDropout;
        }
        
        public double getDropoutRate() {
            return dropoutRate;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("ModelParameters{regularization=").append(regularizationParameter);
            sb.append(", learningRate=").append(learningRate);
            sb.append(", dimensions=[");
            for (int i = 0; i < dimensions.length; i++) {
                sb.append(dimensions[i]);
                if (i < dimensions.length - 1) {
                    sb.append(", ");
                }
            }
            sb.append("], useDropout=").append(useDropout);
            sb.append(", dropoutRate=").append(dropoutRate);
            sb.append("}");
            return sb.toString();
        }
    }
}
