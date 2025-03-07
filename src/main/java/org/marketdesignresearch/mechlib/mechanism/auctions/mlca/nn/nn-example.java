import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;

public class ExampleNN {
    
    public static void main(String[] args) {
        // Create sample data
        double[][] xTrainData = {
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
        };
        
        double[][] yTrainData = {
            {0}, {10}, {20}, {30}, {40}, {50}, {60}, {70}
        };
        
        // Convert to ND4J arrays
        INDArray xTrain = Nd4j.create(xTrainData);
        INDArray yTrain = Nd4j.create(yTrainData);
        
        // Create and fit a scaler
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(yTrain);
        
        // Scale the target values
        INDArray yTrainScaled = yTrain.dup();
        scaler.transform(yTrainScaled);
        
        // Define model parameters (regularization, learning rate, hidden layers, dropout flag, dropout rate)
        NN.ModelParameters modelParams = new NN.ModelParameters(
            0.001,  // regularization parameter
            0.001,  // learning rate
            new int[]{10, 5},  // hidden layer dimensions
            true,   // use dropout
            0.2     // dropout rate
        );
        
        // Create and initialize the neural network
        NN neuralNetwork = new NN(modelParams, xTrain, yTrainScaled, scaler);
        neuralNetwork.initializeModel("l2");  // Use L2 regularization
        
        // Train the model (no validation data in this example)
        int epochs = 1000;
        int batchSize = 4;
        double[] loss = neuralNetwork.fit(epochs, batchSize, null, null, null);
        
        // Display loss information with plotting
        neuralNetwork.lossInfo(batchSize, true);
        
        // Print final loss
        System.out.println("Final training loss (scaled): " + loss[0]);
        System.out.println("Final training loss (original): " + loss[2]);
    }
}
