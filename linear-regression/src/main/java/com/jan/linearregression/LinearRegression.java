package com.jan.linearregression;

// @author Joan Lozano Funosas, PROP
public class LinearRegression {
    private double weight;
    private double bias;
    
    public LinearRegression() {
        weight = 0.0;
        bias = 0.0;
    }
    
    public double getWeight() {
        return weight;
    }
    
    public double getBias() {
        return bias;
    }
    
    public double predict(double x) {
        return weight * x + bias;
    }
    
    // x has feature samples for predictions.
    // y is the target prediction we want.
    public void fit(double[] x, double[] y, int epochs, double learningRate) {
       // 1. Predict y for every sample.
       if(x.length != y.length) {
           throw new IllegalArgumentException("x and y must have same length");
       }
       if (x.length == 0) {
           throw new IllegalArgumentException("Training data cannot be empty");
       }
       
       if(epochs <= 0) {
           throw new IllegalArgumentException("Epochs must be bigger than zero.");
       }
       
       if (learningRate <= 0) {
           throw new IllegalArgumentException("Learning Rate must be bigger than zero.");
       }
       
       for(int epoch = 0; epoch < epochs; ++epoch) {     
            if(epoch % 100 == 0) {
                double mse = meanSquaredError(x, y);
                System.out.println("Epoch: " + epoch + ", MSE: " + mse);
            }

            // 3. Calculate how weight and bias should change
            // Gradient of weight = 2 / n * sum(e_i*x_i)
            int n = x.length;
            double gradientW = 0.0;
            double gradientB = 0.0;
            for(int i = 0; i < n; ++i) {
                double prediction = predict(x[i]);
                double error = prediction - y[i];
                
                gradientW += error * x[i];
                gradientB += error;
            }
            
            gradientW *= (2.0 / n);
            gradientB *= (2.0 / n);

            // 4. Update parameters: Gradient descent updates each parameter
            // by subtracting its gradient times the learning rate.
            weight -= learningRate * gradientW;
            bias -= learningRate * gradientB;
       }
       
       System.out.println("Final MSE: " + meanSquaredError(x, y));

    }
    
    public double meanSquaredError(double[] x, double[] targets) {
        if(x.length <= 0 || targets.length <= 0) {
            throw new IllegalArgumentException("Samples or targets are smaller than 1.");
        }
        
        if (x.length != targets.length) {
            throw new IllegalArgumentException("Inputs and targets must have the same length.");
        }
        
        int n = x.length;
        double mse = 0.0;
        for(int i = 0; i < n; ++i) {
            double error = targets[i] - predict(x[i]);
            mse += error * error;
        }
        
        mse /= n;
        return mse;
    }
}

