/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.jan.linearregression;

/**
 *
 * @author koros2
 */
public class Main {
    public static void main(String[] args) {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {5, 8, 11,14, 17};
        
        LinearRegression model = new LinearRegression();
        model.fit(x, y, 1000, 0.01);
        System.out.println("Prediction for x = 6: " + model.predict(6));
        System.out.println("Prediction for x = 10: " + model.predict(10));

        System.out.println("Weight: " + model.getWeight() + ", Bias: " + model.getBias());
    }
}
