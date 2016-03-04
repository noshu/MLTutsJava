/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.noshu.mltuts.tests;

import com.noshu.mltuts.Knn;
import com.noshu.mltuts.util.ReadFile;
import java.io.IOException;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author naushad
 */
public class KnnTest {
    public static void main(String[] args) throws IOException, InterruptedException {
        System.out.println("reading data......");
        ReadFile train = new ReadFile("data/train.csv", ",", 0, true);
        ReadFile test = new ReadFile("data/test.csv", ",", 0, false);
        System.out.println("getting train test sets.......");
        INDArray xTrain = train.getFeatures();
        INDArray yTrain = train.getTarget();
        INDArray xTest = test.getFeatures();
        INDArray yTest = test.getTarget();
        Knn knn =  new Knn();
        double correct = 0;
        System.out.println("Classifying.......");
        for(int i = 0;i<xTest.size(0);i++){
            double pred = knn.Classify(xTrain, yTrain, xTest.getRow(i), 5);
            System.out.println("original value: "+yTest.getRow(i)+" predicted value: "+pred);
            if(yTest.getRow(i).getDouble(0)==pred){
                correct++;
            }
        }
        System.out.println("Accurecy is: "+correct/xTest.size(0)+"%");
    }
}
