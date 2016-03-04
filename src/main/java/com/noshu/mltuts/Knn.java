/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.noshu.mltuts;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author naushad
 */
public class Knn {
    
    public double Classify(INDArray para,INDArray label,INDArray input,int k){
        INDArray distances = Nd4j.create(label.rows(), 1);
        for(int x =0;x<para.columns();x++){
            distances.addi(Transforms.pow(para.getColumn(x).sub(input.getColumn(x)),2));
        }
        distances = Transforms.sqrt(distances);
        INDArray[] sorted = Nd4j.sortWithIndices(distances, 0, true);
        int[] count = new int[(int)label.max(0).getDouble(0)+1];
        for(int x=0;x<k;x++){
            count[(int)label.getRow((int)sorted[0].getDouble(x)).getDouble(0)]++;
        }
        int max = -1;
        double pred = -1;
        for(int i = 0;i<count.length;i++){
            if(count[i]>max){
                max = count[i];
                pred = i;
            }
        }
        return pred;
    }
    
}
