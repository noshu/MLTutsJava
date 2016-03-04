/*
 * This class Calculate Eucladian distance between input and each features
 * then we vote the labels of nearest neighbour
 * the label which has the maximum ammount of vote is returned
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
        //vector that holds the distances
        INDArray distances = Nd4j.create(label.rows(), 1);
        //caculate the distance of each feature from the input
        for(int x =0;x<para.columns();x++){
            distances.addi(Transforms.pow(para.getColumn(x).sub(input.getColumn(x)),2));
        }
        //square root the distance
        distances = Transforms.sqrt(distances);
        //get the sorted indicies of our distance vector
        INDArray[] sorted = Nd4j.sortWithIndices(distances, 0, true);
        int[] count = new int[(int)label.max(0).getDouble(0)+1];
        //voting the labels
        for(int x=0;x<k;x++){
            count[(int)label.getRow((int)sorted[0].getDouble(x)).getDouble(0)]++;
        }
        int max = -1;
        double pred = -1;
        //get the label with maximum vote
        for(int i = 0;i<count.length;i++){
            if(count[i]>max){
                max = count[i];
                pred = i;
            }
        }
        return pred;
    }
    
}
