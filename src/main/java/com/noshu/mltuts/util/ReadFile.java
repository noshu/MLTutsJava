/*
 * simple utility class for reading character separated data
 * return feature Matrix
 * return responce vector
 */
package com.noshu.mltuts.util;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author naushad
 */
public class ReadFile {
    private final String path;
    private final String delimeter;
    private final LinkedList<String[]> rawData;
    //the responce column number
    int target;
    //check if the file contains header
    boolean headers;
    //reads file form path
    public ReadFile(String path, String delimeter, int target, boolean headers) {
        this.rawData = new LinkedList();
        this.path = path;
        this.delimeter = delimeter;
        this.target = target;
        this.headers = headers;
        load();
    }
    //load the data from path
    private void load() {
        LineIterator it = null;
        try {
            it = FileUtils.lineIterator(new File(path), "UTF-8");
            while (it.hasNext()) {
                String line = it.nextLine();
                rawData.add(line.split(delimeter));
            }
        } catch (IOException e) {
            System.err.println("Error reading file");
        } finally {
            if (it != null) {
                LineIterator.closeQuietly(it);
            }
        }
    }
    //returns a feature matrix or vector
    public INDArray getFeatures() {
        int c = 0;
        if (headers) {
            c = 1;
        }
        //adjusting the cursor if the file contains header
        int cut = c;
        double[][] features = new double[rawData.size() - cut][rawData.getFirst().length - 1];
        while (c < rawData.size()) {
            //counter for features becuse feature vector column number is less then raw twodimensional list data
            int fc = 0;
            for (int x = 0; x < rawData.getFirst().length; x++) {
                if (x != target) {
                    features[c-cut][fc] = Double.parseDouble(rawData.get(c)[x]);
                    fc++;
                }

            }
            c++;
        }
        return Nd4j.create(features);

    }
    //return responce vector
    public INDArray getTarget() {
        int c = 0;
        if (headers) {
            c = 1;
        }
        //adjusting the cursor if the file contains header
        int cut = c;
        double[] targetvec = new double[rawData.size() - cut];
        while (c < rawData.size()) {
            targetvec[c-cut] = Double.parseDouble(rawData.get(c)[target]);
            c++;
        }
        return Nd4j.create(targetvec, new int[]{targetvec.length, 1});
    }

}
