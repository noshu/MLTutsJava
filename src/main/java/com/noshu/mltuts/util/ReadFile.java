/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
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
    int target;
    boolean headers;

    public ReadFile(String path, String delimeter, int target, boolean headers) {
        this.rawData = new LinkedList();
        this.path = path;
        this.delimeter = delimeter;
        this.target = target;
        this.headers = headers;
        load();
    }

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

    public INDArray getFeatures() {
        int c = 0;
        if (headers) {
            c = 1;
        }
        int cut = c;
        double[][] features = new double[rawData.size() - cut][rawData.getFirst().length - 1];
        while (c < rawData.size()) {
            //counter for features
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

    public INDArray getTarget() {
        int c = 0;
        if (headers) {
            c = 1;
        }
        int cut = c;
        double[] targetvec = new double[rawData.size() - cut];
        while (c < rawData.size()) {
            targetvec[c-cut] = Double.parseDouble(rawData.get(c)[target]);
            c++;
        }
        return Nd4j.create(targetvec, new int[]{targetvec.length, 1});
    }

}
