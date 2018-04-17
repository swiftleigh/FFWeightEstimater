/*
*  * Copyright 2017 Skymind, Inc.
*  *
*  *    Licensed under the Apache License, Version 2.0 (the "License");
*  *    you may not use this file except in compliance with the License.
*  *    You may obtain a copy of the License at
*  *
*  *        http://www.apache.org/licenses/LICENSE-2.0
*  *
*  *    Unless required by applicable law or agreed to in writing, software
*  *    distributed under the License is distributed on an "AS IS" BASIS,
*  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  *    See the License for the specific language governing permissions and
*  *    limitations under the License.
*/

package org.datavec.image.recordreader.objdetect;

import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.util.ImageUtils;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * An image record reader for object detection.
 * <p>
 * Format of returned values: 4d array, with dimensions [minibatch, 4+C, h, w]
 * Where the image is quantized into h x w grid locations.
 * <p>
 * Note that this matches the format required for Deeplearning4j's Yolo2OutputLayer
 *
 * @author Alex Black
 */
public class ObjectDetectionRecordReader extends BaseImageRecordReader {

    private final int gridW;
    private final int gridH;
    private final ImageObjectLabelProvider labelProvider;

    /**
     *
     * @param height        Height of the output images
     * @param width         Width of the output images
     * @param channels      Number of channels for the output images
     * @param gridH         Grid/quantization size (along  height dimension) - Y axis
     * @param gridW         Grid/quantization size (along  height dimension) - X axis
     * @param labelProvider ImageObjectLabelProvider - used to look up which objects are in each image
     */
    public ObjectDetectionRecordReader(int height, int width, int channels, int gridH, int gridW, ImageObjectLabelProvider labelProvider) {
        super(height, width, channels, null);
        this.gridW = gridW;
        this.gridH = gridH;
        this.labelProvider = labelProvider;

    }

    @Override
    public List<Writable> next() {
        return next(1);
    }

    @Override
    public void initialize(InputSplit split) throws IOException {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }
        inputSplit = split;
        URI[] locations = split.locations();
        Set<String> labelSet = new HashSet<>();
        if (locations != null && locations.length >= 1) {
            for (URI location : locations) {
                System.out.println("ODRR: " + location);
                List<ImageObject> imageObjects = labelProvider.getImageObjectsForPath(location);
                for (ImageObject io : imageObjects) {
                    String name = io.getLabel();
                    if (!labelSet.contains(name)) {
                        labelSet.add(name);
                    }
                }
            }
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator()); //This handles randomization internally if necessary
        } else
            throw new IllegalArgumentException("No path locations found in the split.");

        if (split instanceof FileSplit) {
            //remove the root directory
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        labels = new ArrayList<>(labelSet);
        Collections.sort(labels);
    }

    @Override
    public List<Writable> next(int num) {
        List<File> files = new ArrayList<>(num);
        List<List<ImageObject>> objects = new ArrayList<>(num);

        for (int i = 0; i < num && hasNext(); i++) {
            File f = iter.next();
            this.currentFile = f;
            if (!f.isDirectory()) {
                files.add(f);
                objects.add(labelProvider.getImageObjectsForPath(f.getPath()));
            }
        }


        int nClasses = labels.size();

        INDArray outImg = Nd4j.create(files.size(), channels, height, width);
        INDArray outLabel = Nd4j.create(files.size(), 4 + nClasses, gridH, gridW);

        int exampleNum = 0;
        for (int i = 0; i < files.size(); i++) {
            File imageFile = files.get(i);
            this.currentFile = imageFile;
            try {
                this.invokeListeners(imageFile);
                Image image = this.imageLoader.asImageMatrix(imageFile);
                Nd4j.getAffinityManager().ensureLocation(image.getImage(), AffinityManager.Location.DEVICE);

                outImg.put(new INDArrayIndex[]{point(exampleNum), all(), all(), all()}, image.getImage());

                List<ImageObject> objectsThisImg = objects.get(exampleNum);

                double oW = image.getOrigW();
                double oH = image.getOrigH();

                //put the label data into the output label array
                for (ImageObject io : objectsThisImg) {
                    double cx = io.getXCenterPixels();
                    double cy = io.getYCenterPixels();

                    double[] cxyPostScaling = ImageUtils.translateCoordsScaleImage(cx, cy, oW, oH, width, height);
                    double[] tlPost = ImageUtils.translateCoordsScaleImage(io.getX1(), io.getY1(), oW, oH, width, height);
                    double[] brPost = ImageUtils.translateCoordsScaleImage(io.getX2(), io.getY2(), oW, oH, width, height);

                    //Get grid position for image
                    int imgGridX = (int) (cxyPostScaling[0] / width * gridW);
                    int imgGridY = (int) (cxyPostScaling[1] / height * gridH);

                    //Convert pixels to grid position, for TL and BR X/Y
                    tlPost[0] = tlPost[0] / width * gridW;
                    tlPost[1] = tlPost[1] / height * gridH;
                    brPost[0] = brPost[0] / width * gridW;
                    brPost[1] = brPost[1] / height * gridH;

                    //Put TL, BR into label array:
                    outLabel.putScalar(exampleNum, 0, imgGridY, imgGridX, tlPost[0]);
                    outLabel.putScalar(exampleNum, 1, imgGridY, imgGridX, tlPost[1]);
                    outLabel.putScalar(exampleNum, 2, imgGridY, imgGridX, brPost[0]);
                    outLabel.putScalar(exampleNum, 3, imgGridY, imgGridX, brPost[1]);

                    //Put label class into label array: (one-hot representation)
                    int labelIdx = labels.indexOf(io.getLabel());
                    outLabel.putScalar(exampleNum, 4 + labelIdx, imgGridY, imgGridX, 1.0);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            exampleNum++;
        }

        return Arrays.<Writable>asList(new NDArrayWritable(outImg), new NDArrayWritable(outLabel));
    }

}