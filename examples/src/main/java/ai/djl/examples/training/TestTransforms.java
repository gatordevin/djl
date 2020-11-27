/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ai.djl.examples.training;

import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.pairedtransform.PairedRandomFlipLeftRight;
import ai.djl.modality.cv.pairedtransform.PairedRandomFlipTopBottom;
import ai.djl.modality.cv.pairedtransform.PairedResize;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.PairedPipeline;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *
 * @author techgarage
 */
public class TestTransforms {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        PairedPipeline pairedPipeline = new PairedPipeline(new PairedResize(640,284), new PairedRandomFlipLeftRight(), new PairedRandomFlipTopBottom());
        try(NDManager manager = NDManager.newBaseManager()) {
            BufferedImageFactory bif = new BufferedImageFactory();
            Image img = bif.fromFile(Paths.get("/home/techgarage/Downloads/segmentation.jpg"));
            NDArray nd = img.toNDArray(manager);
            NDList[] transformResult = pairedPipeline.transform(new NDList(nd), new NDList(manager.ones(new Shape(1, 5))));
//            System.out.println(transformResult[1].get(0));
            Path outputDir = Paths.get("/home/techgarage/Downloads/output");
            Files.createDirectories(outputDir);
            Path imagePath = outputDir.resolve("flippedImage" + 0 + ".png");
            bif.fromNDArray(transformResult[0].get(0).transpose(2,0,1)).save(Files.newOutputStream(imagePath), "png");
        }
        
    }
    
}
