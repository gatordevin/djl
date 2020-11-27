/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.modality.cv.pairedtransform;

import ai.djl.modality.cv.transform.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.PairedTransform;
import ai.djl.translate.Transform;
import java.util.Random;

/**
 * A {@link Transform} that randomly flip the input image left to right with a probability of 0.5.
 */
public class PairedRandomFlipTopBottom implements PairedTransform {

    /** {@inheritDoc} */
    @Override
    public NDArray[] transform(NDArray inputArray, NDArray targetBox) {
        if(new Random().nextBoolean()){
            targetBox.set(new NDIndex(":, 1:"), array -> array.mul(-1).add(1)); //Only flip y values for vertical flip
            targetBox.set(new NDIndex(":, 1::2"), array -> array.mul(-1).add(1));
            System.out.println(targetBox.get(":, :"));
            return new NDArray[] {inputArray.flip(0), targetBox};
        }else{
            return new NDArray[] {inputArray, targetBox};
        }
    }
}
