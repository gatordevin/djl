/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** {@code PairedPipeline} allows applying multiple transforms on an input {@link NDList}. */
public class PairedPipeline {

    private PairList<IndexKey, PairedTransform> transforms;

    /** Creates a new instance of {@code PairedPipeline} that has no {@link PairedTransform} defined yet. */
    public PairedPipeline() {
        transforms = new PairList<>();
    }

    /**
     * Creates a new instance of {@code PairedPipeline} that can apply the given transforms on its input.
     *
     * <p>Since no keys are provided for these transforms, they will be applied to the first element
     * in the input {@link NDList} when the {@link #transform(NDList) transform} method is called on
     * this object.
     *
     * @param transforms the transforms to be applied when the {@link #transform(NDList) transform}
     *     method is called on this object
     */
    public PairedPipeline(PairedTransform... transforms) {
        this.transforms = new PairList<>();
        for (PairedTransform transform : transforms) {
            this.transforms.add(new IndexKey(0), transform);
        }
    }

    /**
     * Adds the given {@link PairedTransform} to the list of transforms to be applied on the input when
     * the {@link #transform(NDList) transform} method is called on this object.
     *
     * <p>Since no keys are provided for this {@link PairedTransform}, it will be applied to the first
     * element in the input {@link NDList}.
     *
     * @param transform the {@link PairedTransform} to be added
     * @return this {@code PairedPipeline}
     */
    public PairedPipeline add(PairedTransform transform) {
        transforms.add(new IndexKey(0), transform);
        return this;
    }

    /**
     * Adds the given {@link PairedTransform} to the list of transforms to be applied on the {@link
     * NDArray} at the given index in the input {@link NDList}.
     *
     * @param index the index corresponding to the {@link NDArray} in the input {@link NDList} on
     *     which the given transform must be applied to
     * @param transform the {@link PairedTransform} to be added
     * @return this {@code PairedPipeline}
     */
    public PairedPipeline add(int index, PairedTransform transform) {
        transforms.add(new IndexKey(index), transform);
        return this;
    }

    /**
     * Adds the given {@link PairedTransform} to the list of transforms to be applied on the {@link
     * NDArray} with the given key as name in the input {@link NDList}.
     *
     * @param name the key corresponding to the {@link NDArray} in the input {@link NDList} on which
     *     the given transform must be applied to
     * @param transform the {@code PairedTransform} to be applied when the {@link #transform(NDList)
     *     transform} method is called on this object
     * @return this {@code PairedPipeline}
     */
    public PairedPipeline add(String name, PairedTransform transform) {
        transforms.add(new IndexKey(name), transform);
        return this;
    }

    /**
     * Inserts the given {@link PairedTransform} to the list of transforms at the given position.
     *
     * <p>Since no keys or indices are provided for this {@link Transform}, it will be applied to
     * the first element in the input {@link NDList} when the {@link #transform(NDList) transform}
     * method is called on this object.
     *
     * @param position the position at which the {@link Transform} must be inserted
     * @param transform the {@code PairedTransform} to be inserted
     * @return this {@code PairedPipeline}
     */
    public PairedPipeline insert(int position, PairedTransform transform) {
        transforms.add(position, new IndexKey(0), transform);
        return this;
    }

    /**
     * Inserts the given {@link PairedTransform} to the list of transforms at the given position to be
     * applied on the {@link NDArray} at the given index in the input {@link NDList}.
     *
     * @param position the position at which the {@link PairedTransform} must be inserted
     * @param index the index corresponding to the {@link NDArray} in the input {@link NDList} on
     *     which the given transform must be applied to
     * @param transform the {@code PairedTransform} to be inserted
     * @return this {@code PairedPipeline}
     */
    public PairedPipeline insert(int position, int index, PairedTransform transform) {
        transforms.add(position, new IndexKey(index), transform);
        return this;
    }

    /**
     * Inserts the given {@link PairedTransform} to the list of transforms at the given position to be
     * applied on the {@link NDArray} with the given name in the input {@link NDList}.
     *
     * @param position the position at which the {@link PairedTransform} must be inserted
     * @param name the key corresponding to the {@link NDArray} in the input {@link NDList} on which
     *     the given transform must be applied to
     * @param transform the {@code Transform} to be inserted
     * @return this {@code PairedPipeline}
     */
    public PairedPipeline insert(int position, String name, PairedTransform transform) {
        transforms.add(position, new IndexKey(name), transform);
        return this;
    }

    /**
     * Applies the transforms configured in this object on the input {@link NDList}.
     *
     * <p>If a key is specified with the transform, those transforms will only be applied to the
     * {@link NDArray} in the input {@link NDList}. If a key is not specified, it will be applied to
     * the first element in the input {@link NDList}.
     *
     * @param input the input {@link NDList} on which the tranforms are to be applied
     * @return the output {@link NDList} after applying the tranforms
     */
    public NDList[] transform(NDList input, NDList target) {
        if (transforms.isEmpty() || input.isEmpty() || target.isEmpty()) {
            return new NDList[] {input,target};
        }

        NDArray[] inputArrays = input.toArray(new NDArray[0]);
        NDArray[] targetArrays = target.toArray(new NDArray[0]);

        Map<IndexKey, Integer> inputMap = new ConcurrentHashMap<>();
        Map<IndexKey, Integer> targetMap = new ConcurrentHashMap<>();
        // create mapping
        for (int i = 0; i < input.size(); i++) {
            String key = input.get(i).getName();
            if (key != null) {
                inputMap.put(new IndexKey(key), i);
            }
            inputMap.put(new IndexKey(i), i);
        }
        for (int i = 0; i < target.size(); i++) {
            String key = target.get(i).getName();
            if (key != null) {
                targetMap.put(new IndexKey(key), i);
            }
            targetMap.put(new IndexKey(i), i);
        }
        // apply transform
        for (Pair<IndexKey, PairedTransform> transform : transforms) {
            IndexKey key = transform.getKey();
            int inputIndex = inputMap.get(key);
            NDArray inputArray = inputArrays[inputIndex];
            
            int targetIndex = targetMap.get(key);
            NDArray targetArray = targetArrays[targetIndex];

            NDArray[] transformResult = transform.getValue().transform(inputArray, targetArray);
            
            inputArrays[inputIndex] = transformResult[0];
            targetArrays[targetIndex] = transformResult[1];
            
            inputArrays[inputIndex].setName(inputArray.getName());
            targetArrays[targetIndex].setName(targetArray.getName());
        }

        return new NDList[] {new NDList(inputArrays),new NDList(targetArrays)};
    }

    private static final class IndexKey {
        private String key;
        private int index;

        private IndexKey(String key) {
            this.key = key;
        }

        private IndexKey(int index) {
            this.index = index;
        }

        /** {@inheritDoc} */
        @Override
        public int hashCode() {
            if (key == null) {
                return index;
            }
            return key.hashCode();
        }

        /** {@inheritDoc} */
        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof IndexKey)) {
                return false;
            }
            IndexKey other = (IndexKey) obj;
            if (key == null) {
                return index == other.index;
            }
            return key.equals(other.key);
        }
    }
}
