
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

package org.datavec.image.data;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
@Data
public class Image {
    private INDArray image;
    private int origC;
    private int origH;
    private int origW;

    public Image(INDArray image, int origC, int origH, int origW){
        this.image = image;
        this.origC = origC;
        this.origH = origH;
        this.origW = origW;

    }

    public int getOrigC() {
        return origC;
    }

    public int getOrigH() {
        return origH;
    }

    public int getOrigW() {
        return origW;
    }

    public INDArray getImage() {
        return image;
    }
}