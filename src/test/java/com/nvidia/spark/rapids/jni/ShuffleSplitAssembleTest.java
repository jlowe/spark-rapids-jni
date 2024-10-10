/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Table;
import com.nvidia.spark.rapids.jni.ShuffleSplitAssemble.HostSplitResult;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ShuffleSplitAssembleTest {
  @Test
  void testEmptySplitHost() {
    int[] splitIndices = new int[]{0, 0, 0};
    try (ColumnVector c0 = ColumnVector.fromInts();
         Table t = new Table(c0);
         HostTable ht = HostTable.fromTable(t, Cuda.DEFAULT_STREAM);
         HostSplitResult sr = ShuffleSplitAssemble.splitOnHost(ht, splitIndices)) {
      assertEquals(splitIndices.length, sr.getOffsets().length);
      HostMemoryBuffer buffer = sr.getBuffer();

      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < buffer.getLength(); i++) {
        sb.append(String.format("%02x", buffer.getByte(i)));
      }
      System.out.println(sb.toString());

      int emptyHeaderSize = 8;
      assertEquals(splitIndices.length * emptyHeaderSize, buffer.getLength());
      ByteBuffer bb = buffer.asByteBuffer();
      for (int i = 0; i < splitIndices.length; i++) {
        // total size of payload, should be just 4 bytes for the row count
        assertEquals(4, bb.getInt());
        // row count should be zero
        assertEquals(0, bb.getInt());
      }
    }
  }

  @Test
  void testEmptyRoundTrip() {
    int[] splitIndices = new int[]{0, 0, 0};
    try (ColumnVector c0 = ColumnVector.fromInts();
         Table t = new Table(c0);
         HostTable ht = HostTable.fromTable(t, Cuda.DEFAULT_STREAM)) {

    }
  }
}
