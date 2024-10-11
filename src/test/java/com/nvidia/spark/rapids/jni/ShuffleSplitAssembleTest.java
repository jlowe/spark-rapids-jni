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
  void testSimpleSplit() {
    int[] splitIndices = new int[]{0, 3, 3, 5};
    try (Table t = new Table.TestBuilder().column(7, 9, 1, null, -1, -4).build();
         HostTable ht = HostTable.fromTable(t, Cuda.DEFAULT_STREAM);
         HostSplitResult sr = ShuffleSplitAssemble.splitOnHost(ht, splitIndices)) {
      long[] offsets = sr.getOffsets();
      assertEquals(splitIndices.length, offsets.length);
      HostMemoryBuffer buffer = sr.getBuffer();
      ByteBuffer bb = buffer.asByteBuffer();
      // Check partition 0
      assertEquals(0, offsets[0]);
      // total size of partition
      assertEquals(24, bb.getInt());
      // row count
      assertEquals(3, bb.getInt());
      // no null masks
      assertEquals(0, bb.getInt());
      // data values
      assertEquals(7, bb.getInt());
      assertEquals(9, bb.getInt());
      assertEquals(1, bb.getInt());
      // padding to a multiple of 8 bytes
      assertEquals(0, bb.getInt());

      // Check partition 1
      assertEquals(bb.position(), offsets[1]);
      // total size of partition
      assertEquals(4, bb.getInt());
      // row count
      assertEquals(0, bb.getInt());

      // Check partition 2
      assertEquals(bb.position(), offsets[2]);
      // total size of partition
      assertEquals(24, bb.getInt());
      // row count
      assertEquals(2, bb.getInt());
      // null mask is present
      assertEquals(1, bb.getInt());
      // validity mask padded to 8 bytes
      assertEquals(1, bb.getInt());
      assertEquals(0, bb.getInt());
      // data values, skip null checks since they could be anything
      bb.getInt();
      assertEquals(-1, bb.getInt());

      // Check partition 3
      assertEquals(bb.position(), offsets[3]);
      // total size of partition
      assertEquals(20, bb.getInt());
      // row count
      assertEquals(1, bb.getInt());
      // no null masks
      assertEquals(0, bb.getInt());
      // data values padded to 8 bytes
      assertEquals(-4, bb.getInt());
      assertEquals(0, bb.getInt());

      assertEquals(0, bb.remaining());
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
