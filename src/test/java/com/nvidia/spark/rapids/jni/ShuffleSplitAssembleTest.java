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

import ai.rapids.cudf.*;
import com.nvidia.spark.rapids.jni.ShuffleSplitAssemble.DeviceSplitResult;
import com.nvidia.spark.rapids.jni.ShuffleSplitAssemble.HostSplitResult;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ShuffleSplitAssembleTest {
  //@Test
  void testEmptySplitGpu() {
    int[] splitIndices = new int[]{0, 0, 0};
    try (ColumnVector c0 = ColumnVector.fromInts();
         Table t = new Table(c0);
         DeviceSplitResult sr = ShuffleSplitAssemble.splitOnDevice(t, splitIndices)) {
      DeviceMemoryBuffer devOffsets = sr.getOffsets();
      DeviceMemoryBuffer devBuffer = sr.getBuffer();
      assertEquals(splitIndices.length * 4, devOffsets.getLength());
      int emptyPartitionSize = 16;
      assertEquals(splitIndices.length * emptyPartitionSize, devBuffer.getLength());
      try (HostMemoryBuffer offsetsBuffer = copyToHost(devOffsets);
           HostMemoryBuffer buffer = copyToHost(devBuffer)) {
        ByteBuffer offsets = offsetsBuffer.asByteBuffer();
        ByteBuffer data = buffer.asByteBuffer();
        for (int i = 0; i < splitIndices.length; i++) {
          assertEquals(i * emptyPartitionSize, offsets.getLong());
          // row count
          assertEquals(0, data.getInt());
          // has validity mask
          assertEquals(0, data.getInt());
          // padding to 16 bytes
          assertEquals(0, data.getInt());
          assertEquals(0, data.getInt());
        }
        assertEquals(0, offsets.remaining());
        assertEquals(0, data.remaining());
      }
    }
  }

  @Test
  void testEmptySplitHost() {
    int[] splitIndices = new int[]{0, 0, 0};
    try (ColumnVector c0 = ColumnVector.fromInts();
         Table t = new Table(c0);
         HostTable ht = HostTable.fromTable(t, Cuda.DEFAULT_STREAM);
         HostSplitResult sr = ShuffleSplitAssemble.splitOnHost(ht, splitIndices)) {
      long[] offsets = sr.getOffsets();
      assertEquals(splitIndices.length, offsets.length);
      HostMemoryBuffer buffer = sr.getBuffer();
      ByteBuffer bb = buffer.asByteBuffer();
      int emptyHeaderSize = 8;
      for (int i = 0; i < splitIndices.length; i++) {
        assertEquals(i * emptyHeaderSize, offsets[i]);
        // total size of payload, should be just 4 bytes for the row count
        assertEquals(4, bb.getInt());
        // row count should be zero
        assertEquals(0, bb.getInt());
      }
    }
  }

  @Test
  void testEmptyRoundTripHost() {
    int[] splitIndices = new int[]{0, 0, 0};
    try (ColumnVector c0 = ColumnVector.fromInts();
         Table t = new Table(c0);
         HostTable htin = HostTable.fromTable(t, Cuda.DEFAULT_STREAM);
         HostSplitResult sr = ShuffleSplitAssemble.splitOnHost(htin, splitIndices)) {
      // build offsets with partition total size skipped
      long[] offsets = new long[sr.getOffsets().length];
      for (int i = 0; i < offsets.length; i++) {
        offsets[i] = sr.getOffsets()[i] + 4;
      }
      int[] meta = new int[]{DType.DTypeEnum.INT32.getNativeId(), 0};
      try (HostTable htout = ShuffleSplitAssemble.concatToHostTable(meta, sr.getBuffer(), offsets);
           Table actual = htout.toTable()) {
        AssertUtils.assertTablesAreEqual(t, actual);
      }
    }
  }

  @Test
  void testSimpleSplitNoNullsGpu() {
    int[] splitIndices = new int[]{3, 3, 5, 6};
    try (Table t = new Table.TestBuilder().column(7, 9, 1, -10, -1, -4).build();
         DeviceSplitResult sr = ShuffleSplitAssemble.splitOnDevice(t, splitIndices)) {
      DeviceMemoryBuffer devOffsets = sr.getOffsets();
      DeviceMemoryBuffer devBuffer = sr.getBuffer();
      assertEquals((splitIndices.length + 1) * Long.BYTES, devOffsets.getLength());
      try (HostMemoryBuffer offsetsBuffer = copyToHost(devOffsets);
           HostMemoryBuffer buffer = copyToHost(devBuffer)) {
        ByteBuffer offsets = offsetsBuffer.asByteBuffer();
        ByteBuffer data = buffer.asByteBuffer();
        System.out.print("OFFSETS: ");
        debug(offsets);
        System.out.print("DATA: ");
        debug(data);
        // Check partition 0
        assertEquals(0, offsets.getLong());
        // row count
        assertEquals(3, data.getInt());
        // no columns with nulls
        assertEquals(0, data.getInt());
        // padding to 16 bytes
        data.getLong();
        // data values
        assertEquals(7, data.getInt());
        assertEquals(9, data.getInt());
        assertEquals(1, data.getInt());
        // padding to a multiple of 16 bytes
        data.getInt();
        // Check partition 1
        assertEquals(data.position(), offsets.getLong());
        // row count
        assertEquals(0, data.getInt());
        // padding to 16 bytes
        data.getInt();
        data.getLong();
        // Check partition 2
        assertEquals(data.position(), offsets.getLong());
        // row count
        assertEquals(2, data.getInt());
        // no columns with nulls
        assertEquals(0, data.getInt());
        // padding to 16 bytes
        data.getLong();
        // data values, skip null checks since they could be anything
        assertEquals(-10, data.getInt());
        assertEquals(-1, data.getInt());
        // padding to 16 bytes
        data.getLong();
        // Check partition 3
        assertEquals(data.position(), offsets.getLong());
        // row count
        assertEquals(1, data.getInt());
        // no columns with nulls
        assertEquals(0, data.getInt());
        // padding to 16 bytes
        data.getLong();
        // data values padded to 16 bytes
        assertEquals(-4, data.getInt());
        assertEquals(0, data.getInt());
        assertEquals(0, data.getLong());
        // Check partition 4
        assertEquals(data.position(), offsets.getLong());
        // row count
        assertEquals(0, data.getInt());
        // padding to 16 bytes
        data.getInt();
        data.getLong();
        // verify all values have been examined
        assertEquals(0, offsets.remaining());
        assertEquals(0, data.remaining());
      }
    }
  }

  @Test
  void testSimpleSplitHost() {
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
      assertEquals(32, bb.getInt());
      // row count
      assertEquals(3, bb.getInt());
      // first column has a null mask
      assertEquals(1, bb.getInt());
      // null mask has all three values valid
      assertEquals(7, bb.getInt() & 0x7);
      // padding to a multiple of 8 bytes
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
      // first column has a null mask
      assertEquals(1, bb.getInt());
      // validity mask padded to 8 bytes
      assertEquals(2, bb.getInt() & 0x3);
      assertEquals(0, bb.getInt());
      // data values, skip null checks since they could be anything
      bb.getInt();
      assertEquals(-1, bb.getInt());

      // Check partition 3
      assertEquals(bb.position(), offsets[3]);
      // total size of partition
      assertEquals(24, bb.getInt());
      // row count
      assertEquals(1, bb.getInt());
      // first column has a null mask
      assertEquals(1, bb.getInt());
      // validity mask padded to 8 bytes
      assertEquals(1, bb.getInt() & 0x1);
      assertEquals(0, bb.getInt());
      // data values padded to 8 bytes
      assertEquals(-4, bb.getInt());
      assertEquals(0, bb.getInt());

      assertEquals(0, bb.remaining());
    }
  }

  @Test
  void testSimpleRoundTripNoNullsGpu() {
    int[] splitIndices = new int[]{3, 3, 5, 6};
    try (Table t = new Table.TestBuilder().column(7, 9, 1, -10, -1, -4).build();
         DeviceSplitResult sr = ShuffleSplitAssemble.splitOnDevice(t, splitIndices)) {
      int[] meta = new int[]{DType.DTypeEnum.INT32.getNativeId(), 0};
      DeviceMemoryBuffer offsets = sr.getOffsets();
      DeviceMemoryBuffer data = sr.getBuffer();
      try (Table result = ShuffleSplitAssemble.assembleOnDevice(meta, data, offsets)) {
        AssertUtils.assertTablesAreEqual(t, result);
      }
    }
  }

  //@Test
  void testSimpleRoundTripHost() {
    int[] splitIndices = new int[]{0, 3, 3, 5};
    try (Table t = new Table.TestBuilder().column(7, 9, 1, null, -1, -4).build();
         HostTable htin = HostTable.fromTable(t, Cuda.DEFAULT_STREAM);
         HostSplitResult sr = ShuffleSplitAssemble.splitOnHost(htin, splitIndices)) {
      // build offsets with partition total size skipped
      long[] offsets = new long[sr.getOffsets().length];
      for (int i = 0; i < offsets.length; i++) {
        offsets[i] = sr.getOffsets()[i] + 4;
      }
      int[] meta = new int[]{DType.DTypeEnum.INT32.getNativeId(), 0};
      try (HostTable htout = ShuffleSplitAssemble.concatToHostTable(meta, sr.getBuffer(), offsets);
           Table actual = htout.toTable()) {
        AssertUtils.assertTablesAreEqual(t, actual);
      }
    }
  }

  private HostMemoryBuffer copyToHost(DeviceMemoryBuffer buffer) {
    try (HostMemoryBuffer hmb = HostMemoryBuffer.allocate(buffer.getLength())) {
      hmb.copyFromDeviceBuffer(buffer);
      hmb.incRefCount();
      return hmb;
    }
  }

  private void debug(ByteBuffer bb) {
    bb.mark();
    while (bb.hasRemaining()) {
      System.out.print(Integer.toHexString(bb.getInt()) + " ");
    }
    System.out.println();
    bb.reset();
  }
}
