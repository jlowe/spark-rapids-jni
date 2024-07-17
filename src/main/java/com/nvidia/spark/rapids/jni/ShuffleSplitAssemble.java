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

public class ShuffleSplitAssemble {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static class Metadata {
    private final int[] numChildren;
    private final int[] types;

    public Metadata(int[] numChildren, int[] types) {
      this.numChildren = numChildren;
      this.types = types;
    }

    public int[] getNumChildren() {
      return numChildren;
    }

    public int[] getTypes() {
      return types;
    }
  }

  public static class DeviceSplitResult implements AutoCloseable {
    private final DeviceMemoryBuffer buffer;
    private final DeviceMemoryBuffer offsets;

    DeviceSplitResult(DeviceMemoryBuffer buffer, DeviceMemoryBuffer offsets) {
      this.buffer = buffer;
      this.offsets = offsets;
    }

    public DeviceMemoryBuffer getBuffer() {
      return buffer;
    }

    public DeviceMemoryBuffer getOffsets() {
      return offsets;
    }

    @Override
    public void close() {
      buffer.close();
      offsets.close();
    }
  }

  public static class HostSplitResult implements AutoCloseable {
    private final HostMemoryBuffer buffer;
    private final HostMemoryBuffer offsets;

    HostSplitResult(HostMemoryBuffer buffer, HostMemoryBuffer offsets) {
      this.buffer = buffer;
      this.offsets = offsets;
    }

    public HostMemoryBuffer getBuffer() {
      return buffer;
    }

    public HostMemoryBuffer getOffsets() {
      return offsets;
    }

    @Override
    public void close() {
      buffer.close();
      offsets.close();
    }
  }

  public static DeviceSplitResult splitOnDevice(Metadata meta, Table table, int[] splitIndices) {
    long[] result = splitOnDevice(meta.getNumChildren(), meta.getTypes(), table.getNativeView(),
        splitIndices);
    assert(result.length == 6);
    long bufferAddr = result[0];
    long bufferSize = result[1];
    long bufferHandle = result[2];
    long offsetsAddr = result[3];
    long offsetsSize = result[4];
    long offsetsHandle = result[5];
    DeviceMemoryBuffer buffer = DeviceMemoryBuffer.fromRmm(bufferAddr, bufferSize, bufferHandle);
    DeviceMemoryBuffer offsets = DeviceMemoryBuffer.fromRmm(offsetsAddr, offsetsSize,
        offsetsHandle);
    return new DeviceSplitResult(buffer, offsets);
  }

  public static HostSplitResult splitOnHost(Metadata meta, ColumnVector[] columns, int[] splitIndices) {
    throw new UnsupportedOperationException();
  }

  public static Table assembleOnDevice(Metadata metadata,
                                       BaseDeviceMemoryBuffer parts,
                                       BaseDeviceMemoryBuffer partOffsets) {
    // offsets buffer must be an array of long values
    assert(partOffsets.getLength() % 8 == 0);
    return new Table(assembleOnDevice(
        metadata.getNumChildren(), metadata.getTypes(),
        parts.getAddress(), parts.getLength(),
        partOffsets.getAddress(), partOffsets.getLength() / 8));
  }

  public static HostMemoryBuffer concatOnHost(Metadata metadata,
                                              HostMemoryBuffer parts,
                                              HostMemoryBuffer partOffsets) {
    // offsets buffer must be an array of long values
    assert(partOffsets.getLength() % 8 == 0);
    throw new UnsupportedOperationException();
  }
  private static native long[] splitOnDevice(int[] metaNumChildren, int[] metaTypes, long table,
                                             int[] splitIndices);
  private static native long[] assembleOnDevice(int[] numChildren, int[] types,
                                                long partsAddr, long partsSize,
                                                long partOffsetsAddr, long partOffsetsCount);
}
