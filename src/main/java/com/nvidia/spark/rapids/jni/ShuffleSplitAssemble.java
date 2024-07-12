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

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

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

  public static class SplitResult implements AutoCloseable {
    private final DeviceMemoryBuffer buffer;
    private final int[] offsets;

    SplitResult(long bufferAddr, long bufferSize, long bufferHandle, int[] offsets) {
      this.buffer = DeviceMemoryBuffer.fromRmm(bufferAddr, bufferSize, bufferHandle);
      this.offsets = offsets;
    }

    public DeviceMemoryBuffer getBuffer() {
      return buffer;
    }

    public int[] getOffsets() {
      return offsets;
    }

    @Override
    public void close() {
      buffer.close();
    }
  }

  public static SplitResult split(Table table, int[] splitIndices) {
    return split(table.getNativeView(), splitIndices);
  }

  public static Table assemble(Metadata metadata,
                               BaseDeviceMemoryBuffer parts,
                               BaseDeviceMemoryBuffer partOffsets) {
    return new Table(assemble(
        metadata.getNumChildren(), metadata.getTypes(),
        parts.getAddress(), parts.getLength(),
        partOffsets.getAddress(), partOffsets.getLength()));
  }

  private static native SplitResult split(long table, int[] splitIndices);
  private static native long[] assemble(int[] numChildren, int[] types,
                                        long partsAddr, long partsSize,
                                        long partOffsetsAddr, long partOffsetsSize);
}
