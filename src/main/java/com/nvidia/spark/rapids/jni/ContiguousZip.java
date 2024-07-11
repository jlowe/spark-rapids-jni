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

import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.Table;

public class ContiguousZipSplit {
  public static class SplitResult implements AutoCloseable {
    private final DeviceMemoryBuffer buffer;
    private final DeviceMemoryBuffer offsets;

    SplitResult(DeviceMemoryBuffer buffer, DeviceMemoryBuffer offsets) {
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

  public static SplitResult split(Table table, int[] splitIndices) {
    long[] results = split(table.getNativeView(), splitIndices);
    long bufferAddr = results[0];
    long bufferSize = results[1];
    long bufferHandle = results[2];
    long offsetsAddr = results[3];
    long offsetsSize = results[4];
    long offsetsHandle = results[5];
    DeviceMemoryBuffer buffer = DeviceMemoryBuffer.fromRmm(bufferAddr, bufferSize, bufferHandle);
    DeviceMemoryBuffer offsets = DeviceMemoryBuffer.fromRmm(offsetsAddr, offsetsSize,
        offsetsHandle);
    return new SplitResult(buffer, offsets);
  }

  private static native long[] split(long table, int[] splitIndices);
}
