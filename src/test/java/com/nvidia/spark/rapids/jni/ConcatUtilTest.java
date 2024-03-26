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
import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.DataType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructType;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.util.Arrays;

public class ConcatUtilTest {
  @Test
  void testConcatEmpty() throws Exception {
    DataType listStringsType = new ListType(true, new BasicType(true, DType.STRING));
    DataType mapType = new ListType(true,
        new StructType(true,
            new BasicType(false, DType.STRING),
            new BasicType(false, DType.STRING)));
    DataType structType = new StructType(true,
        new BasicType(true, DType.INT8),
        new BasicType(false, DType.FLOAT32));
    try (ColumnVector emptyInt = ColumnVector.fromInts();
         ColumnVector emptyDouble = ColumnVector.fromDoubles();
         ColumnVector emptyString = ColumnVector.fromStrings();
         ColumnVector emptyListString = ColumnVector.fromLists(listStringsType);
         ColumnVector emptyMap = ColumnVector.fromLists(mapType);
         ColumnVector emptyStruct = ColumnVector.fromStructs(structType);
         Table expected = new Table(emptyInt, emptyInt, emptyDouble, emptyString,
             emptyListString, emptyMap, emptyStruct)) {
      // serialize the empty table
      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      JCudfSerialization.writeToStream(expected, bout, 0, 0);
      // compute header size
      ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
      DataInputStream din = new DataInputStream(bin);
      JCudfSerialization.SerializedTableHeader header = new JCudfSerialization.SerializedTableHeader(din);
      long headerSize = header.getSerializedHeaderSizeInBytes();
      assert headerSize < Integer.MAX_VALUE;
      byte[] rawData = bout.toByteArray();
      System.out.println("RAW DATA: " + Arrays.toString(rawData));
      try (HostMemoryBuffer hmb = HostMemoryBuffer.allocate(2L * rawData.length)) {
        // make two copies of the serialized table to concatenate together
        hmb.setBytes(0, rawData, 0, rawData.length);
        hmb.setBytes(rawData.length, rawData, 0, rawData.length);
        long[] headerAddrs = new long[]{ hmb.getAddress(), hmb.getAddress() + rawData.length };
        long[] cpuDataRanges = new long[]{
            headerAddrs[0] + headerSize, rawData.length - headerSize,
            headerAddrs[1] + headerSize, rawData.length - headerSize };
        System.out.println(Arrays.toString(headerAddrs));
        System.out.println(Arrays.toString(cpuDataRanges));
        try (Table actual = ConcatUtil.concatSerializedTables(headerAddrs, cpuDataRanges)) {
          AssertUtils.assertTablesAreEqual(expected, actual);
        }
      }
    }
  }

  public static void main(String[] argv) throws Exception {
    ConcatUtilTest t = new ConcatUtilTest();
    t.testConcatEmpty();
  }
}
