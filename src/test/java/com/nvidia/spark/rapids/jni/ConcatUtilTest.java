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
import ai.rapids.cudf.HostColumnVector.StructData;
import ai.rapids.cudf.HostColumnVector.StructType;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.util.Arrays;
import java.util.Collections;

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
      byte[] rawData = bout.toByteArray();
      try (HostMemoryBuffer hmb = HostMemoryBuffer.allocate(2L * rawData.length)) {
        // make two copies of the serialized table to concatenate together
        hmb.setBytes(0, rawData, 0, rawData.length);
        hmb.setBytes(rawData.length, rawData, 0, rawData.length);
        long[] headerAddrs = new long[]{ hmb.getAddress(), hmb.getAddress() + rawData.length };
        long[] cpuDataRanges = new long[]{
            headerAddrs[0] + headerSize, rawData.length - headerSize,
            headerAddrs[1] + headerSize, rawData.length - headerSize };
        try (Table actual = ConcatUtil.concatSerializedTables(headerAddrs, cpuDataRanges)) {
          AssertUtils.assertTablesAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  void testConcat() throws Exception {
    DataType listStringsType = new ListType(true, new BasicType(true, DType.STRING));
    DataType mapType = new ListType(true,
        new StructType(true,
            new BasicType(false, DType.STRING),
            new BasicType(false, DType.STRING)));
    DataType structType = new StructType(true,
        new BasicType(true, DType.INT32),
        new BasicType(false, DType.FLOAT32));
    try (ColumnVector ints = ColumnVector.fromBoxedInts(0, 1, null, 3, 4);
         ColumnVector longs = ColumnVector.fromBoxedLongs(null, null, 7L, 8L, 9L);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(-1., 2., -3., 4.5, null);
         ColumnVector strings = ColumnVector.fromStrings("a", "bcd", "defg", null, "");
         ColumnVector listStrings = ColumnVector.fromLists(listStringsType,
             Arrays.asList(null, "foo", "bar"),
             Arrays.asList("x", "y", "z", "w"),
             Arrays.asList(null, null, null, null, null),
             Collections.emptyList(),
             Arrays.asList("the", "quick", "brown", "fox"));
         ColumnVector maps = ColumnVector.fromLists(mapType,
             Arrays.asList(new StructData("a", "b"), null),
             Arrays.asList(new StructData("ax", "bx"), new StructData("ay", "by")),
             Collections.emptyList(),
             Arrays.asList(new StructData("p", "q"), new StructData("z", "z"), new StructData("s", "t")),
             null);
         ColumnVector structs = ColumnVector.fromStructs(structType,
             new StructData(42, 0.123f),
             null,
             new StructData(null, -0.00007f),
             new StructData(52, 0.0001012f),
             new StructData(null, -321.09f));
         Table table1 = new Table(ints, doubles, strings,
             listStrings, maps, structs, longs);
         ColumnVector ints2 = ColumnVector.fromInts(12, 10);
         ColumnVector longs2 = ColumnVector.fromLongs(80, 90);
         ColumnVector doubles2 = ColumnVector.fromDoubles(-10., 20.);
         ColumnVector strings2 = ColumnVector.fromStrings("apple", "banana");
         ColumnVector listStrings2 = ColumnVector.fromLists(listStringsType,
             Arrays.asList("foobar", "fizgig"),
             Arrays.asList("s"));
         ColumnVector maps2 = ColumnVector.fromLists(mapType,
             Arrays.asList(new StructData("a2", "b2")),
             Arrays.asList(new StructData("c2", "d2"), new StructData("e2", "f2")));
         ColumnVector structs2 = ColumnVector.fromStructs(structType,
             new StructData(72, -0.321f),
             new StructData(-127, -642.0919f));
         Table table2 = new Table(ints2, doubles2, strings2,
             listStrings2, maps2, structs2, longs2);
         Table expected = Table.concatenate(table1, table2)) {
      // serialize the tables
      ByteArrayOutputStream bout1 = new ByteArrayOutputStream();
      JCudfSerialization.writeToStream(table1, bout1, 0, table1.getRowCount());
      ByteArrayOutputStream bout2 = new ByteArrayOutputStream();
      JCudfSerialization.writeToStream(table2, bout2, 0, table2.getRowCount());
      // compute header sizes
      byte[] rawData1 = bout1.toByteArray();
      ByteArrayInputStream bin = new ByteArrayInputStream(rawData1);
      DataInputStream din = new DataInputStream(bin);
      JCudfSerialization.SerializedTableHeader header = new JCudfSerialization.SerializedTableHeader(din);
      long headerSize1 = header.getSerializedHeaderSizeInBytes();
      byte[] rawData2 = bout2.toByteArray();
      bin = new ByteArrayInputStream(rawData2);
      din = new DataInputStream(bin);
      header = new JCudfSerialization.SerializedTableHeader(din);
      long headerSize2 = header.getSerializedHeaderSizeInBytes();
      try (HostMemoryBuffer hmb1 = HostMemoryBuffer.allocate(rawData1.length);
           HostMemoryBuffer hmb2 = HostMemoryBuffer.allocate(rawData2.length)) {
        hmb1.setBytes(0, rawData1, 0, rawData1.length);
        hmb2.setBytes(0, rawData2, 0, rawData2.length);
        long[] headerAddrs = new long[]{ hmb1.getAddress(), hmb2.getAddress() };
        long[] cpuDataRanges = new long[]{
            headerAddrs[0] + headerSize1, rawData1.length - headerSize1,
            headerAddrs[1] + headerSize2, rawData2.length - headerSize2 };
        try (Table actual = ConcatUtil.concatSerializedTables(headerAddrs, cpuDataRanges)) {
          TableDebug.get().debug("actual", actual);
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
