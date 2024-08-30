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

#include "cudf_jni_apis.hpp"
#include "host_table_view.hpp"

#include <cudf/utilities/span.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

//====================
//temporary declarations until headers are available
#include <cudf/types.hpp>
namespace spark_rapids_jni {
struct shuffle_split_col_data {
  cudf::size_type num_children;
  cudf::type_id type;
};

struct shuffle_split_metadata {
  // depth-first traversal
  // - a fixed column or strings have 0
  // - a list column always has 1 (I guess we could also make this
  //   0 to make it more string-like. in my brain I always think of lists
  //   as explicitly having a child because the type is completely arbitrary)
  // - structs have N
  //
  std::vector<shuffle_split_col_data> col_info{};
};
struct shuffle_split_result {
  // packed partition buffers. very similar to what would have come out of
  // contiguous_split, except:
  // - it is one big buffer where all of the partitions are glued together instead
  //   of one buffer per-partition
  // - each partition is prepended by some data-dependent info: row count, presence-of-validity
  // etc
  std::unique_ptr<rmm::device_buffer>    partitions;
  // offsets into the partition buffer for each partition. offsets.size() will be
  // num partitions
  std::unique_ptr<rmm::device_uvector<size_t>>   offsets;
};
shuffle_split_result shuffle_split(cudf::table_view const& input,
                        shuffle_split_metadata const& global_metadata,
                        std::vector<cudf::size_type> const& splits,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::table> shuffle_assemble(shuffle_split_metadata const& global_metadata,
                                              cudf::device_span<int8_t const> partitions,
                                              cudf::device_span<size_t const> partition_offsets);
}
//====================

namespace {

using spark_rapids_jni::host_column_view;
using spark_rapids_jni::host_table_view;

spark_rapids_jni::shuffle_split_metadata to_metadata(JNIEnv* env, jintArray jmeta)
{
  cudf::jni::native_jintArray metadata(env, jmeta);
  std::vector<spark_rapids_jni::shuffle_split_col_data> result;
  result.reserve(metadata.size());
  for (int i = 0; i < metadata.size(); i += 2) {
    auto num_children = metadata[i];
    auto type_id = static_cast<cudf::type_id>(metadata[i+1]);
    result.push_back({num_children, type_id});
  }
  metadata.cancel();
  return spark_rapids_jni::shuffle_split_metadata{std::move(result)};
}

std::size_t num_total_columns(host_column_view const& c)
{
  std::size_t sum = 1;  // include the current column
  return std::accumulate(c.child_begin(), c.child_end(), sum,
    [](std::size_t sum, host_column_view const& c) { return sum + num_total_columns(c); });
}

std::size_t num_total_columns(host_table_view const& t)
{
  std::size_t sum = 0;
  return std::accumulate(t.begin(), t.end(), sum,
    [](std::size_t sum, host_column_view const& c) { return sum + num_total_columns(c); });
}

std::size_t header_size(host_table_view const& t)
{
  // header is:
  //   - 4 byte row count
  //   - null mask presence for each column, one bit per column in depth-first traversal
  // padded to multiple of 16 bytes
  std::size_t null_mask_presence_size = (num_total_columns(t) + 7) / 8;
  std::size_t prepadded_size = 4 + null_mask_presence_size;
  return rmm::align_up(prepadded_size, 16);
}

std::size_t total_headers_size(host_table_view const& t, cudf::jni::native_jintArray split_indices)
{
  std::size_t single_header_size = header_size(t);
  std::size_t sum = 0;
  if (split_indices.size() > 0) {
    if (split_indices[0] != 0) {
      throw std::runtime_error("first split does not start at 0");
    }
    for (int i = 0; i < split_indices.size() - 1; i++) {
      if (split_indices[i] != split_indices[i + 1]) {
        sum += single_header_size;
      }
    }
    if (split_indices[split_indices.size() - 1] != t.num_rows()) {
      sum +== single_header_size;
    }
  }
}

std::size_t split_on_host_size(host_table_view const& t, uint8_t const* bp,
                               uint8_t const* bp_end, cudf::jni::native_jintArray split_indices)
{
  std::size_t total_size = total_headers_size(t, split_indices);

}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_splitOnDevice(
  JNIEnv* env, jclass, jintArray jmeta, jlong jtable, jintArray jsplit_indices)
{
  JNI_NULL_CHECK(env, jmeta, "metadata is null", nullptr);
  JNI_NULL_CHECK(env, jtable, "input table is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto const table = reinterpret_cast<cudf::table_view const*>(jtable);
    auto metadata = to_metadata(env, jmeta);
    cudf::jni::native_jintArray indices(env, jsplit_indices);
    auto split_result = spark_rapids_jni::shuffle_split(*table, metadata, indices.to_vector(),
                                                        cudf::get_default_stream(),
                                                        rmm::mr::get_current_device_resource());
    indices.cancel();
    cudf::jni::native_jlongArray result(env, 6);
    auto offsets_buffer = std::make_unique<rmm::device_buffer>(split_result.offsets->release());
    result[0] = cudf::jni::ptr_as_jlong(split_result.partitions->data());
    result[1] = split_result.partitions->size();
    result[2] = cudf::jni::release_as_jlong(split_result.partitions);
    result[3] = cudf::jni::ptr_as_jlong(offsets_buffer->data());
    result[4] = offsets_buffer->size();
    result[5] = cudf::jni::release_as_jlong(offsets_buffer);
    return result.get_jArray();
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_assembleOnDevice(
  JNIEnv* env, jclass, jintArray jmeta, jlong parts_addr, jlong parts_size,
  jlong offsets_addr, jlong offsets_count)
{
  JNI_NULL_CHECK(env, jmeta, "metadata is null", nullptr);
  JNI_NULL_CHECK(env, parts_addr, "partitions buffer is null", nullptr);
  JNI_NULL_CHECK(env, offsets_addr, "offsets buffer is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto const parts_ptr = reinterpret_cast<int8_t const*>(parts_addr);
    cudf::device_span<int8_t const> parts_span{parts_ptr, static_cast<size_t>(parts_size)};
    auto const offsets_ptr = reinterpret_cast<size_t const*>(offsets_addr);
    cudf::device_span<size_t const> offsets_span{offsets_ptr, static_cast<size_t>(offsets_count)};
    auto meta = to_metadata(env, jmeta);
    auto table = spark_rapids_jni::shuffle_assemble(meta, parts_span, offsets_span);
    return cudf::jni::convert_table_for_return(env, table);
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_splitOnHostSize(
  JNIEnv* env, jclass, jlong jhost_table, jlong data_address, jlong data_size,
  jlongArray jsplit_indices)
{
  JNI_NULL_CHECK(env, jhost_table, "table is null", 0);
  JNI_NULL_CHECK(env, jsplit_indices, "indices is null", 0);
  try {
    auto t = reinterpret_cast<spark_rapids_jni::host_table_view const*>(jhost_table);
    auto bp = reinterpret_cast<uint8_t const*>(data_address);
    auto bp_end = bp + data_size;
    cudf::jni::native_jintArray split_indices(env, jsplit_indices);
    return static_cast<jlong>(split_on_host_size(*t, bp, bp_end, split_indices));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_splitOnHost(
  JNIEnv* env, jclass, jlong jhost_table, jlong data_address, jlong dest_address, jlong dest_size)
{
  JNI_NULL_CHECK(env, jhost_table, "table is null", 0);
  try {
    auto t = reinterpret_cast<spark_rapids_jni::host_table_view const*>(jhost_table);
    auto src_ptr = reinterpret_cast<uint8_t const*>(data_address);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dest_address);
    todo
  }
  CATCH_STD(env, nullptr);
}

}  // extern "C"
