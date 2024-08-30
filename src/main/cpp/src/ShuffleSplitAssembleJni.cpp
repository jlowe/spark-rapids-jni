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

// pad all data buffer sizes to a multiple of 8 to make copying the data easier on the GPU
constexpr static int PAD_SIZE = 8;

std::size_t pad_size(std::size_t s)
{
  return rmm::align_up(size, PAD_SIZE);
}

// empty tables are serialized just as a 4-byte row count
constexpr static int EMPTY_HEADER_SIZE = 4;

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

std::size_t non_empty_header_size(host_table_view const& t)
{
  // header is:
  //   - 4 byte row count
  //   - null mask presence for each column, one bit per column in depth-first traversal
  // padded to multiple of 8 bytes
  std::size_t null_mask_presence_size = (num_total_columns(t) + 7) / 8;
  std::size_t prepadded_size = 4 + null_mask_presence_size;
  return pad_size(prepadded_size);
}

std::size_t get_string_data_split_size(host_column_view const& c, cudf::size_type split_start, cudf::size_type split_rows)
{
  // last offset is the size of the data
  auto const& offsets = c.child(0);
  if (offsets.type.id() == cudf::type_id::INT32) {
    return offsets.data<int32_t const*>()[c.size()];
  } else if (offsets.type.id() == cudf::type_id::INT64) {
    return offsets.data<int64_t const*>()[c.size()];
  } else {
    throw std::runtime_error(std::string("unexpected offset type: "
      + std::to_string(static_cast<int>(dtype.id())));
  }
}

std::size_t get_data_size(host_column_view const& c, cudf::size_type split_start, cudf::size_type split_rows)
{
  auto data = c.data<uint8_t const*>();
  if (data == nullptr) {
    return 0;
  }
  auto dtype = c.type();
  if (data == nullptr) {  if (cudf::is_fixed_width(dtype)) {
    return cudf::size_of(dtype) * split_rows;
  } else if (dtype.id() == cudf::type_id::STRING) {
    return get_string_data_split_size(c, split_start, split_rows)
  } else {
    throw std::runtime_error(std::string("unexpected data type: ")
      + std::to_string(static_cast<int>(dtype.id())));
  }
}

cudf::size_type get_list_child_split_rows(host_column_view const& offsets, cudf::size_type split_start, cudf::size_type split_rows)
{
  auto split_end = split_start + split_rows;
  CUDF_EXPECTS(split_end < offsets.size(), "split range exceeds offsets");
  CUDF_EXPECTS(offsets.type().id() == cudf::type_id::INT32, "offsets are not INT32");
  auto data_ptr = offsets.data<int32_t const*>();
  CUDF_EXPECTS(data_ptr != nullptr, "offsets are missing");
  auto child_split_start = data_ptr[split_start];
  auto child_split_end = data_ptr[split_start + split_rows];
  auto child_split_rows = child_split_end - child_split_start;
  CUDF_EXPECTS(child_split_rows >= 0, "split range is negative");
  return child_split_rows;
}

std::size_t split_size(host_column_view const& c, cudf::size_type split_start, cudf::size_type split_rows)
{
  std::size_t sum = 0;
  if (c.has_nulls()) {
    sum += pad_size((split_rows + 7) / 8);
  }
  sum += pad_size(get_data_size(c));
  if (c.num_children() > 0) {
    auto type_id = c.type().id();
    if (type_id == cudf::type_id::STRING || type_id == cudf::type_id::LIST) {
      // account for size of offsets column which contains one more entry than parent row count
      auto const& offsets = c.child(0);
      sum += cudf::size_of(offsets.type()) * (split_rows + 1);
      if (type_id == cudf::type_id::LIST) {
        auto split_end = split_start + split_rows;
        CUDF_EXPECTS(split_end < offsets.size(), "split range exceeds offsets");
        CUDF_EXPECTS(offsets.type().id() == cudf::type_id::INT32, "offsets are not INT32");
        auto offsets_ptr = offsets.data<int32_t const*>();
        auto child_split_start = data_ptr[split_start];
        auto child_split_rows = data_ptr[split_end] - split_start;
        sum += split_size(c.child(1), child_split_start, child_split_rows);
      }
    }
    else if (type_id == cudf::type_id::STRUCT) {
      sum += std::accumulate(c.child_begin(), c.child_end(), 0,
        [split_start, split_rows](std::size_t s, host_column_view const& child) {
          return s + split_size(child, split_start, split_rows);
        });
    } else {
      throw std::runtime_error(std::string("unexpected type: ") + std::to_string(static_cast<int>(type_id)));
    }
  }
  return sum;
}

std::size_t split_on_host_size(host_table_view const& t, uint8_t const* bp, uint8_t const* bp_end,
                               cudf::jni::native_jintArray const& split_indices)
{
  if (t.num_rows() == 0) {
    return EMPTY_HEADER_SIZE * split_indices.size();
  }
  auto single_header_size = non_empty_header_size(t);
  std::size_t sum = 0;
  for (int i = 0; i < split_indices.size() - 1; i++) {
    auto split_start = static_cast<cudf::size_type>(split_indices[i]);
    auto split_rows = static_cast<cudf::size_type>(split_indices[i + 1] - split_start);
    if (split_rows == 0) {
      sum += EMPTY_HEADER_SIZE;
    } else {
      sum += std::accumulate(t.begin(), t.end(), single_header_size,
        [split_rows, split_start](std::size_t s, host_column_view const& c) {
          return s + split_size(c, split_start, split_rows);
        }
      )
    }
  }
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
