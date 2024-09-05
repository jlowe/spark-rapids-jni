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

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <arpa/inet.h>
#include <cstring>
#include <limits>
#include <tuple>
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
  return rmm::align_up(s, PAD_SIZE);
}

// Empty tables are serialized as a total size and row count. Using cudf::size_type for
// total size since Spark cannot handle serialized objects larger than 2G.
constexpr static int EMPTY_HEADER_SIZE = sizeof(cudf::size_type) * 2;

[[noreturn]] void type_error(cudf::type_id id)
{
  throw std::runtime_error(std::string("unexpected type: ") + std::to_string(static_cast<int>(id)));
}

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
  // offset columns of strings and lists are not counted
  if (c.num_children() > 0) {
    switch (c.type().id()) {
      case cudf::type_id::STRING:
        // string offset column is skipped
        break;
      case cudf::type_id::LIST:
        // list offset column is skipped
        sum += num_total_columns(c.lists_child());
        break;
      case cudf::type_id::STRUCT:
        sum += std::accumulate(c.child_begin(), c.child_end(), 0,
          [](std::size_t sum, host_column_view const& c) { return sum + num_total_columns(c); });
        break;
      default:
        type_error(c.type().id());
    }
  }
  return sum;
}

std::size_t num_total_columns(host_table_view const& t)
{
  return std::accumulate(t.begin(), t.end(), 0,
    [](std::size_t sum, host_column_view const& c) { return sum + num_total_columns(c); });
}

std::size_t non_empty_header_size(host_table_view const& t)
{
  // header is:
  //   - 4 byte total split size
  //   - 4 byte row count
  //   - null mask presence for each column, one bit per column in depth-first traversal
  // padded to multiple of 8 bytes not including initial total size
  std::size_t null_mask_presence_size = (num_total_columns(t) + 7) / 8;
  std::size_t prepadded_size = 4 + null_mask_presence_size;
  return 4 + pad_size(prepadded_size);
}

std::size_t get_data_size(host_column_view const& c, cudf::size_type split_start, cudf::size_type num_rows)
{
  auto data = c.data<uint8_t const*>();
  if (data == nullptr) {
    return 0;
  }
  auto dtype = c.type();
  if (cudf::is_fixed_width(dtype)) {
    return cudf::size_of(dtype) * num_rows;
  } else if (dtype.id() == cudf::type_id::STRING) {
    auto const& offsets_col = c.strings_offsets();
    if (offsets_col.type().id() != cudf::type_id::INT32) {
      type_error(offsets_col.type().id());
    }
    auto offsets = offsets_col.data<int32_t const>();
    return offsets[split_start + num_rows] - offsets[split_start];
  } else {
    type_error(dtype.id());
  }
}

std::pair<cudf::size_type, cudf::size_type>
get_offsets_split_info(host_column_view const& offsets, cudf::size_type split_start, cudf::size_type num_rows)
{
  auto split_end = split_start + num_rows;
  CUDF_EXPECTS(split_end < offsets.size(), "split range exceeds offsets");
  CUDF_EXPECTS(offsets.type().id() == cudf::type_id::INT32, "offsets are not INT32");
  auto data_ptr = offsets.data<int32_t const>();
  CUDF_EXPECTS(data_ptr != nullptr, "offsets are missing");
  auto child_split_start = data_ptr[split_start];
  auto child_split_end = data_ptr[split_start + num_rows];
  auto child_num_rows = child_split_end - child_split_start;
  CUDF_EXPECTS(child_num_rows >= 0, "split range is negative");
  return std::make_pair(child_split_start, child_num_rows);
}

std::size_t split_size(host_column_view const& c, cudf::size_type split_start, cudf::size_type num_rows)
{
  std::size_t sum = 0;
  if (c.has_nulls()) {
    sum += pad_size((num_rows + 7) / 8);
  }
  sum += pad_size(get_data_size(c, split_start, num_rows));
  if (c.num_children() > 0) {
    switch(c.type().id()) {
      case cudf::type_id::STRING:
      {
        // account for size of offsets column which contains one more entry than parent row count
        auto const& offsets = c.strings_offsets();
        sum += pad_size(cudf::size_of(offsets.type()) * (num_rows + 1));
        auto [child_split_start, child_num_rows] = get_offsets_split_info(offsets, split_start, num_rows);
        sum += child_num_rows;
        break;
      }
      case cudf::type_id::LIST:
      {
        // account for size of offsets column which contains one more entry than parent row
        auto const& offsets = c.lists_offsets();
        sum += pad_size(cudf::size_of(offsets.type()) * (num_rows + 1));
        auto [child_split_start, child_num_rows] = get_offsets_split_info(offsets, split_start, num_rows);
        sum += split_size(c.lists_child(), child_split_start, child_num_rows);
        break;
      }
      case cudf::type_id::STRUCT:
        sum += std::accumulate(c.child_begin(), c.child_end(), 0,
          [split_start, num_rows](std::size_t s, host_column_view const& child) {
            return s + split_size(child, split_start, num_rows);
          });
        break;
      default:
        type_error(c.type().id());
    }
  }
  return sum;
}

std::size_t split_on_host_size(host_table_view const& t, cudf::jni::native_jintArray const& split_indices)
{
  if (t.num_rows() == 0) {
    return EMPTY_HEADER_SIZE * split_indices.size();
  }
  auto single_header_size = non_empty_header_size(t);
  std::size_t sum = 0;
  for (int i = 0; i < split_indices.size() - 1; i++) {
    auto split_start = static_cast<cudf::size_type>(split_indices[i]);
    auto num_rows = static_cast<cudf::size_type>(split_indices[i + 1] - split_start);
    if (num_rows == 0) {
      sum += EMPTY_HEADER_SIZE;
    } else {
      sum += std::accumulate(t.begin(), t.end(), single_header_size,
        [num_rows, split_start](std::size_t s, host_column_view const& c) {
          return s + split_size(c, split_start, num_rows);
        });
    }
  }
  return sum;
}

void size_check(uint8_t* op, uint8_t* op_end, std::size_t incr)
{
  if (op + incr > op_end) {
    throw std::logic_error("output buffer overflow");
  }
}

std::pair<uint32_t, int>
update_has_nulls_mask(host_column_view const& c, std::vector<uint32_t>& mask, uint32_t bits, int bit_index)
{
  if (c.has_nulls()) {
    bits |= 1 << bit_index;
  }
  bit_index++;
  if (bit_index == 32) {
    mask.push_back(bits);
    bit_index = 0;
  }
  if (c.num_children() > 0) {
    switch (c.type().id()) {
      case cudf::type_id::STRING:
        // string offset column is skipped
        break;
      case cudf::type_id::LIST:
        // list offset column is skipped
        std::tie(bits, bit_index) = update_has_nulls_mask(c.lists_child(), mask, bits, bit_index);
        break;
      case cudf::type_id::STRUCT:
        std::for_each(c.child_begin(), c.child_end(), [&](host_column_view const& c) {
          std::tie(bits, bit_index) = update_has_nulls_mask(c, mask, bits, bit_index);
        });
        break;
      default:
        type_error(c.type().id());
    }
  }
  return std::make_pair(bits, bit_index);
}

std::vector<uint32_t> compute_has_nulls_mask(host_table_view const& t)
{
  std::vector<uint32_t> mask;
  if (t.num_rows() > 0) {
    uint32_t bits = 0;
    int bit_index = 0;
    std::for_each(t.begin(), t.end(), [&](host_column_view const& c) {
      std::tie(bits, bit_index) = update_has_nulls_mask(c, mask, bits, bit_index);
    });
    if (bit_index != 0) {
      mask.push_back(bits);
    }
    // add padding for row_count + mask_bytes if necessary
    auto mask_byte_size = mask.size() * sizeof(mask[0]);
    auto header_size = sizeof(cudf::size_type) + mask_byte_size;
    auto padded_size = pad_size(header_size);
    if (padded_size != header_size) {
      if (padded_size - header_size != sizeof(mask[0])) {
        throw std::logic_error("incorrect validity mask padding");
      }
      mask.push_back(0);
    }
  }
  return mask;
}

uint8_t* copy_validity(cudf::bitmask_type const* mask, cudf::size_type split_start, cudf::size_type num_rows,
                       uint8_t* out, uint8_t* out_end)
{
  auto const num_mask_bytes = (num_rows + 7) / 8;
  auto const padded_size = pad_size(num_mask_bytes);
  size_check(out, out_end, padded_size);
  auto op = out;
  if (split_start % 8 == 0) {
    auto mask_start = reinterpret_cast<uint8_t const*>(mask) + (split_start/8);
    std::memcpy(op, mask_start, num_mask_bytes);
    op += num_mask_bytes;
    for (std::size_t i = 0; i < padded_size - num_mask_bytes; i++) {
      *op++ = 0;
    }
  } else {
    // TODO: consider adding SSE/AVX/NEON versions of this
    auto const bits_per_word = sizeof(cudf::bitmask_type) * 8;
    auto const split_end = split_start + num_rows;
    auto const num_input_mask_words = ((split_end - 1) / bits_per_word) - (split_start / bits_per_word) + 1;
    if (num_input_mask_words > 0) {
      auto imp = mask + (split_start / bits_per_word);
      auto omp = reinterpret_cast<cudf::bitmask_type*>(op);
      int const shift = split_start % bits_per_word;
      int const merge_shift = shift + bits_per_word;
      static_assert(bits_per_word == 32, "bitmask shifted copy needs update");
      uint64_t bits = htonl(*imp++) >> shift;
      for (std::size_t i = 0; i < num_input_mask_words - 1; i++) {
        bits |= htonl(*imp++) << merge_shift;
        *omp++ = ntohl(static_cast<cudf::bitmask_type>(bits));
        bits >>= 32;
      }
      *omp++ = ntohl(static_cast<cudf::bitmask_type>(bits));
      if (num_input_mask_words % 2 != 0) {
        *omp++ = 0;
      }
      op = reinterpret_cast<uint8_t*>(omp);
    }
  }
  if (static_cast<std::size_t>(op - out) != padded_size) {
    throw std::logic_error("validity copy buffer error");
  }
  return op;
}

uint8_t* copy_data(host_column_view const& c, cudf::size_type split_start, cudf::size_type num_rows,
                   uint8_t* out, uint8_t* out_end)
{
  auto op = out;
  auto data = c.data<uint8_t const>();
  auto dtype = c.type();
  if (data != nullptr) {
    uint8_t const* src = nullptr;
    std::size_t size = 0;
    if (cudf::is_fixed_width(dtype)) {
      auto type_size = cudf::size_of(dtype);
      src = data + (split_start * type_size);
      size = num_rows * type_size;
    } else if (dtype.id() == cudf::type_id::STRING) {
      auto [chars_start, chars_size] = get_offsets_split_info(c.strings_offsets(), split_start, num_rows);
      src = data + chars_start;
      size = chars_size;
    } else {
      type_error(dtype.id());
    }
    auto padded_size = pad_size(size);
    size_check(op, out_end, padded_size);
    std::memcpy(op, src, size);
    op += size;
    for (std::size_t i = 0; i < padded_size - size; i++) {
      *op++ = 0;
    }
  }
  return op;
}

uint8_t* copy_offsets(host_column_view const& c, cudf::size_type split_start, cudf::size_type num_rows,
                      uint8_t* out, uint8_t* out_end)
{
  auto op = out;
  if (num_rows > 0 && c.num_children() > 0) {
    switch(c.type().id()) {
      case cudf::type_id::STRING:
        // Need one more offset entry than number of rows in column
        op = copy_data(c.strings_offsets(), split_start, num_rows + 1, op, out_end);
        break;
      case cudf::type_id::LIST:
        // Need one more offset entry than number of rows in column
        op = copy_data(c.lists_offsets(), split_start, num_rows + 1, op, out_end);
        break;
      case cudf::type_id::STRUCT:
        // no offsets to copy
        break;
      default:
        type_error(c.type().id());
    }
  }
  return op;
}

uint8_t* single_split_on_host(host_column_view const& c, cudf::size_type split_start, cudf::size_type num_rows,
                              uint8_t* out, uint8_t* out_end)
{
  auto op = out;
  if (c.has_nulls()) {
    op = copy_validity(c.null_mask(), split_start, num_rows, op, out_end);
  }
  op = copy_offsets(c, split_start, num_rows, op, out_end);
  op = copy_data(c, split_start, num_rows, op, out_end);
  if (c.num_children() > 0) {
    switch(c.type().id()) {
      case cudf::type_id::LIST:
      {
        auto [child_split_start, child_num_rows] = get_offsets_split_info(c.lists_offsets(), split_start, num_rows);
        op = single_split_on_host(c.lists_child(), child_split_start, child_split_start + child_num_rows, op, out_end);
        break;
      }
      case cudf::type_id::STRUCT:
        std::for_each(c.child_begin(), c.child_end(), [=, &op](host_column_view const& c) {
          op = single_split_on_host(c, split_start, num_rows, op, out_end);
        });
        break;
      case cudf::type_id::STRING:
        // offsets were copied above, nothing left to do.
        break;
      default:
        type_error(c.type().id());
    }
  }
  return op;
}

uint8_t* single_split_on_host(host_table_view const& t, std::vector<uint32_t> const& has_nulls_mask,
                              cudf::size_type split_start, cudf::size_type num_rows, uint8_t* out, uint8_t* out_end)
{
  // write the common header parts, total size will be filled in at the end
  size_check(out, out_end, EMPTY_HEADER_SIZE);
  auto header_p = reinterpret_cast<cudf::size_type*>(out);
  header_p[1] = num_rows;
  auto op = out + EMPTY_HEADER_SIZE;
  if (num_rows > 0) {
    auto has_nulls_mask_byte_size = has_nulls_mask.size() * sizeof(has_nulls_mask[0]);
    size_check(op, out_end, has_nulls_mask_byte_size);
    std::memcpy(op, has_nulls_mask.data(), has_nulls_mask_byte_size);
    op += has_nulls_mask_byte_size;
    std::for_each(t.begin(), t.end(), [=, &op](host_column_view const& c) {
      op = single_split_on_host(c, split_start, num_rows, op, out_end);
    });
  }
  // fill in the total size now that it is known
  auto size = op - out;
  if (size > std::numeric_limits<cudf::size_type>::max()) {
    throw std::runtime_error("maximum split size exceeded");
  }
  header_p[0] = size;
  return op;
}

std::vector<int64_t> split_on_host(host_table_view const& t, uint8_t* out, std::size_t out_size,
                                   cudf::jni::native_jintArray const& split_indices)
{
  auto const num_splits = split_indices.size();
  std::vector<int64_t> offsets;
  offsets.reserve(num_splits);
  auto op = out;
  auto const op_end = out + out_size;
  auto const has_nulls_mask = compute_has_nulls_mask(t);
  for (int i = 0; i < num_splits; i++) {
    cudf::size_type split_start = static_cast<cudf::size_type>(split_indices[i]);
    cudf::size_type split_end = (i == num_splits - 1)
      ? t.num_rows() : static_cast<cudf::size_type>(split_indices[i + 1]);
    auto const num_rows = split_end - split_start;
    op = single_split_on_host(t, has_nulls_mask, split_start, num_rows, op, op_end);
    offsets.push_back(op - out);
  }
  if (op != op_end) {
    throw std::logic_error("output buffer not fully used");
  }
  return offsets;
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
  JNIEnv* env, jclass, jlong jhost_table, jintArray jsplit_indices)
{
  JNI_NULL_CHECK(env, jhost_table, "table is null", 0);
  JNI_NULL_CHECK(env, jsplit_indices, "indices is null", 0);
  try {
    auto t = reinterpret_cast<spark_rapids_jni::host_table_view const*>(jhost_table);
    cudf::jni::native_jintArray split_indices(env, jsplit_indices);
    return static_cast<jlong>(split_on_host_size(*t, split_indices));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_splitOnHost(
  JNIEnv* env, jclass, jlong jhost_table, jlong dest_address, jlong dest_size,
  jintArray jsplit_indices)
{
  JNI_NULL_CHECK(env, jhost_table, "table is null", nullptr);
  JNI_NULL_CHECK(env, jsplit_indices, "indices is null", nullptr);
  JNI_NULL_CHECK(env, dest_address, "dest is null", nullptr);
  try {
    auto t = reinterpret_cast<spark_rapids_jni::host_table_view const*>(jhost_table);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dest_address);
    cudf::jni::native_jintArray split_indices(env, jsplit_indices);
    auto split_offsets = split_on_host(*t, dst_ptr, static_cast<std::size_t>(dest_size), split_indices);
    return cudf::jni::native_jlongArray(env, split_offsets).get_jArray();
  }
  CATCH_STD(env, nullptr);
}

}  // extern "C"
