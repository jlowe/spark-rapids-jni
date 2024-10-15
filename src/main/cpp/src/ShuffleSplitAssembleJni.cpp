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

#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

//====================
//temporary declarations until headers are available
#include <cudf/types.hpp>
namespace spark_rapids_jni {
struct shuffle_split_col_data {
  cudf::data_type dtype;
  cudf::size_type num_children;
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

constexpr static int NULL_MASK_WORD_BITS = sizeof(cudf::bitmask_type) * 8;

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
    auto type_id = static_cast<cudf::type_id>(metadata[i]);
    cudf::data_type dtype(type_id);
    auto num_children = 0;
    switch (type_id) {
      case cudf::type_id::STRUCT:
        num_children = metadata[i + 1];
        break;
      case cudf::type_id::LIST:
        num_children = 1;
        break;
      case cudf::type_id::DECIMAL32: [[fallthrough]];
      case cudf::type_id::DECIMAL64: [[fallthrough]];
      case cudf::type_id::DECIMAL128:
        dtype = cudf::data_type(type_id, metadata[i + 1]);
        break;
      default:
        break;
    }
    result.push_back({dtype, num_children});
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

std::size_t non_empty_header_size(std::size_t num_columns)
{
  // header is:
  //   - 4 byte row count
  //   - null mask presence for each column, one bit per column in depth-first traversal
  // padded to multiple of 8 bytes
  std::size_t null_mask_presence_size = (num_columns + 7) / 8;
  std::size_t prepadded_size = 4 + null_mask_presence_size;
  return pad_size(prepadded_size);
}

std::size_t non_empty_header_size(host_table_view const& t)
{
  return non_empty_header_size(num_total_columns(t));
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
  // total size prepended to normal header
  // 4-byte size precedes the normal header
  auto split_header_size = non_empty_header_size(t) + 4;
  std::size_t sum = 0;
  for (int i = 0; i < split_indices.size(); i++) {
    auto split_start = static_cast<cudf::size_type>(split_indices[i]);
    auto split_end = (i < split_indices.size() - 1) ? split_indices[i + 1] : t.num_rows();
    auto num_rows = split_end - split_start;
    if (num_rows == 0) {
      sum += EMPTY_HEADER_SIZE;
    } else {
      sum += std::accumulate(t.begin(), t.end(), 0,
        [num_rows, split_header_size, split_start](std::size_t s, host_column_view const& c) {
          auto ss = split_size(c, split_start, num_rows) + split_header_size;
          return s + ss;
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
      uint64_t bits = *imp++ >> shift;
      for (std::size_t i = 0; i < num_input_mask_words - 1; i++) {
        bits |= *imp++ << merge_shift;
        *omp++ = static_cast<cudf::bitmask_type>(bits);
        bits >>= 32;
      }
      *omp++ = static_cast<cudf::bitmask_type>(bits);
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

uint8_t* single_split_on_host(host_table_view const& t, cudf::size_type split_start,
                              cudf::size_type num_rows, uint8_t* out, uint8_t* out_end)
{
  auto op = out;
  std::for_each(t.begin(), t.end(), [=, &op](host_column_view const& c) {
    op = single_split_on_host(c, split_start, num_rows, op, out_end);
  });
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
  auto const has_nulls_mask_bytesize = has_nulls_mask.size() * sizeof(uint32_t);
  for (int i = 0; i < num_splits; i++) {
    offsets.push_back(op - out);
    cudf::size_type split_start = static_cast<cudf::size_type>(split_indices[i]);
    cudf::size_type split_end = (i == num_splits - 1)
      ? t.num_rows() : static_cast<cudf::size_type>(split_indices[i + 1]);
    auto const num_rows = split_end - split_start;
    // start writing header bytes, total size filled in afterwards
    size_check(op, op_end, EMPTY_HEADER_SIZE);
    auto header = reinterpret_cast<uint32_t*>(op);
    auto op_start = op;
    op += EMPTY_HEADER_SIZE;
    header[1] = num_rows;
    if (num_rows > 0) {
      size_check(op, op_end, has_nulls_mask_bytesize);
      std::memcpy(op, has_nulls_mask.data(), has_nulls_mask_bytesize);
      op += has_nulls_mask_bytesize;
      op = single_split_on_host(t, split_start, num_rows, op, op_end);
    }
    header[0] = op - op_start - 4;  // do not count 4 bytes in header for total size
  }
  if (op != op_end) {
    throw std::logic_error("output buffer not fully used");
  }
  return offsets;
}

struct column_concat_meta {
  std::size_t data_size;
  int num_rows;
  int num_children;
  cudf::data_type dtype;
  bool has_nulls;

  column_concat_meta(cudf::data_type dt, int num_child)
    : data_size(0), num_rows(0), num_children(num_child), dtype(dt), has_nulls(false) {}
};

std::pair<int, uint8_t const*>
update_column_concat_meta(std::vector<column_concat_meta>& meta, int num_rows,
                          uint32_t const* has_nulls_mask, int column_index, uint8_t const* buffer)
{
  column_concat_meta& column_meta = meta[column_index];
  column_meta.num_rows += num_rows;
  bool has_validity = has_nulls_mask != nullptr &&
    has_nulls_mask[column_index / 32] & (1 << column_index);
  if (has_nulls_mask) {
    std::cerr << "*has_nulls_mask: " << std::hex << *has_nulls_mask << std::endl;
  }
  std::cerr << "column " << column_index << " has validity: " << has_validity << std::endl;
  if (has_validity) {
    column_meta.has_nulls = true;
    buffer += pad_size((num_rows + 7)/8);
  }
  column_index += 1;
  if (num_rows > 0) {
    switch (column_meta.dtype.id()) {
      case cudf::type_id::STRING:
      case cudf::type_id::LIST:
      {
        auto offsets = reinterpret_cast<uint32_t const*>(buffer);
        buffer += pad_size((num_rows + 1) * sizeof(uint32_t));
        auto child_rows = offsets[num_rows] - offsets[0];
        if (column_meta.dtype.id() == cudf::type_id::STRING) {
          column_meta.data_size += child_rows;
        } else {
          return update_column_concat_meta(meta, child_rows, has_nulls_mask, column_index, buffer);
        }
        break;
      }
      case cudf::type_id::STRUCT:
      {
        for (int child_index = 0; child_index < column_meta.num_children; child_index++) {
          auto [next_column_index, next_buffer] = update_column_concat_meta(meta, num_rows,
            has_nulls_mask, column_index, buffer);
          column_index = next_column_index;
          buffer = next_buffer;
        }
        break;
      }
      default:
        column_meta.data_size += cudf::size_of(column_meta.dtype) * num_rows;
        break;
    }
  }
  return std::make_pair(column_index, buffer);
}

std::vector<column_concat_meta> build_column_concat_meta(cudf::jni::native_jintArray const& meta, uint8_t const* buffer,
                                                         std::size_t buffer_size,
                                                         cudf::jni::native_jlongArray const& offsets)
{
  if (meta.size() % 2 != 0) {
    throw std::logic_error("metadata size is odd");
  }
  int total_columns = meta.size() / 2;
  std::vector<column_concat_meta> concat_meta;
  concat_meta.reserve(total_columns);
  for (int i = 0; i < total_columns; i++) {
    auto type_id = static_cast<cudf::type_id>(meta[i * 2]);
    cudf::data_type dtype(type_id);
    auto num_children = 0;
    switch (type_id) {
      case cudf::type_id::STRUCT:
        num_children = meta[(i * 2) + 1];
        break;
      case cudf::type_id::LIST:
        num_children = 1;
        break;
      case cudf::type_id::DECIMAL32: [[fallthrough]];
      case cudf::type_id::DECIMAL64: [[fallthrough]];
      case cudf::type_id::DECIMAL128:
        dtype = cudf::data_type(type_id, meta[(i * 2) + 1]);
        num_children = 0;
        break;
      default:
        num_children = 0;
        break;
    }
    concat_meta.emplace_back(dtype, num_children);
  }
  std::for_each(offsets.begin(), offsets.end(), [=, &meta, &concat_meta](jlong offset) {
    auto part_buffer = buffer + offset;
    auto word_ptr = reinterpret_cast<uint32_t const*>(part_buffer);
    auto num_rows = *word_ptr++;
    auto has_nulls_mask = num_rows == 0 ? nullptr : word_ptr;
    std::cerr << "offset " << offset << " row count: " << num_rows << std::endl;
    // move past header
    part_buffer += pad_size(4 + (total_columns + 7)/8);
    int column_index = 0;
    while (column_index != total_columns) {
      auto [next_column_index, next_buffer] =
        update_column_concat_meta(concat_meta, num_rows, has_nulls_mask, column_index, part_buffer);
      column_index = next_column_index;
      part_buffer = next_buffer;
    }
  });
  return concat_meta;
}

std::size_t concat_to_host_table_size(std::vector<column_concat_meta> const& concat_meta)
{
  return std::accumulate(concat_meta.cbegin(), concat_meta.cend(), 0,
    [](std::size_t sum, column_concat_meta const& c) {
      sum += pad_size(c.data_size);
      if (c.has_nulls) {
        sum += pad_size((c.num_rows + 7)/8);
      }
      if (c.num_rows > 0 && (c.dtype.id() == cudf::type_id::STRING || c.dtype.id() == cudf::type_id::LIST)) {
        sum += pad_size((c.num_rows + 1) * 4);
      }
      return sum;
    });
}

struct column_concat_tracker {
  cudf::data_type dtype;
  int num_rows;
  int num_children;
  uint8_t* next_data;
  cudf::bitmask_type* validity_start;
  cudf::size_type* offsets_start;
  uint8_t* data_start;

  column_concat_tracker(cudf::data_type type, int num_child, cudf::bitmask_type* null_mask, cudf::size_type* offsets, uint8_t* data)
    : dtype(type), num_rows(0), num_children(num_child), next_data(data),
      validity_start(null_mask), offsets_start(offsets), data_start(data) {}
};

std::pair<column_concat_tracker, uint8_t*>
to_column_concat_tracker(column_concat_meta const& m, uint8_t* bp, uint8_t* bp_end)
{
  cudf::bitmask_type* null_mask = nullptr;
  cudf::size_type* offsets = nullptr;
  uint8_t* data = nullptr;
  if (m.num_rows > 0) {
    if (m.has_nulls) {
      null_mask = reinterpret_cast<cudf::bitmask_type*>(bp);
      bp += cudf::bitmask_allocation_size_bytes(m.num_rows);
    }
    switch (m.dtype.id()) {
      case cudf::type_id::STRING: [[falthrough]];
      case cudf::type_id::LIST:
        offsets = reinterpret_cast<cudf::size_type*>(bp);
        bp += (m.num_rows + 1) * sizeof(cudf::size_type);
        break;
      default:
        break;
    }
    if (m.data_size > 0) {
      data = bp;
      bp += m.data_size;
    }
  }
  if (bp > bp_end) {
    throw std::logic_error("buffer overrun");
  }
  std::cerr << "column null mask: " << null_mask << std::endl;
  return std::make_pair(column_concat_tracker(m.dtype, m.num_children, null_mask, offsets, data), bp);
}

std::vector<column_concat_tracker>
to_column_concat_trackers(std::vector<column_concat_meta> const& metas, uint8_t* dest_buffer,
                          std::size_t dest_buffer_size)
{
  auto bp = dest_buffer;
  auto bp_end = dest_buffer + dest_buffer_size;
  std::vector<column_concat_tracker> trackers;
  trackers.reserve(metas.size());
  std::transform(metas.cbegin(), metas.cend(), std::back_inserter(trackers),
    [&](column_concat_meta const& m) {
      auto [tracker, next_bp] = to_column_concat_tracker(m, bp, bp_end);
      bp = next_bp;
      return tracker;
    });
  return trackers;
}

void copy_validity_part(cudf::bitmask_type* dest, cudf::bitmask_type const* src, int dest_start_bit,
                   int num_bits)
{
  if (dest_start_bit > 0) {
    if (num_bits < NULL_MASK_WORD_BITS - dest_start_bit) {
      auto mask = (1 << num_bits) - 1;
      auto shifted_mask = mask << dest_start_bit;
      *dest = (*dest & ~shifted_mask) | ((*src & mask) << dest_start_bit);
    } else {
      // TODO: consider adding SSE/AVX/NEON versions of this
      static_assert(NULL_MASK_WORD_BITS == 32, "bitmask shifted copy needs update");
      uint64_t bits = static_cast<uint64_t>(*src++) << dest_start_bit;
      *dest = (*dest & ~((1 << dest_start_bit) - 1)) | static_cast<cudf::bitmask_type>(bits);
      dest += 1;
      num_bits -= NULL_MASK_WORD_BITS - dest_start_bit;
      while (num_bits >= NULL_MASK_WORD_BITS) {
        bits >>= 32;
        bits |= static_cast<uint64_t>(*src++) << dest_start_bit;
        *dest++ = static_cast<cudf::bitmask_type>(bits);
      }
      if (num_bits > 0) {
        *dest++ = static_cast<cudf::bitmask_type>(bits >> 32);
      }
    }
  } else {
    auto num_whole_words = num_bits / NULL_MASK_WORD_BITS;
    std::memcpy(dest, src, num_whole_words * sizeof(cudf::bitmask_type));
    src += num_whole_words;
    dest += num_whole_words;
    num_bits -= num_whole_words * NULL_MASK_WORD_BITS;
    if (num_bits > 0) {
      *dest = *src & ((1 << num_bits) - 1);
    }
  }
}

void fill_validity_part(cudf::bitmask_type* dest, int dest_start_bit, int num_bits)
{
  // fill in any partial starting word
  if (dest_start_bit > 0) {
    if (num_bits < NULL_MASK_WORD_BITS - dest_start_bit) {
      *dest |= ((1 << num_bits) - 1) << dest_start_bit;
      return;
    } else {
      *dest++ |= (~0) << dest_start_bit;
      num_bits -= NULL_MASK_WORD_BITS - dest_start_bit;
    }
  }
  auto num_whole_words = num_bits / NULL_MASK_WORD_BITS;
  if (num_whole_words > 0) {
    std::memset(dest, 0xFF, num_whole_words * sizeof(cudf::bitmask_type));
    dest += num_whole_words;
    num_bits -= num_whole_words * NULL_MASK_WORD_BITS;
  }
  if (num_bits > 0) {
    *dest = (1 << num_bits) - 1;
  }
}

std::pair<int, uint8_t const*>
copy_column_part(std::vector<column_concat_tracker>& trackers, uint32_t num_rows,
                 uint32_t const* has_nulls_mask, std::size_t column_index, uint8_t const* bp)
{
  column_concat_tracker& tracker = trackers[column_index];
  cudf::size_type child_rows = 0;
  if (num_rows > 0) {
    if (tracker.validity_start != nullptr) {
      auto dest_validity = tracker.validity_start + (tracker.num_rows / NULL_MASK_WORD_BITS);
      auto dest_start_bit = tracker.num_rows % NULL_MASK_WORD_BITS;
      if (has_nulls_mask[column_index / 32] & (1 << column_index)) {
        auto src_validity = reinterpret_cast<uint32_t const*>(bp);
        bp += pad_size((num_rows + 7) / 8);
        copy_validity_part(dest_validity, src_validity, dest_start_bit, num_rows);
      } else {
        fill_validity_part(dest_validity, dest_start_bit, num_rows);
      }
    }
    if (tracker.offsets_start != nullptr) {
      auto src_offsets = reinterpret_cast<cudf::size_type const*>(bp);
      bp += pad_size((num_rows + 1) * sizeof(cudf::size_type));
      auto dest_offsets = tracker.offsets_start;
      int num_offsets = num_rows + 1;
      // src offsets may not be 0-based
      cudf::size_type offset_adjust = -src_offsets[0];
      child_rows = src_offsets[num_rows] - src_offsets[0];
      if (tracker.num_rows > 0) {
        // adding onto existing offsets, so skip the first source offset
        src_offsets += 1;
        dest_offsets += tracker.num_rows + 1;
        num_offsets = num_rows;
        offset_adjust += tracker.offsets_start[tracker.num_rows];
      }
      for (int i = 0; i < num_offsets; i++) {
        dest_offsets[i] = src_offsets[i] + offset_adjust;
      }
    }
    if (tracker.next_data != nullptr) {
      std::size_t data_size = 0;
      if (cudf::is_fixed_width(tracker.dtype)) {
        data_size = cudf::size_of(tracker.dtype) * num_rows;
      } else if (tracker.dtype.id() == cudf::type_id::STRING) {
        data_size = child_rows;
      } else {
        type_error(tracker.dtype.id());
      }
      std::memcpy(tracker.next_data, bp, data_size);
      tracker.next_data += data_size;
      bp += pad_size(data_size);
    }
  }
  tracker.num_rows += num_rows;
  column_index += 1;
  for (int child_index = 0; child_index < tracker.num_children; child_index++) {
    auto [next_column_index, next_bp] =
      copy_column_part(trackers, child_rows, has_nulls_mask, column_index, bp);
    column_index = next_column_index;
    bp = next_bp;
  }
  return std::make_pair(column_index, bp);
}

inline unsigned int popc32(uint32_t x)
{
  // TODO: Use popcnt instruction when available
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0F0F0F0F;
  return (x * 0x01010101) >> 24;
}

inline unsigned int count_zeros(cudf::bitmask_type bits)
{
  auto flipped = ~bits;
  if (flipped == 0) {
    return 0;
  }
  return popc32(flipped);
}

cudf::size_type count_nulls(cudf::bitmask_type const* validity, int num_rows)
{
  cudf::size_type null_count = 0;
  int num_whole_words = num_rows / NULL_MASK_WORD_BITS;
  for (int i = 0; i < num_whole_words; i++) {
    std::cerr << "count_zeroes full word: " << std::hex << validity[i];
    null_count += count_zeros(validity[i]);
    std::cerr << " null count: " << null_count << std::endl;
  }
  auto remaining_rows = num_rows % NULL_MASK_WORD_BITS;
  if (remaining_rows != 0) {
    std::cerr << "count_zeroes partial word: " << std::hex << validity[num_whole_words];
    cudf::bitmask_type bits = validity[num_whole_words];
    // set all the bits that don't correspond to valid rows to 1
    bits |= ~((1 << remaining_rows) - 1);
    std::cerr << " count_zeroes bits: " << std::hex << bits;
    null_count += count_zeros(bits);
    std::cerr << " null count: " << null_count << std::endl;
  }
  return null_count;
}

std::size_t convert_to_view(std::vector<column_concat_tracker> const& trackers, std::size_t column_index,
                            std::vector<host_column_view>& views)
{
  column_concat_tracker const& tracker = trackers[column_index];
  column_index += 1;
  auto null_count = 0;
  if (tracker.validity_start != nullptr) {
    null_count = count_nulls(tracker.validity_start, tracker.num_rows);
  }
  std::vector<host_column_view> child_views;
  if (tracker.offsets_start != nullptr) {
    child_views.push_back(host_column_view(cudf::data_type(cudf::type_id::INT32),
      tracker.num_rows + 1, tracker.offsets_start, nullptr, 0));
  } else {
    for (int child_index = 0; child_index < tracker.num_children; child_index++) {
      column_index = convert_to_view(trackers, column_index, child_views);
    }
  }
  views.push_back(host_column_view(tracker.dtype, tracker.num_rows, tracker.data_start,
    tracker.validity_start, null_count, child_views));
  return column_index;
}

std::vector<host_column_view> to_host_column_views(std::vector<column_concat_tracker> const& trackers)
{
  std::vector<host_column_view> views;
  views.reserve(trackers.size());
  std::size_t column_index = 0;
  while (column_index != trackers.size()) {
    column_index = convert_to_view(trackers, column_index, views);
  }
  return views;
}

std::unique_ptr<host_table_view> concat_to_host_table(std::vector<column_concat_meta> const& concat_meta,
                                                      uint8_t const* buffer, std::size_t buffer_size,
                                                      cudf::jni::native_jlongArray const& offsets,
                                                      uint8_t* dest_buffer, std::size_t dest_buffer_size)
{
  auto total_columns = concat_meta.size();
  auto column_trackers = to_column_concat_trackers(concat_meta, dest_buffer, dest_buffer_size);
  std::for_each(offsets.begin(), offsets.end(), [=, &column_trackers](jlong offset) {
    auto part_buffer = buffer + offset;
    auto word_ptr = reinterpret_cast<uint32_t const*>(part_buffer);
    auto num_rows = *word_ptr++;
    auto has_nulls_mask = word_ptr;
    // move past header
    part_buffer += pad_size(4 + (total_columns + 7)/8);
    std::size_t column_index = 0;
    while (column_index != total_columns) {
      auto [next_column_index, next_buffer] =
        copy_column_part(column_trackers, num_rows, has_nulls_mask, column_index, part_buffer);
      column_index = next_column_index;
      part_buffer = next_buffer;
    }
  });
  auto column_views = to_host_column_views(column_trackers);
  return std::make_unique<host_table_view>(column_views);
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

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_buildConcatToHostTableMeta(
  JNIEnv* env, jclass, jintArray jmeta, jlong jbuffer_addr, jlong jbuffer_size, jlongArray joffsets)
{
  JNI_NULL_CHECK(env, jmeta, "meta is null", 0);
  JNI_NULL_CHECK(env, jbuffer_addr, "buffer is null", 0);
  JNI_NULL_CHECK(env, joffsets, "offsets is null", 0);
  try {
    cudf::jni::native_jintArray meta(env, jmeta);
    cudf::jni::native_jlongArray offsets(env, joffsets);
    auto buffer = reinterpret_cast<uint8_t const*>(jbuffer_addr);
    auto buffer_size = static_cast<std::size_t>(jbuffer_size);
    auto concatMeta = std::make_unique<std::vector<column_concat_meta>>(build_column_concat_meta(meta, buffer, buffer_size, offsets));
    return cudf::jni::release_as_jlong(concatMeta);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_freeConcatToHostTableMeta(
  JNIEnv* env, jclass, jlong jmeta)
{
  JNI_NULL_CHECK(env, jmeta, "meta is null", );
  try {
    auto meta = reinterpret_cast<std::vector<column_concat_meta>*>(jmeta);
    delete meta;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_concatToHostTableSize(
  JNIEnv* env, jclass, jlong jmeta)
{
  JNI_NULL_CHECK(env, jmeta, "meta is null", 0);
  try {
    auto concat_meta = reinterpret_cast<std::vector<column_concat_meta> const*>(jmeta);
    return static_cast<jlong>(concat_to_host_table_size(*concat_meta));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_concatToHostTable(
  JNIEnv* env, jclass, jintArray jmeta, jlong jbuffer_addr, jlong jbuffer_size, jlongArray joffsets,
  jlong jdest_buffer_addr, jlong jdest_buffer_size)
{
  JNI_NULL_CHECK(env, jmeta, "meta is null", 0);
  JNI_NULL_CHECK(env, jbuffer_addr || jbuffer_addr == 0, "buffer is null", 0);
  JNI_NULL_CHECK(env, joffsets, "offsets is null", 0);
  JNI_NULL_CHECK(env, jdest_buffer_size || jdest_buffer_size == 0, "dest is null", 0);
  try {
    auto concat_meta = reinterpret_cast<std::vector<column_concat_meta> const*>(jmeta);
    cudf::jni::native_jlongArray offsets(env, joffsets);
    auto buffer = reinterpret_cast<uint8_t const*>(jbuffer_addr);
    auto buffer_size = static_cast<std::size_t>(jbuffer_size);
    auto dest_buffer = reinterpret_cast<uint8_t*>(jdest_buffer_addr);
    auto dest_buffer_size = static_cast<std::size_t>(jdest_buffer_size);
    return cudf::jni::release_as_jlong(
      concat_to_host_table(*concat_meta, buffer, buffer_size, offsets, dest_buffer, dest_buffer_size));
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
