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
#include "dtype_utils.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/device_buffer.hpp>

#include <vector>

namespace {

std::vector<std::unique_ptr<rmm::device_buffer>> copy_ranges(std::vector<jlong> const& host_ranges) {
  std::vector<std::unique_ptr<rmm::device_buffer>> gpu_ranges;
  gpu_ranges.reserve(host_ranges.size() / 2);
  for (auto p = host_ranges.begin(); p != host_ranges.end();) {
    auto host_addr = reinterpret_cast<void const*>(*p++);
    auto size = static_cast<std::size_t>(*p++);
    gpu_ranges.emplace_back(
        std::make_unique<rmm::device_buffer>(host_addr, size, cudf::get_default_stream()));
  }
  return gpu_ranges;
}

int32_t read_int(uint8_t const*& p) {
  int32_t v = static_cast<int32_t>(*p++) << 24;
  v |= static_cast<int32_t>(*p++) << 16;
  v |= static_cast<int32_t>(*p++) << 8;
  v |= static_cast<int32_t>(*p++);
  return v;
}

int64_t read_long(uint8_t const*& p) {
  int64_t v = static_cast<int64_t>(*p++) << 56;
  v |= static_cast<int64_t>(*p++) << 48;
  v |= static_cast<int64_t>(*p++) << 40;
  v |= static_cast<int64_t>(*p++) << 32;
  v |= static_cast<int64_t>(*p++) << 24;
  v |= static_cast<int64_t>(*p++) << 16;
  v |= static_cast<int64_t>(*p++) << 8;
  v |= static_cast<int64_t>(*p++);
  return v;
}

uint64_t align64(uint64_t offset) {
  return ((offset + 63) / 64) * 64;
}

uint32_t get_validity_byte_size(uint32_t num_rows) {
  uint32_t padded_rows = ((num_rows + 63) / 64) * 64;
  return align64(padded_rows / 8);
}

cudf::column_view create_column_view(uint8_t const*& header_ptr,
                                     uint32_t num_rows,
                                     uint8_t const* host_buffer,
                                     uint8_t const* gpu_buffer,
                                     int64_t& buffer_offset) {
  // column header:
  // - 4-byte type ID
  // - 4-byte type scale
  // - 4-byte null count
  // if list:
  //   - 4-byte row count of child
  // if struct:
  //   - 4-byte child count
  auto const dtype_id = read_int(header_ptr);
  auto const scale = read_int(header_ptr);
  auto const null_count = read_int(header_ptr);
  cudf::data_type dtype = cudf::jni::make_data_type(dtype_id, scale);
  cudf::bitmask_type const* null_mask = nullptr;
  if (null_count != 0) {
    null_mask = reinterpret_cast<cudf::bitmask_type const*>(gpu_buffer + buffer_offset);
    buffer_offset += get_validity_byte_size(num_rows);
  }
  std::vector<cudf::column_view> children;
  void const* data = nullptr;
  if (dtype.id() == cudf::type_id::STRING || dtype.id() == cudf::type_id::LIST) {
    cudf::size_type offsets_count = (num_rows > 0) ? num_rows + 1 : 0;
    if (offsets_count > 0 || dtype.id() == cudf::type_id::LIST) {
      children.emplace_back(
        cudf::data_type{cudf::type_to_id<cudf::size_type>()},
        offsets_count,
        offsets_count > 0 ? gpu_buffer + buffer_offset : nullptr,
        nullptr,
        0);
      auto const offsets_len = offsets_count * sizeof(cudf::size_type);
      auto host_offsets = reinterpret_cast<cudf::size_type const*>(host_buffer + buffer_offset);
      buffer_offset += align64(offsets_len);
      if (offsets_count > 0 && dtype.id() == cudf::type_id::STRING) {
        auto start_offset = host_offsets[0];
        auto end_offset = host_offsets[num_rows];
        data = gpu_buffer + buffer_offset;
        buffer_offset += align64(end_offset - start_offset);
      }
    }
    if (dtype.id() == cudf::type_id::LIST) {
      auto child_num_rows = read_int(header_ptr);
      children.emplace_back(create_column_view(header_ptr, child_num_rows, host_buffer, gpu_buffer,
          buffer_offset));
    }
  } else if (dtype.id() == cudf::type_id::STRUCT) {
    auto num_children = read_int(header_ptr);
    for (int child_idx = 0; child_idx < num_children; ++child_idx) {
      children.emplace_back(create_column_view(header_ptr, num_rows, host_buffer, gpu_buffer,
          buffer_offset));
    }
  } else {
    data = gpu_buffer + buffer_offset;
    auto data_len = cudf::size_of(dtype) * num_rows;
    buffer_offset += align64(data_len);
  }
  return cudf::column_view(dtype, num_rows, data, null_mask, null_count, 0, children);
}

cudf::table_view create_table_view(uint8_t const*& header_ptr,
                                   uint8_t const* host_buffer,
                                   uint8_t const* gpu_buffer,
                                   int64_t& buffer_offset) {
  // table header:
  // - 4-byte magic number
  // - 2-byte version number
  // - 4-byte column count
  // - 4-byte row count
  // column headers
  // - 8-byte data buffer length
  auto const magic = read_int(header_ptr);
  if (magic != 0x43554446) {
    throw std::runtime_error("Bad table header magic");
  }
  if (*header_ptr++ != 0x00 || *header_ptr++ != 0x00) {
    throw std::runtime_error("Bad table header version");
  }
  auto const num_columns = read_int(header_ptr);
  auto const num_rows = read_int(header_ptr);
  auto const buffer_offset_start = buffer_offset;
  std::vector<cudf::column_view> column_views;
  column_views.reserve(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    column_views.emplace_back(create_column_view(header_ptr, num_rows, host_buffer, gpu_buffer,
      buffer_offset));
  }
  auto const data_len = read_long(header_ptr);
  if (buffer_offset - buffer_offset_start != data_len) {
    throw std::runtime_error("table deserialization error");
  }
  return cudf::table_view(column_views);
}

std::vector<cudf::table_view> create_table_views(
    std::vector<jlong> const& header_addrs,
    std::vector<jlong> const& host_ranges,
    std::vector<std::unique_ptr<rmm::device_buffer>> const& gpu_buffers) {
  std::vector<cudf::table_view> views;
  views.reserve(header_addrs.size());
  std::size_t next_buffer_index = 0;
  uint8_t const* host_buffer = nullptr;
  uint8_t const* gpu_buffer = nullptr;
  int64_t buffer_offset = 0;
  int64_t buffer_size = 0;
  for (auto header_addr_ptr = header_addrs.begin();
       header_addr_ptr != header_addrs.end();
       ++header_addr_ptr) {
    if (buffer_offset == buffer_size) {
      buffer_offset = 0;
      host_buffer = reinterpret_cast<uint8_t const*>(host_ranges[next_buffer_index * 2]);
      gpu_buffer = static_cast<uint8_t const*>(gpu_buffers[next_buffer_index]->data());
      buffer_size = gpu_buffers[next_buffer_index]->size();
      ++next_buffer_index;
    }
    auto header_data_ptr = reinterpret_cast<uint8_t const*>(*header_addr_ptr);
    views.emplace_back(create_table_view(header_data_ptr, host_buffer, gpu_buffer, buffer_offset));
    if (buffer_offset > buffer_size) {
      throw std::runtime_error("buffer overflow during table deserialization");
    }
  }
  if (buffer_offset != buffer_size || next_buffer_index != gpu_buffers.size()) {
    throw std::runtime_error("error deserializing table views");
  }
  return views;
}

}

extern "C" {

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_ConcatUtil_concatSerialized(
    JNIEnv *env, jclass, jlongArray j_header_addrs, jlongArray j_data_ranges) {
  JNI_NULL_CHECK(env, j_header_addrs, "null headers", NULL);
  JNI_NULL_CHECK(env, j_data_ranges, "null ranges", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray header_addrs{env, j_header_addrs};
    cudf::jni::native_jlongArray data_ranges{env, j_data_ranges};
    auto headers_vec = header_addrs.to_vector();
    auto ranges_vec = data_ranges.to_vector();
    auto gpu_buffers = copy_ranges(ranges_vec);
    auto tables = create_table_views(headers_vec, ranges_vec, gpu_buffers);
    data_ranges.cancel();
    header_addrs.cancel();
    auto result = cudf::concatenate(tables);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

}
