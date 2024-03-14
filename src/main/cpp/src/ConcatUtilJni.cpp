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
  for (jlong const* p = host_ranges.begin(); p != host_ranges.end();) {
    auto host_addr = reinterpret_cast<void const*>(*p++);
    auto size = static_cast<std::size_t>(*p++);
    gpu_ranges.emplace_back(std::make_unique(host_addr, size, cudf::get_default_stream()));
  }
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

cudf::column_view create_column_view(uint8_t const* header_ptr,
                                     uint32_t num_rows,
                                     uint8_t const*& buffer_pos,
                                     uint8_t const* buffer_end) {
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
  void const* data = nullptr;
  cudf::bitmask_type const* null_mask = nullptr;
  std::vector<cudf::column_view> children;

  cudf::data_type = cudf::jni::make_data_type()
  return cudf::column_view(dtype, num_rows, data, null_mask, 0, children);
}

cudf::table_view create_table_view(uint8_t const* header_ptr,
                                   uint8_t const*& buffer_pos,
                                   uint8_t const* buffer_end) {
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
  auto const data_len = read_long(header_ptr);
  std::vector<cudf::column_view> column_views;
  column_views.reserve(num_columns);
  for (uint32_t i = 0; i < num_columns; ++i) {
    column_views.emplace_back(create_column_view(header_ptr, num_rows, buffer_pos, buffer_end));
  }
  return cudf::table_view(column_views);
}

std::vector<cudf::table_view> create_table_views(
    std::vector<jlong> header_addrs, std::vector<std::unique_ptr<rmm::device_buffer>> buffers) {
  std::vector<cudf::table_view> views;
  views.reserve(header_addrs.size());
  std::size_t next_buffer_index = 0;
  uint8_t const* buffer_pos = nullptr;
  uint8_t const* buffer_end = buffer_pos;
  for (jlong const* header_addr_ptr = header_addrs.begin();
       header_addr_ptr != header_addrs.end();
       ++header_addr_ptr) {
    if (buffer_pos == buffer_end) {
      buffer_pos = static_cast<uint8_t const*>(buffers[next_buffer_index]->data());
      buffer_end = buffer_pos + buffers[next_buffer_index]->size();
      ++next_buffer_index;
    }
    auto header_data_ptr = reinterpret_cast<uint8_t const*>(*header_addr_ptr);
    views.emplace_back(create_table_view(header_data_ptr, buffer_pos, buffer_end));
  }
  if (buffer_pos != buffer_end || next_buffer_index != buffers.size()) {
    throw std::runtime_error("error deserializing table views");
  }
  return views;
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
    data_ranges.cancel();
    auto tables = create_table_views(headers_vec, gpu_buffers);
    header_addrs.cancel();
    auto result = cudf::concatenate(tables);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

}
