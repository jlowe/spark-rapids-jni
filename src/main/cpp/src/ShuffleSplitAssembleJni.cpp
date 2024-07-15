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

#include <cudf/utilities/span.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

//====================
//temporary declarations until headers are available
#include <cudf/types.hpp>
#include <vector>
namespace spark_rapids_jni {
struct cz_metadata {
  // depth-first traversal
  // - a fixed column or strings have 0
  // - a list column always has 1 (I guess we could also make this
  //   0 to make it more string-like. in my brain I always think of lists
  //   as explicitly having a child because the type is completely arbitrary)
  // - structs have N
  //
  std::vector<cudf::size_type>      num_children{};
  std::vector<cudf::type_id>  types{};
};
struct cz_result {
  // packed partition buffers. very similar to what would have come out of
  // contiguous_split, except:
  // - it is one big buffer where all of the partitions are glued together instead
  //   of one buffer per-partition
  // - each partition is prepended by some data-dependent info: row count, presence-of-validity
  // etc
  std::unique_ptr<rmm::device_buffer>    partitions;
  // offsets into the partition buffer for each partition. offsets.size() will be
  // num partitions + 1
  std::unique_ptr<rmm::device_uvector<size_t>>   offsets;
};
cz_result shuffle_split(cudf::table_view const& input,
                        cz_metadata const& global_metadata,
                        std::vector<cudf::size_type> const& splits,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::table> shuffle_assemble(cz_metadata const& global_metadata,
                                        cudf::device_span<int8_t const> partitions,
                                        cudf::device_span<size_t const> partition_offsets);
}
//====================


extern "C" {

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_split(
  JNIEnv* env, jclass, jintArray jmeta_num_children, jintArray jmeta_types, jlong jtable,
  jintArray jsplit_indices)
{
  JNI_NULL_CHECK(env, jmeta_num_children, "metadata num children is null", nullptr);
  JNI_NULL_CHECK(env, jmeta_types, "metadata types is null", nullptr);
  JNI_NULL_CHECK(env, jtable, "input table is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto const table = reinterpret_cast<cudf::table_view const*>(jtable);
    cudf::jni::native_jintArray meta_num_children(env, jmeta_num_children);
    cudf::jni::native_jintArray meta_types(env, jmeta_types);
    std::vector<cudf::type_id> cudf_types;
    cudf_types.reserve(meta_types.size());
    std::transform(meta_types.begin(), meta_types.end(), std::back_inserter(cudf_types),
                   [](auto id) { return static_cast<cudf::type_id>(id); });
    cudf::jni::native_jintArray indices(env, jsplit_indices);
    spark_rapids_jni::cz_metadata meta{meta_num_children.to_vector(), std::move(cudf_types)};
    auto split_result = spark_rapids_jni::shuffle_split(*table, meta, indices.to_vector(),
                                                        cudf::get_default_stream(),
                                                        rmm::mr::get_current_device_resource());
    meta_num_children.cancel();
    meta_types.cancel();
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

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_ShuffleSplitAssemble_assemble(
  JNIEnv* env, jclass, jintArray jmeta_num_children, jintArray jmeta_types,
  jlong parts_addr, jlong parts_size, jlong offsets_addr, jlong offsets_count)
{
  JNI_NULL_CHECK(env, jmeta_num_children, "metadata num children is null", nullptr);
  JNI_NULL_CHECK(env, jmeta_types, "metadata types is null", nullptr);
  JNI_NULL_CHECK(env, parts_addr, "partitions buffer is null", nullptr);
  JNI_NULL_CHECK(env, offsets_addr, "offsets buffer is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto const parts_ptr = reinterpret_cast<int8_t const*>(parts_addr);
    cudf::device_span<int8_t const> parts_span{parts_ptr, static_cast<size_t>(parts_size)};
    auto const offsets_ptr = reinterpret_cast<size_t const*>(offsets_addr);
    cudf::device_span<size_t const> offsets_span{offsets_ptr, static_cast<size_t>(offsets_count)};
    cudf::jni::native_jintArray meta_num_children(env, jmeta_num_children);
    cudf::jni::native_jintArray meta_types(env, jmeta_types);
    std::vector<cudf::type_id> cudf_types;
    cudf_types.reserve(meta_types.size());
    std::transform(meta_types.begin(), meta_types.end(), std::back_inserter(cudf_types),
                   [](auto id) { return static_cast<cudf::type_id>(id); });
    spark_rapids_jni::cz_metadata meta{meta_num_children.to_vector(), cudf_types};
    auto table = spark_rapids_jni::shuffle_assemble(meta, parts_span, offsets_span);
    return cudf::jni::convert_table_for_return(env, table);
  }
  CATCH_STD(env, nullptr);
}

}  // extern "C"
