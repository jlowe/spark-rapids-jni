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

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <vector>

namespace cudf::spark_rapids_jni {

static constexpr std::size_t shuffle_split_partition_data_align = 16;

struct shuffle_split_col_data {
  cudf::size_type num_children;
  cudf::type_id type;
};

// generated by Spark, independent of any cudf data
struct shuffle_split_metadata {
  // depth-first traversal of the input table, by children.
  std::vector<shuffle_split_col_data> col_info;
};

struct shuffle_split_result {
  // packed partition buffers.
  // - it is one big buffer where all of the partitions are glued together instead
  //   of one buffer per-partition
  // - each partition is prepended with a metadata buffer
  //   the metadata is of the format:
  //   - for each entry in shuffle_split_metadata.col_info
  //     - 2x4-byte row count per string column
  //     - 1x4-byte row count for all other types
  //   - for each entry in shuffle_split_metadata.col_info
  //     - 1 bit per column indicating whether or not validity info is
  //       included, rounded up the nearest bitmask_type number of elements
  //   - pad to partition_data_align bytes
  //   - the contiguous-split style buffer of column data (which is also padded to partition_data_align bytes)
  std::unique_ptr<rmm::device_buffer>   partitions{};

  // offsets into the partition buffer for each partition. offsets.size() will be
  // num partitions
  rmm::device_uvector<size_t>           offsets{0, cudf::get_default_stream()};
};


/**
 * @brief Performs a split operation on a cudf table, returning a buffer of data containing
 * all of the sub-tables as a contiguous buffer of anonymous bytes.
 * 
 * Performs a split on the input table similar to the cudf::split function.
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 * 
 * The result is returned as a blob of bytes representing the individual partitions resulting from
 * the splits, and a set of offsets indicating the beginning of each resulting partition in the result.
 * The function also returns a shuffle_split_metadat struct which contains additional information needed
 * to reconstruct the buffer during shuffle_assemble.
 *
 * @param input The input table
 * @param splits The set of splits to split the table with
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return A shuffle_split_result struct containing the resulting buffer and offsets to each partition, and
 * a shuffle_split_metadata struct which contains the metadata needed to reconstruct a table using shuffle_assemble.
 */
std::pair<shuffle_split_result, shuffle_split_metadata> shuffle_split(cudf::table_view const& input,
                                                                      std::vector<cudf::size_type> const& splits,
                                                                      rmm::cuda_stream_view stream,
                                                                      rmm::device_async_resource_ref mr);

/**
 * @brief Reassembles a set of partitions generated by shuffle_split into a complete cudf table.
 * 
 * @param metadata Metadata describing the contents of the partitions
 * @param partitions A buffer of anonymous bytes representing multiple partitions of data to be merged
 * @param partition_offsets Offsets into the partitions buffer indicating where each individual partition begins.
 *                          The number of partitions is partition_offsets.size()
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return A cudf table
 */
std::unique_ptr<cudf::table> shuffle_assemble(shuffle_split_metadata const& metadata,
                                              cudf::device_span<int8_t const> partitions,
                                              cudf::device_span<size_t const> partition_offsets,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

} // namespace cudf::spark_rapids_jni