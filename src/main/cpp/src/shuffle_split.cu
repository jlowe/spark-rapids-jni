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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>

#include "shuffle_split.hpp"

namespace spark_rapids_jni {

namespace {

using namespace cudf;

// The size that contiguous split uses internally as the GPU unit of work.
// The number of `desired_batch_size` batches equals the number of CUDA blocks
// that will be used for the main kernel launch (`copy_partitions`).
constexpr std::size_t desired_batch_size = 1 * 1024 * 1024;

// there will only be one copy of this 
struct cz_metadata_internal {
  // size_type                         num_columns = 0;
  size_type                         per_partition_metadata_size = 0;
  size_type                         max_depth = 0;
  shuffle_split_metadata            global_metadata;
};

/**
 * @brief Struct which contains information on a source buffer.
 *
 * The definition of "buffer" used throughout this module is a component piece of a
 * cudf column. So for example, a fixed-width column with validity would have 2 associated
 * buffers : the data itself and the validity buffer.  contiguous_split operates by breaking
 * each column up into it's individual components and copying each one as a separate kernel
 * block.
 */
struct src_buf_info {
  src_buf_info(cudf::type_id _type,
               int const* _offsets,
               int _offset_stack_pos,
               int _parent_offsets_index,
               bool _is_validity,
               size_type _column_offset)
    : type(_type),
      offsets(_offsets),
      offset_stack_pos(_offset_stack_pos),
      parent_offsets_index(_parent_offsets_index),
      is_validity(_is_validity),
      column_offset(_column_offset)
  {
  }

  src_buf_info(){}

  cudf::type_id type;
  int const* offsets;        // a pointer to device memory offsets if I am an offset buffer
  int offset_stack_pos;      // position in the offset stack buffer
  int parent_offsets_index;  // immediate parent that has offsets, or -1 if none
  bool is_validity;          // if I am a validity buffer
  size_type column_offset;   // offset in the case of a sliced column
};

/**
 * @brief Struct which contains information on a destination buffer.
 *
 * Similar to src_buf_info, dst_buf_info contains information on a destination buffer we
 * are going to copy to.  If we have N input buffers (which come from X columns), and
 * M partitions, then we have N*M destination buffers.
 */
struct dst_buf_info {
  // constant across all copy commands for this buffer
  std::size_t buf_size;  // total size of buffer, including padding
  int num_elements;      // # of elements to be copied
  int element_size;      // size of each element in bytes
  int num_rows;          // # of rows to be copied(which may be different from num_elements in the case of
                         // validity or offset buffers)

  int src_element_index;   // element index to start reading from my associated source buffer
  std::size_t dst_offset;  // my offset into the per-partition allocation, not including the per-partition metadata header size
  int value_shift;         // amount to shift values down by (for offset buffers)
  int bit_shift;           // # of bits to shift right by (for validity buffers)
  size_type valid_count;   // validity count for this block of work

  int src_buf_index;       // source buffer index
  int root_num_rows;     // for string columns, num_rows will be the number of chars. root_num_rows will be the number of top level rows
};

constexpr size_t size_to_batch_count(size_t bytes)
{
  return std::max(std::size_t{1}, util::round_up_unsafe(bytes, desired_batch_size) / desired_batch_size);
}

/**
 * @brief Copy a single buffer of column data, shifting values (for offset columns),
 * and validity (for validity buffers) as necessary.
 *
 * Copies a single partition of a source column buffer to a destination buffer. Shifts
 * element values by value_shift in the case of a buffer of offsets (value_shift will
 * only ever be > 0 in that case).  Shifts elements bitwise by bit_shift in the case of
 * a validity buffer (bit_shift will only ever be > 0 in that case).  This function assumes
 * value_shift and bit_shift will never be > 0 at the same time.
 *
 * This function expects:
 * - src may be a misaligned address
 * - dst must be an aligned address
 *
 * This function always does the ALU work related to value_shift and bit_shift because it is
 * entirely memory-bandwidth bound.
 *
 * @param dst Destination buffer
 * @param src Source buffer
 * @param t Thread index
 * @param num_elements Number of elements to copy
 * @param element_size Size of each element in bytes
 * @param src_element_index Element index to start copying at
 * @param stride Size of the kernel block
 * @param value_shift Shift incoming 4-byte offset values down by this amount
 * @param bit_shift Shift incoming data right by this many bits
 * @param num_rows Number of rows being copied
 * @param valid_count Optional pointer to a value to store count of set bits
 */
template <int block_size>
__device__ void copy_buffer(uint8_t* __restrict__ dst,
                            uint8_t const* __restrict__ src,
                            int t,
                            std::size_t num_elements,
                            std::size_t element_size,
                            std::size_t src_element_index,
                            uint32_t stride,
                            int value_shift,
                            int bit_shift,
                            std::size_t num_rows,
                            size_type* valid_count)
{
  src += (src_element_index * element_size);

  size_type thread_valid_count = 0;

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  std::size_t const num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  uint32_t const ofs = reinterpret_cast<uintptr_t>(src) % 4;
  std::size_t pos    = t * 16;
  stride *= 16;
  while (pos + 20 <= num_bytes) {
    // read from the nearest aligned address.
    uint32_t const* in32 = reinterpret_cast<uint32_t const*>((src + pos) - ofs);
    uint4 v              = uint4{in32[0], in32[1], in32[2], in32[3]};
    if (ofs || bit_shift) {
      v.x = __funnelshift_r(v.x, v.y, ofs * 8 + bit_shift);
      v.y = __funnelshift_r(v.y, v.z, ofs * 8 + bit_shift);
      v.z = __funnelshift_r(v.z, v.w, ofs * 8 + bit_shift);
      v.w = __funnelshift_r(v.w, in32[4], ofs * 8 + bit_shift);
    }
    v.x -= value_shift;
    v.y -= value_shift;
    v.z -= value_shift;
    v.w -= value_shift;
    reinterpret_cast<uint4*>(dst)[pos / 16] = v;
    if (valid_count) {
      thread_valid_count += (__popc(v.x) + __popc(v.y) + __popc(v.z) + __popc(v.w));
    }
    pos += stride;
  }

  // copy trailing bytes
  if (t == 0) {
    std::size_t remainder;
    if (num_bytes < 16) {
      remainder = num_bytes;
    } else {
      std::size_t const last_bracket = (num_bytes / 16) * 16;
      remainder                      = num_bytes - last_bracket;
      if (remainder < 4) {
        // we had less than 20 bytes for the last possible 16 byte copy, so copy 16 + the extra
        remainder += 16;
      }
    }

    // if we're performing a value shift (offsets), or a bit shift (validity) the # of bytes and
    // alignment must be a multiple of 4. value shifting and bit shifting are mutually exclusive
    // and will never both be true at the same time.
    if (value_shift || bit_shift) {
      std::size_t idx = (num_bytes - remainder) / 4;
      uint32_t v = remainder > 0 ? (reinterpret_cast<uint32_t const*>(src)[idx] - value_shift) : 0;

      constexpr size_type rows_per_element = 32;
      auto const have_trailing_bits = ((num_elements * rows_per_element) - num_rows) < bit_shift;
      while (remainder) {
        // if we're at the very last word of a validity copy, we do not always need to read the next
        // word to get the final trailing bits.
        auto const read_trailing_bits = bit_shift > 0 && remainder == 4 && have_trailing_bits;
        uint32_t const next           = (read_trailing_bits || remainder > 4)
                                          ? (reinterpret_cast<uint32_t const*>(src)[idx + 1] - value_shift)
                                          : 0;

        uint32_t const val = (v >> bit_shift) | (next << (32 - bit_shift));
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint32_t*>(dst)[idx] = val;
        v                                     = next;
        idx++;
        remainder -= 4;
      }
    } else {
      while (remainder) {
        std::size_t const idx = num_bytes - remainder--;
        uint32_t const val    = reinterpret_cast<uint8_t const*>(src)[idx];
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint8_t*>(dst)[idx] = val;
      }
    }
  }

  if (valid_count) {
    if (num_bytes == 0) {
      if (!t) { *valid_count = 0; }
    } else {
      using BlockReduce = cub::BlockReduce<size_type, block_size>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      size_type block_valid_count{BlockReduce(temp_storage).Sum(thread_valid_count)};
      if (!t) {
        // we may have copied more bits than there are actual rows in the output.
        // so we need to subtract off the count of any bits that shouldn't have been
        // considered during the copy step.
        std::size_t const max_row    = (num_bytes * 8);
        std::size_t const slack_bits = max_row > num_rows ? max_row - num_rows : 0;
        auto const slack_mask        = set_most_significant_bits(slack_bits);
        if (slack_mask > 0) {
          uint32_t const last_word = reinterpret_cast<uint32_t*>(dst + (num_bytes - 4))[0];
          block_valid_count -= __popc(last_word & slack_mask);
        }
        *valid_count = block_valid_count;
      }
    }
  }
}

/**
 * @brief Kernel which copies data from multiple source buffers to multiple
 * destination buffers.
 *
 * When doing a contiguous_split on X columns comprising N total internal buffers
 * with M splits, we end up having to copy N*M source/destination buffer pairs.
 * These logical copies are further subdivided to distribute the amount of work
 * to be done as evenly as possible across the multiprocessors on the device.
 * This kernel is arranged such that each block copies 1 source/destination pair.
 *
 * @param index_to_buffer A function that given a `buf_index` returns the destination buffer
 * @param src_bufs Input source buffers
 * @param buf_info Information on the range of values to be copied for each destination buffer
 */
template <int block_size, typename IndexToDstBuf>
CUDF_KERNEL void copy_partitions(IndexToDstBuf index_to_buffer,
                                 uint8_t const** src_bufs,
                                 dst_buf_info* buf_info)
{
  auto const buf_index     = blockIdx.x;
  auto const src_buf_index = buf_info[buf_index].src_buf_index;

  /*
  if(threadIdx.x == 0){
    printf("buf_index = %d, src_buf_index = %d, offset = %lu\n", (int)buf_index, (int)src_buf_index, (size_t)(buf_info[buf_index].dst_offset));
  }
  */

  // copy, shifting offsets and validity bits as needed
  copy_buffer<block_size>(
    // each buffer has a block of metadata at the very beginning that we need to skip past
    index_to_buffer(buf_index) + buf_info[buf_index].dst_offset,
    src_bufs[src_buf_index],
    threadIdx.x,
    buf_info[buf_index].num_elements,
    buf_info[buf_index].element_size,
    buf_info[buf_index].src_element_index,
    blockDim.x,
    buf_info[buf_index].value_shift,
    buf_info[buf_index].bit_shift,
    buf_info[buf_index].num_rows,
    buf_info[buf_index].valid_count > 0 ? &buf_info[buf_index].valid_count : nullptr);

  /*
  if(threadIdx.x == 0){
    printf("V(%d): %d\n", (int)buf_index, ((int*)(index_to_buffer(buf_index) + buf_info[buf_index].dst_offset))[0]);
  }
  */
}

// The block of functions below are all related:
//
// compute_offset_stack_size()
// setup_src_buf_data()
// count_src_bufs()
// setup_source_buf_info()
// build_output_columns()
//
// Critically, they all traverse the hierarchy of source columns and their children
// in a specific order to guarantee they produce various outputs in a consistent
// way.  For example, setup_src_buf_info() produces a series of information
// structs that must appear in the same order that setup_src_buf_data() produces
// buffers.
//
// So please be careful if you change the way in which these functions and
// functors traverse the hierarchy.

/**
 * @brief Returns whether or not the specified type is a column that contains offsets.
 */
bool is_offset_type(type_id id) { return (id == type_id::STRING or id == type_id::LIST); }

/**
 * @brief Compute total device memory stack size needed to process nested
 * offsets per-output buffer.
 *
 * When determining the range of rows to be copied for each output buffer
 * we have to recursively apply the stack of offsets from our parent columns
 * (lists or strings).  We want to do this computation on the gpu because offsets
 * are stored in device memory.  However we don't want to do recursion on the gpu, so
 * each destination buffer gets a "stack" of space to work with equal in size to
 * it's offset nesting depth.  This function computes the total size of all of those
 * stacks.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param offset_depth Current offset nesting depth
 *
 * @returns Total offset stack size needed for this range of columns
 */
template <typename InputIter>
std::size_t compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth = 0)
{
  return std::accumulate(begin, end, 0, [offset_depth](auto stack_size, column_view const& col) {
    auto const num_buffers = 1 + (col.nullable() ? 1 : 0);
    return stack_size + (offset_depth * num_buffers) +
           compute_offset_stack_size(
             col.child_begin(), col.child_end(), offset_depth + is_offset_type(col.type().id()));
  });
}

/**
 * @brief Retrieve all buffers for a range of source columns.
 *
 * Retrieve the individual buffers that make up a range of input columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param out_buf Iterator into output buffer infos
 *
 * @returns next output buffer iterator
 */
template <typename InputIter, typename OutputIter>
OutputIter setup_src_buf_data(InputIter begin, InputIter end, OutputIter out_buf)
{
  std::for_each(begin, end, [&out_buf](column_view const& col) {
    if (col.nullable()) {
      *out_buf = reinterpret_cast<uint8_t const*>(col.null_mask());
      out_buf++;
    }
    // NOTE: we're always returning the base pointer here. column-level offset is accounted
    // for later. Also, for some column types (string, list, struct) this pointer will be null
    // because there is no associated data with the root column.
    *out_buf = col.head<uint8_t>();
    out_buf++;

    out_buf = setup_src_buf_data(col.child_begin(), col.child_end(), out_buf);
  });
  return out_buf;
}

/**
 * @brief Count the total number of source buffers we will be copying
 * from.
 *
 * This count includes buffers for all input columns. For example a
 * fixed-width column with validity would be 2 buffers (data, validity).
 * A string column with validity would be 3 buffers (chars, offsets, validity).
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 *
 * @returns total number of source buffers for this range of columns
 */
template <typename InputIter>
size_type count_src_bufs(InputIter begin, InputIter end)
{
  auto buf_iter = thrust::make_transform_iterator(begin, [](column_view const& col) {
    auto const children_counts = count_src_bufs(col.child_begin(), col.child_end());
    return 1 + (col.nullable() ? 1 : 0) + children_counts;
  });
  return std::accumulate(buf_iter, buf_iter + std::distance(begin, end), 0);
}

/**
 * @brief Computes source buffer information for the copy kernel.
 *
 * For each input column to be split we need to know several pieces of information
 * in the copy kernel.  This function traverses the input columns and prepares this
 * information for the gpu.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param head Beginning of source buffer info array
 * @param current Current source buffer info to be written to
 * @param offset_stack_pos Integer representing our current offset nesting depth
 * (how many list or string levels deep we are)
 * @param parent_offset_index Index into src_buf_info output array indicating our nearest
 * containing list parent. -1 if we have no list parent
 * @param offset_depth Current offset nesting depth (how many list levels deep we are)
 *
 * @returns next src_buf_output after processing this range of input columns
 */
// setup source buf info
template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          std::vector<size_type>& metadata_col_to_buf_index,
                                                          rmm::cuda_stream_view stream,
                                                          int offset_stack_pos    = 0,
                                                          int parent_offset_index = -1,
                                                          int offset_depth        = 0);

/**
 * @brief Functor that builds source buffer information based on input columns.
 *
 * Called by setup_source_buf_info to build information for a single source column.  This function
 * will recursively call setup_source_buf_info in the case of nested types.
 */
struct buf_info_functor {
  src_buf_info* head;

  template <typename T>
  std::pair<src_buf_info*, size_type> operator()(column_view const& col,
                                                 src_buf_info* current,
                                                 int offset_stack_pos,
                                                 int parent_offset_index,
                                                 int offset_depth,
                                                 std::vector<size_type>& metadata_col_to_buf_index,
                                                 rmm::cuda_stream_view)
  {
    auto start = current;

    if (col.nullable()) {
      std::tie(current, offset_stack_pos) =
        add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
    }

    // info for the data buffer
    *current = src_buf_info(
      col.type().id(), nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
    current++;

    // starts as a count of buffers per input column. will be scanned later.
    metadata_col_to_buf_index.push_back(current - start);

    return {current, offset_stack_pos + offset_depth};
  }

  template <typename T, typename... Args>
  std::enable_if_t<std::is_same_v<T, cudf::dictionary32>, std::pair<src_buf_info*, size_type>>
  operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type");
  }

 private:
  std::pair<src_buf_info*, size_type> add_null_buffer(column_view const& col,
                                                      src_buf_info* current,
                                                      int offset_stack_pos,
                                                      int parent_offset_index,
                                                      int offset_depth)
  {
    // info for the validity buffer
    *current = src_buf_info(
      type_id::INT32, nullptr, offset_stack_pos, parent_offset_index, true, col.offset());

    return {current + 1, offset_stack_pos + offset_depth};
  }
};

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::string_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  std::vector<size_type>& metadata_col_to_buf_index,
  rmm::cuda_stream_view stream)
{
  auto start = current;

  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // the way strings are arranged, the strings column itself contains char data, but our child
  // offsets column actually contains our offsets. So our parent_offset_index is actually our child.

  // string columns don't necessarily have children if they are empty
  auto const has_offsets_child = col.num_children() > 0;

  // string columns contain the underlying chars data.
  *current = src_buf_info(type_id::STRING,
                          nullptr,
                          offset_stack_pos,
                          // if I have an offsets child, it's index will be my parent_offset_index
                          has_offsets_child ? ((current + 1) - head) : parent_offset_index,
                          false,
                          col.offset());

  // if I have offsets, I need to include that in the stack size
  offset_stack_pos += has_offsets_child ? offset_depth + 1 : offset_depth;
  current++;

  if (has_offsets_child) {
    CUDF_EXPECTS(col.num_children() == 1, "Encountered malformed string column");
    strings_column_view scv(col);

    // info for the offsets buffer
    auto offset_col = current;
    CUDF_EXPECTS(not scv.offsets().nullable(), "Encountered nullable string offsets column");
    *current = src_buf_info(type_id::INT32,
                            // note: offsets can be null in the case where the string column
                            // has been created with empty_like().
                            scv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                            offset_stack_pos,
                            parent_offset_index,
                            false,
                            col.offset());

    current++;
    offset_stack_pos += offset_depth;

    // since we are crossing an offset boundary, calculate our new depth and parent offset index.
    offset_depth++;
    parent_offset_index = offset_col - head;
  }

  // starts as a count of buffers per input column. will be scanned later.
  metadata_col_to_buf_index.push_back(current - start);

  return {current, offset_stack_pos};
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::list_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  std::vector<size_type>& metadata_col_to_buf_index,
  rmm::cuda_stream_view stream)
{
  lists_column_view lcv(col);
  auto start = current;

  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // list columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::LIST, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed list column");

  // info for the offsets buffer
  auto offset_col = current;
  *current        = src_buf_info(type_id::INT32,
                          // note: offsets can be null in the case where the lists column
                          // has been created with empty_like().
                          lcv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                          offset_stack_pos,
                          parent_offset_index,
                          false,
                          col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // since we are crossing an offset boundary, calculate our new depth and parent offset index.
  offset_depth++;
  parent_offset_index = offset_col - head;

  // starts as a count of buffers per input column. will be scanned later.
  metadata_col_to_buf_index.push_back(current - start);

  return setup_source_buf_info(col.child_begin() + 1,
                               col.child_end(),
                               head,
                               current,
                               metadata_col_to_buf_index,
                               stream,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::struct_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth,
  std::vector<size_type>& metadata_col_to_buf_index,
  rmm::cuda_stream_view stream)
{
  auto start = current;

  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // struct columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::STRUCT, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // starts as a count of buffers per input column. will be scanned later.
  metadata_col_to_buf_index.push_back(current - start);

  // recurse on children
  cudf::structs_column_view scv(col);
  std::vector<column_view> sliced_children;
  sliced_children.reserve(scv.num_children());
  std::transform(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(scv.num_children()),
    std::back_inserter(sliced_children),
    [&scv, &stream](size_type child_index) { return scv.get_sliced_child(child_index, stream); });
  return setup_source_buf_info(sliced_children.begin(),
                               sliced_children.end(),
                               head,
                               current,
                               metadata_col_to_buf_index,
                               stream,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          std::vector<size_type>& metadata_col_to_buf_index,
                                                          rmm::cuda_stream_view stream,
                                                          int offset_stack_pos,
                                                          int parent_offset_index,
                                                          int offset_depth)
{
  std::for_each(begin, end, [&](column_view const& col) {
    std::tie(current, offset_stack_pos) = cudf::type_dispatcher(col.type(),
                                                                buf_info_functor{head},
                                                                col,
                                                                current,
                                                                offset_stack_pos,
                                                                parent_offset_index,
                                                                offset_depth,
                                                                metadata_col_to_buf_index,
                                                                stream);
  });
  return {current, offset_stack_pos};
}

/**
 * @brief Given a column, processed split buffers, and a metadata builder, populate
 * the metadata for this column in the builder, and return a tuple of:
 * column size, data offset, bitmask offset and null count.
 *
 * @param src column_view to create metadata from
 * @param current_info dst_buf_info pointer reference, pointing to this column's buffer info
 *                     This is a pointer reference because it is updated by this function as the
 *                     columns's validity and data buffers are visited
 * @param mb A metadata_builder instance to update with the column's packed metadata
 * @param use_src_null_count True for the chunked_pack case where current_info has invalid null
 *                           count information. The null count should be taken
 *                           from `src` because this case is restricted to a single partition
 *                           (no splits)
 * @returns a std::tuple containing:
 *          column size, data offset, bitmask offset, and null count
 */
/*
template <typename BufInfo>
std::tuple<size_type, int64_t, int64_t, size_type> build_output_column_metadata(
  column_view const& src,
  BufInfo& current_info,
  detail::metadata_builder& mb,
  bool use_src_null_count)
{
  auto [bitmask_offset, null_count] = [&]() {
    if (src.nullable()) {
      // offsets in the existing serialized_column metadata are int64_t
      // that's the reason for the casting in this code.
      int64_t const bitmask_offset =
        current_info->num_elements == 0
          ? -1  // this means that the bitmask buffer pointer should be nullptr
          : static_cast<int64_t>(current_info->dst_offset);

      // use_src_null_count is used for the chunked contig split case, where we have
      // no splits: the null_count is just the source column's null_count
      size_type const null_count = use_src_null_count
                                     ? src.null_count()
                                     : (current_info->num_elements == 0
                                          ? 0
                                          : (current_info->num_rows - current_info->valid_count));

      ++current_info;
      return std::pair(bitmask_offset, null_count);
    }
    return std::pair(static_cast<int64_t>(-1), 0);
  }();

  // size/data pointer for the column
  auto const col_size = [&]() {
    // if I am a string column, I need to use the number of rows from my child offset column. the
    // number of rows in my dst_buf_info struct will be equal to the number of chars, which is
    // incorrect. this is a quirk of how cudf stores strings.
    if (src.type().id() == type_id::STRING) {
      // if I have no children (no offsets), then I must have a row count of 0
      if (src.num_children() == 0) { return 0; }

      // otherwise my actual number of rows will be the num_rows field of the next dst_buf_info
      // struct (our child offsets column)
      return (current_info + 1)->num_rows;
    }

    // otherwise the number of rows is the number of elements
    return static_cast<size_type>(current_info->num_elements);
  }();
  int64_t const data_offset =
    col_size == 0 || src.head() == nullptr ? -1 : static_cast<int64_t>(current_info->dst_offset);

  mb.add_column_info_to_meta(
    src.type(), col_size, null_count, data_offset, bitmask_offset, src.num_children());

  ++current_info;
  return {col_size, data_offset, bitmask_offset, null_count};
}
*/

/**
 * @brief Given a set of input columns and processed split buffers, produce
 * output columns.
 *
 * After performing the split we are left with 1 large buffer per incoming split
 * partition.  We need to traverse this buffer and distribute the individual
 * subpieces that represent individual columns and children to produce the final
 * output columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param info_begin Iterator of dst_buf_info structs containing information about each
 * copied buffer
 * @param out_begin Output iterator of column views
 * @param base_ptr Pointer to the base address of copied data for the working partition
 *
 * @returns new dst_buf_info iterator after processing this range of input columns
 */
/*
template <typename InputIter, typename BufInfo, typename Output>
BufInfo build_output_columns(InputIter begin,
                             InputIter end,
                             BufInfo info_begin,
                             Output out_begin,
                             uint8_t const* const base_ptr,
                             detail::metadata_builder& mb)
{
  auto current_info = info_begin;
  std::transform(begin, end, out_begin, [&current_info, base_ptr, &mb](column_view const& src) {
    auto [col_size, data_offset, bitmask_offset, null_count] =
      build_output_column_metadata<BufInfo>(src, current_info, mb, false);

    auto const bitmask_ptr =
      base_ptr != nullptr && bitmask_offset != -1
        ? reinterpret_cast<bitmask_type const*>(base_ptr + static_cast<uint64_t>(bitmask_offset))
        : nullptr;

    // size/data pointer for the column
    uint8_t const* data_ptr = base_ptr != nullptr && data_offset != -1
                                ? base_ptr + static_cast<uint64_t>(data_offset)
                                : nullptr;

    // children
    auto children = std::vector<column_view>{};
    children.reserve(src.num_children());

    current_info = build_output_columns(
      src.child_begin(), src.child_end(), current_info, std::back_inserter(children), base_ptr, mb);

    return column_view{
      src.type(), col_size, data_ptr, bitmask_ptr, null_count, 0, std::move(children)};
  });

  return current_info;
}
*/

/**
 * @brief Given a set of input columns, processed split buffers, and a metadata_builder,
 * append column metadata using the builder.
 *
 * After performing the split we are left with 1 large buffer per incoming split
 * partition.  We need to traverse this buffer and distribute the individual
 * subpieces that represent individual columns and children to produce the final
 * output columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param info_begin Iterator of dst_buf_info structs containing information about each
 * copied buffer
 * @param mb packed column metadata builder
 *
 * @returns new dst_buf_info iterator after processing this range of input columns
 */
/*
template <typename InputIter, typename BufInfo>
BufInfo populate_metadata(InputIter begin,
                          InputIter end,
                          BufInfo info_begin,
                          detail::metadata_builder& mb)
{
  auto current_info = info_begin;
  std::for_each(begin, end, [&current_info, &mb](column_view const& src) {
    build_output_column_metadata<BufInfo>(src, current_info, mb, true);

    // children
    current_info = populate_metadata(src.child_begin(), src.child_end(), current_info, mb);
  });

  return current_info;
}
*/

/**
 * @brief Functor that retrieves the size of a destination buffer
 */
struct buf_size_functor {
  dst_buf_info const* ci;
  size_t num_bufs;
  // std::size_t per_partition_metadata_size;
  std::size_t operator() __device__(int index) 
  {
    return index >= num_bufs ? 0 : ci[index].buf_size;
  }
};

/**
 * @brief Functor that retrieves the split "key" for a given output
 * buffer index.
 *
 * The key is simply the partition index.
 */
struct split_key_functor {
  int const num_src_bufs;
  int operator() __device__(int buf_index) const { return buf_index / num_src_bufs; }
};

/**
 * @brief Output iterator for writing values to the dst_offset field of the
 * dst_buf_info struct
 */
struct dst_offset_output_iterator {
  dst_buf_info* c;
  using value_type        = std::size_t;
  using difference_type   = std::size_t;
  using pointer           = std::size_t*;
  using reference         = std::size_t&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i) { return {c + i}; }

  dst_offset_output_iterator& operator++ __host__ __device__()
  {
    c++;
    return *this;
  }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->dst_offset; }
};

/**
 * @brief Output iterator for writing values to the valid_count field of the
 * dst_buf_info struct
 */
struct dst_valid_count_output_iterator {
  dst_buf_info* c;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_valid_count_output_iterator operator+ __host__ __device__(int i) { return {c + i}; }

  dst_valid_count_output_iterator& operator++ __host__ __device__()
  {
    c++;
    return *this;
  }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->valid_count; }
};

/**
 * @brief Functor for computing size of data elements for a given cudf type.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct size_of_helper {
  template <typename T>
  constexpr std::enable_if_t<!is_fixed_width<T>() && !std::is_same_v<T, cudf::string_view>, size_t>
    __device__ operator()() const
  {    
    return 0;
  }

  template <typename T>
  constexpr std::enable_if_t<!is_fixed_width<T>() && std::is_same_v<T, cudf::string_view>, size_t>
    __device__ operator()() const
  {
    return sizeof(cudf::device_storage_type_t<int8_t>);
  }

  template <typename T>
  constexpr std::enable_if_t<is_fixed_width<T>(), size_t> __device__ operator()() const noexcept
  {
    return sizeof(cudf::device_storage_type_t<T>);
  }
};

/**
 * @brief Functor for returning the number of batches an input buffer is being
 * subdivided into during the repartitioning step.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct num_batches_func {
  thrust::pair<std::size_t, std::size_t> const* const batches;
  __device__ std::size_t operator()(size_type i) const { return thrust::get<0>(batches[i]); }
};

/**
 * @brief Get the size in bytes of a batch described by `dst_buf_info`.
 */
struct batch_byte_size_function {
  size_type const num_batches;
  dst_buf_info const* const infos;
  __device__ std::size_t operator()(size_type i) const
  {
    if (i == num_batches) { return 0; }
    auto const& buf = *(infos + i);
    std::size_t const bytes =
      static_cast<std::size_t>(buf.num_elements) * static_cast<std::size_t>(buf.element_size);
    return util::round_up_unsafe(bytes, shuffle_split_partition_data_align);
  }
};

/**
 * @brief Get the input buffer index given the output buffer index.
 */
struct out_to_in_index_function {
  size_type const* const batch_offsets;
  int const num_bufs;
  __device__ int operator()(size_type i) const
  {
    int ret = static_cast<size_type>(
             thrust::upper_bound(thrust::seq, batch_offsets, batch_offsets + num_bufs + 1, i) -
             batch_offsets) -
           1;    
    return ret;
  }
};

struct partition_buf_size_func {
  cudf::device_span<size_t const> buf_sizes;
  __device__ size_t operator()(int i)
  {
    return i >= buf_sizes.size() ? 0 : buf_sizes[i];
  }
};

// packed block of memory 1: split indices and src_buf_info structs
struct packed_split_indices_and_src_buf_info {
  packed_split_indices_and_src_buf_info(cudf::table_view const& input,
                                        std::vector<size_type> const& splits,
                                        std::size_t num_partitions,
                                        cudf::size_type num_src_bufs,
                                        cz_metadata_internal const& metadata,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref temp_mr)
    : indices_size(
        cudf::util::round_up_safe((num_partitions + 1) * sizeof(size_type), shuffle_split_partition_data_align)),
      src_buf_info_size(
        cudf::util::round_up_safe(num_src_bufs * sizeof(src_buf_info), shuffle_split_partition_data_align)),
      // host-side
      h_indices_and_source_info(indices_size + src_buf_info_size),
      h_indices{reinterpret_cast<size_type*>(h_indices_and_source_info.data())},
      h_src_buf_info{
        reinterpret_cast<src_buf_info*>(h_indices_and_source_info.data() + indices_size)}
  {
    // compute splits -> indices.
    // these are row numbers per split
    h_indices[0]              = 0;
    h_indices[num_partitions] = input.column(0).size();
    std::copy(splits.begin(), splits.end(), std::next(h_indices));

    // mapping of metadata column to src/dst buffer index, which we will need later on to 
    // pack row counts
    std::vector<size_type> metadata_col_to_buf_index;
    metadata_col_to_buf_index.reserve(num_src_bufs); // worst case

    // setup source buf info
    setup_source_buf_info(input.begin(), input.end(), h_src_buf_info, h_src_buf_info, metadata_col_to_buf_index, stream);

    auto const metadata_size = metadata.global_metadata.col_info.size();
    metadata_col_to_buf_index_size = cudf::util::round_up_safe(metadata_size * sizeof(size_type), shuffle_split_partition_data_align);

    offset_stack_partition_size = compute_offset_stack_size(input.begin(), input.end());
    offset_stack_size           = offset_stack_partition_size * num_partitions * sizeof(size_type);
    
    // device-side
    // gpu-only : stack space needed for nested list offset calculation
    d_indices_and_source_info =
      rmm::device_buffer(metadata_col_to_buf_index_size + indices_size + src_buf_info_size + offset_stack_size, stream, temp_mr);
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(d_indices_and_source_info.data());
    
  
    d_metadata_col_to_buf_index = reinterpret_cast<size_type*>(base_ptr);

    d_indices      = reinterpret_cast<size_type*>(base_ptr + metadata_col_to_buf_index_size);
    d_src_buf_info = reinterpret_cast<src_buf_info*>(base_ptr + metadata_col_to_buf_index_size + indices_size);
    d_offset_stack =
      reinterpret_cast<size_type*>(base_ptr + metadata_col_to_buf_index_size + indices_size + src_buf_info_size);

    // compute metadata col index -> buf index 
    cudaMemcpyAsync(d_metadata_col_to_buf_index, metadata_col_to_buf_index.data(), sizeof(size_type) * metadata_col_to_buf_index.size(), cudaMemcpyHostToDevice, stream);
    thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                           d_metadata_col_to_buf_index,
                           d_metadata_col_to_buf_index + metadata_size,
                           d_metadata_col_to_buf_index);
    // print_span(cudf::device_span<size_type const>{d_metadata_col_to_buf_index, metadata_size});

    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_indices, h_indices, indices_size + src_buf_info_size, cudaMemcpyDefault, stream.value()));
  }

  size_type const indices_size;
  std::size_t const src_buf_info_size;
  std::size_t offset_stack_size;

  std::vector<uint8_t> h_indices_and_source_info;
  rmm::device_buffer d_indices_and_source_info;

  size_type* const h_indices;
  src_buf_info* const h_src_buf_info;

  // data for shuffle split
  size_t metadata_col_to_buf_index_size;
  size_type* d_metadata_col_to_buf_index;

  int offset_stack_partition_size;
  size_type* d_indices;
  src_buf_info* d_src_buf_info;
  size_type* d_offset_stack;
};

// packed block of memory 2: partition buffer sizes and dst_buf_info structs
struct packed_partition_buf_size_and_dst_buf_info {
  packed_partition_buf_size_and_dst_buf_info(std::size_t num_partitions,
                                             std::size_t num_bufs,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref temp_mr)
    : stream(stream),
      partition_sizes_size{cudf::util::round_up_safe(num_partitions * sizeof(std::size_t), shuffle_split_partition_data_align)},
      dst_buf_info_size{cudf::util::round_up_safe(num_bufs * sizeof(dst_buf_info), shuffle_split_partition_data_align)},
      // host-side
      h_partition_sizes_and_dst_info(partition_sizes_size + dst_buf_info_size),
      // h_buf_sizes{reinterpret_cast<std::size_t*>(h_buf_sizes_and_dst_info.data())},
      h_dst_buf_info{
        reinterpret_cast<dst_buf_info*>(h_partition_sizes_and_dst_info.data() + partition_sizes_size)},
      // device-side
      d_partition_sizes_and_dst_info(partition_sizes_size + dst_buf_info_size, stream, temp_mr),
      d_partition_sizes{reinterpret_cast<std::size_t*>(d_partition_sizes_and_dst_info.data())},
      // destination buffer info
      d_dst_buf_info{reinterpret_cast<dst_buf_info*>(
        static_cast<uint8_t*>(d_partition_sizes_and_dst_info.data()) + partition_sizes_size)}
  {
  }

  /*
  void copy_to_host()
  {    
    // DtoH buf sizes and col info back to the host
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_buf_sizes,
                                  d_buf_sizes,
                                  buf_sizes_size + dst_buf_info_size,
                                  cudaMemcpyDefault,
                                  stream.value()));                                  
  }
  */

  rmm::cuda_stream_view const stream;

  // partition sizes and destination info (used in batched copies)
  std::size_t const partition_sizes_size;
  std::size_t const dst_buf_info_size;

  std::vector<uint8_t> h_partition_sizes_and_dst_info;
  //std::size_t* const h_buf_sizes;
  dst_buf_info* const h_dst_buf_info;

  std::size_t h_dst_buf_total_size;

  rmm::device_buffer d_partition_sizes_and_dst_info;
  std::size_t* const d_partition_sizes;     // length: the # of partitions
  dst_buf_info* const d_dst_buf_info;       // length: the # of partitions * number of source buffers
};

// Packed block of memory 3:
// Pointers to source and destination buffers (and stack space on the
// gpu for offset computation)
struct packed_src_and_dst_pointers {
  packed_src_and_dst_pointers(cudf::table_view const& input,
                              std::size_t num_partitions,
                              cudf::size_type num_src_bufs,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref temp_mr)
    : stream(stream),
      src_bufs_size{cudf::util::round_up_safe(num_src_bufs * sizeof(uint8_t*), shuffle_split_partition_data_align)},
      dst_bufs_size{cudf::util::round_up_safe(/*num_partitions*/1 * sizeof(uint8_t*), shuffle_split_partition_data_align)},
      // host-side
      h_src_and_dst_buffers(src_bufs_size + dst_bufs_size),
      h_src_bufs{reinterpret_cast<uint8_t const**>(h_src_and_dst_buffers.data())},
      h_dst_buf{reinterpret_cast<uint8_t**>(h_src_and_dst_buffers.data() + src_bufs_size)},
      // device-side
      d_src_and_dst_buffers{rmm::device_buffer(src_bufs_size + dst_bufs_size, stream, temp_mr)},
      d_src_bufs{reinterpret_cast<uint8_t const**>(d_src_and_dst_buffers.data())},
      d_dst_buf{reinterpret_cast<uint8_t**>(
        reinterpret_cast<uint8_t*>(d_src_and_dst_buffers.data()) + src_bufs_size)}
  {
    // setup src buffers
    setup_src_buf_data(input.begin(), input.end(), h_src_bufs);
  }

  void copy_to_device()
  {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_src_and_dst_buffers.data(),
                                  h_src_and_dst_buffers.data(),
                                  src_bufs_size + dst_bufs_size,
                                  cudaMemcpyDefault,
                                  stream.value()));
  }

  rmm::cuda_stream_view const stream;
  std::size_t const src_bufs_size;
  std::size_t const dst_bufs_size;

  std::vector<uint8_t> h_src_and_dst_buffers;
  uint8_t const** const h_src_bufs;
  //uint8_t** const h_dst_bufs;
  uint8_t** const h_dst_buf;

  rmm::device_buffer d_src_and_dst_buffers;
  uint8_t const** const d_src_bufs;
  //uint8_t** const d_dst_bufs;
  uint8_t** const d_dst_buf;
};

/**
 * @brief Create an instance of `packed_src_and_dst_pointers` populating destination
 * partition buffers (if any) from `out_buffers`. In the chunked_pack case
 * `out_buffers` is empty, and the destination pointer is provided separately
 * to the `copy_partitions` kernel.
 *
 * @param input source table view
 * @param num_partitions the number of partitions (1 meaning no splits)
 * @param num_src_bufs number of buffers for the source columns including children
 * @param out_buffers the destination buffers per partition if in the non-chunked case
 * @param stream Optional CUDA stream on which to execute kernels
 * @param temp_mr A memory resource for temporary and scratch space
 *
 * @returns new unique pointer to packed_src_and_dst_pointers
 */
std::unique_ptr<packed_src_and_dst_pointers> setup_src_and_dst_pointers(
  cudf::table_view const& input,
  std::size_t num_partitions,
  cudf::size_type num_src_bufs,
  rmm::device_buffer& out_buffer,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref temp_mr)
{
  auto src_and_dst_pointers = std::make_unique<packed_src_and_dst_pointers>(
    input, num_partitions, num_src_bufs, stream, temp_mr);

  /*
  std::transform(
    out_buffers.begin(), out_buffers.end(), src_and_dst_pointers->h_dst_bufs, [](auto& buf) {
      return static_cast<uint8_t*>(buf.data());
    });
    */
  src_and_dst_pointers->h_dst_buf[0] = static_cast<uint8_t*>(out_buffer.data());
  

  // copy the struct to device memory to access from the kernel
  src_and_dst_pointers->copy_to_device();

  return src_and_dst_pointers;
}

template <typename InputIter>
std::pair<size_type, size_type> count_internal_columns(InputIter begin, InputIter end, int depth = 0)
{ 
  /*
  auto child_count = [&](column_view const& col){
    if(col.type().id() == cudf::type_id::STRUCT){
      return count_internal_columns(col.child_begin(), col.child_end(), depth+1);
    } else if(col.type().id() == cudf::type_id::LIST){
      cudf::lists_column_view lcv(col);
      std::vector<cudf::column_view> children({lcv.child()});
      return count_internal_columns(children.begin(), children.end(), depth+1);
    }
    return {0};
  };
  auto buf_iter = thrust::make_transform_iterator(begin, [&](column_view const& col) {
    auto const children = child_count(col);
    return {1 + children.first, 1 + children.second};
  });
  
  return std::accumulate(buf_iter, buf_iter + std::distance(begin, end), {0, 0}, [](std::pair<size_type, size_type> const& a, std::pair<size_type, size_type> const& b) -> std::pair<size_type, size_type>{
    return {a.first + b.first, std::max(a.second, b.second)};
  });
  */

  auto child_count = [&](column_view const& col, int depth) -> std::pair<size_type, size_type> {
    if(col.type().id() == cudf::type_id::STRUCT){
      return count_internal_columns(col.child_begin(), col.child_end(), depth+1);
    } else if(col.type().id() == cudf::type_id::LIST){
      cudf::lists_column_view lcv(col);
      std::vector<cudf::column_view> children({lcv.child()});
      return count_internal_columns(children.begin(), children.end(), depth+1);
    }
    return {0, depth};
  };

  size_type col_count = 0;
  size_type max_depth = 0;
  std::for_each(begin, end, [&](column_view const& col){
    auto const cc = child_count(col, depth);
    col_count += (1 + cc.first);
    max_depth = std::max(max_depth, cc.second);
  });

  return {col_count, max_depth};
}

template <typename InputIter>
void populate_column_data(cz_metadata_internal& meta, InputIter begin, InputIter end)
{
  std::for_each(begin, end, [&meta](column_view const& col){
    // strings need to store an additional char count
    meta.per_partition_metadata_size += col.type().id() == cudf::type_id::STRING ? 4 : 0;

    switch(col.type().id()){
    case cudf::type_id::STRUCT:
      meta.global_metadata.col_info.push_back({col.type().id(), col.num_children()});
      populate_column_data(meta, col.child_begin(), col.child_end());
      break;
    
    case cudf::type_id::LIST: {
      meta.global_metadata.col_info.push_back({col.type().id(), 1});
      cudf::lists_column_view lcv(col);
      std::vector<cudf::column_view> children({lcv.child()});
      populate_column_data(meta, children.begin(), children.end());
      } break;

    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
      // TODO: scale.
      meta.global_metadata.col_info.push_back({col.type().id(), 0});
      break;

    default:
      meta.global_metadata.col_info.push_back({col.type().id(), 0});
      break;
    }
  });
}

// returns global metadata describing the table and the size of the
// internal per-partition data
cz_metadata_internal compute_metadata(cudf::table_view const& input)
{
  auto const [num_internal_columns, max_depth] = count_internal_columns(input.begin(), input.end());

  // compute the metadata
  cz_metadata_internal ret;
  ret.global_metadata.col_info.reserve(num_internal_columns);
  // 4 byte row count
  ret.per_partition_metadata_size += 4;
  // 1 bit indicating presence of null vector, per internal column
  ret.per_partition_metadata_size += (cudf::util::round_up_safe(num_internal_columns, 32) / 32) * sizeof(bitmask_type);
  populate_column_data(ret, input.begin(), input.end());
  // pad out to shuffle_split_partition_data_align bytes
  ret.per_partition_metadata_size = cudf::util::round_up_safe(ret.per_partition_metadata_size, static_cast<size_type>(shuffle_split_partition_data_align));
  ret.max_depth = max_depth;

  return ret;
}

/**
 * @brief Create an instance of `packed_partition_buf_size_and_dst_buf_info` containing
 * the partition-level dst_buf_info structs for each partition and column buffer.
 *
 * @param input source table view
 * @param splits the numeric value (in rows) for each split, empty for 1 partition
 * @param num_partitions the number of partitions create (1 meaning no splits)
 * @param num_src_bufs number of buffers for the source columns including children
 * @param num_bufs num_src_bufs times the number of partitions
 * @param stream Optional CUDA stream on which to execute kernels
 * @param temp_mr A memory resource for temporary and scratch space
 *
 * @returns new unique pointer to `packed_partition_buf_size_and_dst_buf_info`
 */
std::pair<std::unique_ptr<packed_partition_buf_size_and_dst_buf_info>, std::unique_ptr<packed_split_indices_and_src_buf_info>> compute_splits(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  std::size_t num_partitions,
  cudf::size_type num_src_bufs,
  std::size_t num_bufs,
  cz_metadata_internal const& metadata,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref temp_mr)
{
  auto partition_buf_size_and_dst_buf_info =
    std::make_unique<packed_partition_buf_size_and_dst_buf_info>(
      num_partitions, num_bufs, stream, temp_mr);

  auto const d_dst_buf_info = partition_buf_size_and_dst_buf_info->d_dst_buf_info;
  auto const d_partition_sizes    = partition_buf_size_and_dst_buf_info->d_partition_sizes;

  auto split_indices_and_src_buf_info = std::make_unique<packed_split_indices_and_src_buf_info>(
    input, splits, num_partitions, num_src_bufs, metadata, stream, temp_mr);

  auto const d_src_buf_info = split_indices_and_src_buf_info->d_src_buf_info;
  auto const offset_stack_partition_size =
    split_indices_and_src_buf_info->offset_stack_partition_size;
  auto const d_offset_stack = split_indices_and_src_buf_info->d_offset_stack;
  auto const d_indices      = split_indices_and_src_buf_info->d_indices;

  // compute sizes of each column in each partition, including alignment.
  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    thrust::make_counting_iterator<std::size_t>(0),
    thrust::make_counting_iterator<std::size_t>(num_bufs),
    d_dst_buf_info,
    cuda::proclaim_return_type<dst_buf_info>([d_src_buf_info,
                                              offset_stack_partition_size,
                                              d_offset_stack,
                                              d_indices,
                                              num_src_bufs] __device__(std::size_t t) {
      int const split_index   = t / num_src_bufs;
      int const src_buf_index = t % num_src_bufs;
      auto const& src_info    = d_src_buf_info[src_buf_index];

      // apply nested offsets (lists and string columns).
      //
      // We can't just use the incoming row indices to figure out where to read from in a
      // nested list situation.  We have to apply offsets every time we cross a boundary
      // (list or string).  This loop applies those offsets so that our incoming row_index_start
      // and row_index_end get transformed to our final values.
      //
      int const stack_pos = src_info.offset_stack_pos + (split_index * offset_stack_partition_size);
      size_type* offset_stack  = &d_offset_stack[stack_pos];
      int parent_offsets_index = src_info.parent_offsets_index;
      int stack_size           = 0;
      int root_column_offset   = src_info.column_offset;
      int const root_row_start = d_indices[split_index] + root_column_offset;
      int const root_row_end = d_indices[split_index + 1] + root_column_offset;
      int const root_row_count = root_row_end - root_row_start;
      while (parent_offsets_index >= 0) {
        offset_stack[stack_size++] = parent_offsets_index;
        root_column_offset         = d_src_buf_info[parent_offsets_index].column_offset;
        parent_offsets_index       = d_src_buf_info[parent_offsets_index].parent_offsets_index;
      }
      // make sure to include the -column- offset on the root column in our calculation.
      int row_start = d_indices[split_index] + root_column_offset;
      int row_end   = d_indices[split_index + 1] + root_column_offset;
      while (stack_size > 0) {
        stack_size--;
        auto const offsets = d_src_buf_info[offset_stack[stack_size]].offsets;
        // this case can happen when you have empty string or list columns constructed with
        // empty_like()
        if (offsets != nullptr) {
          row_start = offsets[row_start];
          row_end   = offsets[row_end];
        }
      }

      // final element indices and row count
      int const src_element_index = src_info.is_validity ? row_start / 32 : row_start;
      int const num_rows          = row_end - row_start;
      // if I am an offsets column, all my values need to be shifted
      int const value_shift = src_info.offsets == nullptr ? 0 : src_info.offsets[row_start];
      // if I am a validity column, we may need to shift bits
      int const bit_shift = src_info.is_validity ? row_start % 32 : 0;
      // # of rows isn't necessarily the same as # of elements to be copied.
      auto const num_elements = [&]() {
        if (src_info.offsets != nullptr && num_rows > 0) {
          return num_rows + 1;
        } else if (src_info.is_validity) {
          return (num_rows + 31) / 32;
        }
        return num_rows;
      }();
      int const element_size = cudf::type_dispatcher(data_type{src_info.type}, size_of_helper{});
      std::size_t const bytes =
        static_cast<std::size_t>(num_elements) * static_cast<std::size_t>(element_size);
      return dst_buf_info{util::round_up_unsafe(bytes, shuffle_split_partition_data_align),
                          num_elements,
                          element_size,
                          num_rows,
                          src_element_index,
                          0,
                          value_shift,
                          bit_shift,
                          src_info.is_validity ? 1 : 0,
                          src_buf_index,
                          root_row_count};
    }));
  
  // - compute total size of each partition and total buffer size overall
  // - compute start offset for each destination buffer within each split  
  {
    auto const keys = cudf::detail::make_counting_transform_iterator(
      0, split_key_functor{static_cast<int>(num_src_bufs)});
    auto buf_sizes =
      cudf::detail::make_counting_transform_iterator(0, buf_size_functor{d_dst_buf_info, num_bufs});

    // reduce to compute sizes, then add in per_partition_metadata_size
    thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                          keys,
                          keys + num_bufs,
                          buf_sizes,
                          thrust::make_discard_iterator(),
                          d_partition_sizes);
    
    /*
    {
      std::vector<size_t> h_partition_sizes(num_partitions);
      cudaMemcpy(h_partition_sizes.data(), d_partition_sizes, sizeof(size_t) * num_partitions, cudaMemcpyDeviceToHost);
      for(size_t idx=0; idx<num_partitions; idx++){
        printf("HBS(%lu): %lu\n", idx, h_partition_sizes[idx]);
      }
    }
    */
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                      d_partition_sizes,
                      d_partition_sizes + num_partitions,
                      d_partition_sizes,
                      [per_partition_metadata_size = metadata.per_partition_metadata_size] __device__ (std::size_t partition_size){
                        return util::round_up_unsafe(partition_size + per_partition_metadata_size, shuffle_split_partition_data_align);
                      });

    // print_span(cudf::device_span<size_t const>{d_partition_sizes, num_partitions});
    
    // total size
    partition_buf_size_and_dst_buf_info->h_dst_buf_total_size = thrust::reduce(rmm::exec_policy(stream, temp_mr),
                                                                               d_partition_sizes,
                                                                               d_partition_sizes + num_partitions);

    // scan to per-partition destination buf offsets (num_src_bufs * num_partitions), then add metdata offset
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                           buf_sizes,
                           buf_sizes + num_bufs,
                           dst_offset_output_iterator{d_dst_buf_info},
                           std::size_t{0});
    // add metadata header offset
    auto iter = thrust::make_counting_iterator(0);
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + num_bufs,
                    [per_partition_metadata_size = metadata.per_partition_metadata_size,
                     bufs_per_partition = num_src_bufs,
                     d_dst_buf_info]  __device__ (size_type i){

      auto const partition_index = i / bufs_per_partition;
      auto const metadata_offset = (partition_index + 1) * per_partition_metadata_size;
      d_dst_buf_info[i].dst_offset += metadata_offset;
      // printf("dst(%i): %lu\n", i, d_dst_buf_info[i].dst_offset);
    });    
  }

  /*
  // compute start offset for each output buffer for each split
  {
    auto const keys = cudf::detail::make_counting_transform_iterator(
      0, split_key_functor{static_cast<int>(num_src_bufs)});
    auto values =
      cudf::detail::make_counting_transform_iterator(0, buf_size_functor{d_dst_buf_info});

    thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                  keys,
                                  keys + num_bufs,
                                  values,
                                  dst_offset_output_iterator{d_dst_buf_info},
                                  std::size_t{0});
  }
  */

  // compute start offset for each destination buffer within each split  
  {
    /*
    auto sizes =
        cudf::detail::make_counting_transform_iterator(0, [num_bufs, d_buf_sizes] __device__ (size_t i) -> size_t {
          return i >= num_bufs ? 0 : d_buf_sizes[i];
        });
        */
       /*
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                           d_buf_sizes,
                           d_buf_sizes + num_bufs,
                           dst_offset_output_iterator{d_dst_buf_info},
                           std::size_t{0});
                           */
    
    /*
    size_t last_offset;
    size_t last_size;
    cudaMemcpyAsync(&last_offset, &(d_dst_buf_info[num_bufs-1].dst_offset), sizeof(size_t), cudaMemcpyDeviceToHost, stream);    
    cudaMemcpyAsync(&last_size, &d_buf_sizes[num_bufs-1], sizeof(size_t), cudaMemcpyDeviceToHost, stream);
    stream.synchronize();
    partition_buf_size_and_dst_buf_info->h_dst_buf_total_size = last_offset + last_size;
    */
    
    // blech
    // cudaMemcpyAsync(&(partition_buf_size_and_dst_buf_info->h_dst_buf_total_size), &(d_dst_buf_info[num_bufs].dst_offset), sizeof(size_t), cudaMemcpyDeviceToHost, stream);
  }

  // partition_buf_size_and_dst_buf_info->copy_to_host();

  return {std::move(partition_buf_size_and_dst_buf_info), std::move(split_indices_and_src_buf_info)};
}

/**
 * @brief Struct containing information about the actual batches we will send to the
 * `copy_partitions` kernel and the number of iterations we need to carry out this copy.
 *
 * For the non-chunked contiguous_split case, this contains the batched dst_buf_infos and the
 * number of iterations is going to be 1 since the non-chunked case is single pass.
 *
 * For the chunked_pack case, this also contains the batched dst_buf_infos for all
 * iterations in addition to helping keep the state about what batches have been copied so far
 * and what are the sizes (in bytes) of each iteration.
 */
struct chunk_iteration_state {
  chunk_iteration_state(rmm::device_uvector<dst_buf_info> _d_batched_dst_buf_info,
                        rmm::device_uvector<size_type> _d_batch_offsets,
                        std::vector<std::size_t>&& _h_num_buffs_per_iteration,
                        std::vector<std::size_t>&& _h_size_of_buffs_per_iteration,
                        std::size_t total_size)
    : num_iterations(_h_num_buffs_per_iteration.size()),
      current_iteration{0},
      starting_batch{0},
      d_batched_dst_buf_info(std::move(_d_batched_dst_buf_info)),
      d_batch_offsets(std::move(_d_batch_offsets)),
      h_num_buffs_per_iteration(std::move(_h_num_buffs_per_iteration)),
      h_size_of_buffs_per_iteration(std::move(_h_size_of_buffs_per_iteration)),
      total_size(total_size)
  {
  }

  static std::unique_ptr<chunk_iteration_state> create(
    rmm::device_uvector<thrust::pair<std::size_t, std::size_t>> const& batches,
    int num_bufs,
    dst_buf_info* d_orig_dst_buf_info,
    std::size_t h_dst_buf_total_size,
    std::size_t num_partitions,
    std::size_t user_buffer_size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr);

  /**
   * @brief As of the time of the call, return the starting 1MB batch index, and the
   * number of batches to copy.
   *
   * @return the current iteration's starting_batch and batch count as a pair
   */
  std::pair<std::size_t, std::size_t> get_current_starting_index_and_buff_count() const
  {
    CUDF_EXPECTS(current_iteration < num_iterations,
                 "current_iteration cannot exceed num_iterations");
    auto count_for_current = h_num_buffs_per_iteration[current_iteration];
    return {starting_batch, count_for_current};
  }

  /**
   * @brief Advance the iteration state if there are iterations left, updating the
   * starting batch and returning the amount of bytes were copied in the iteration
   * we just finished.
   * @throws cudf::logic_error If the state was at the last iteration before entering
   * this function.
   * @return size in bytes that were copied in the finished iteration
   */
  std::size_t advance_iteration()
  {
    CUDF_EXPECTS(current_iteration < num_iterations,
                 "current_iteration cannot exceed num_iterations");
    std::size_t bytes_copied = h_size_of_buffs_per_iteration[current_iteration];
    starting_batch += h_num_buffs_per_iteration[current_iteration];
    ++current_iteration;
    return bytes_copied;
  }

  /**
   * Returns true if there are iterations left.
   */
  bool has_more_copies() const { return current_iteration < num_iterations; }

  rmm::device_uvector<dst_buf_info> d_batched_dst_buf_info;  ///< dst_buf_info per 1MB batch
  rmm::device_uvector<size_type> const d_batch_offsets;  ///< Offset within a batch per dst_buf_info
  std::size_t const total_size;                          ///< The aggregate size of all iterations
  int const num_iterations;                              ///< The total number of iterations
  int current_iteration;  ///< Marks the current iteration being worked on

 private:
  std::size_t starting_batch;  ///< Starting batch index for the current iteration
  std::vector<std::size_t> const h_num_buffs_per_iteration;  ///< The count of batches per iteration
  std::vector<std::size_t> const
    h_size_of_buffs_per_iteration;  ///< The size in bytes per iteration
};

std::unique_ptr<chunk_iteration_state> chunk_iteration_state::create(
  rmm::device_uvector<thrust::pair<std::size_t, std::size_t>> const& batches,
  int num_bufs,
  dst_buf_info* d_orig_dst_buf_info,
  std::size_t h_dst_buf_total_size,
  std::size_t num_partitions,
  std::size_t user_buffer_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref temp_mr)
{
  rmm::device_uvector<size_type> d_batch_offsets(num_bufs + 1, stream, temp_mr);

  auto const buf_count_iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<std::size_t>(
      [num_bufs, num_batches = num_batches_func{batches.begin()}] __device__(size_type i) {
        return i == num_bufs ? 0 : num_batches(i);
      }));

  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                         buf_count_iter,
                         buf_count_iter + num_bufs + 1,
                         d_batch_offsets.begin(),
                         0);

  auto const num_batches_iter =
    cudf::detail::make_counting_transform_iterator(0, num_batches_func{batches.begin()});
  size_type const num_batches = thrust::reduce(
    rmm::exec_policy(stream, temp_mr), num_batches_iter, num_batches_iter + batches.size());

  auto out_to_in_index = out_to_in_index_function{d_batch_offsets.begin(), num_bufs};

  auto const iter = thrust::make_counting_iterator(0);

  // load up the batches as d_dst_buf_info
  rmm::device_uvector<dst_buf_info> d_batched_dst_buf_info(num_batches, stream, temp_mr);

  thrust::for_each(
    rmm::exec_policy(stream, temp_mr),
    iter,
    iter + num_batches,
    [d_orig_dst_buf_info,
     d_batched_dst_buf_info = d_batched_dst_buf_info.begin(),
     batches                = batches.begin(),
     d_batch_offsets        = d_batch_offsets.begin(),
     out_to_in_index] __device__(size_type i) {
      size_type const in_buf_index = out_to_in_index(i);
      size_type const batch_index  = i - d_batch_offsets[in_buf_index];
      auto const batch_size        = thrust::get<1>(batches[in_buf_index]);
      dst_buf_info const& in       = d_orig_dst_buf_info[in_buf_index];

      // adjust info
      dst_buf_info& out = d_batched_dst_buf_info[i];
      out.element_size  = in.element_size;
      out.value_shift   = in.value_shift;
      out.bit_shift     = in.bit_shift;
      out.valid_count =
        in.valid_count;  // valid count will be set to 1 if this is a validity buffer
      out.src_buf_index = in.src_buf_index;
      // out.dst_buf_index = in.dst_buf_index;

      size_type const elements_per_batch =
        out.element_size == 0 ? 0 : batch_size / out.element_size;
      out.num_elements = ((batch_index + 1) * elements_per_batch) > in.num_elements
                           ? in.num_elements - (batch_index * elements_per_batch)
                           : elements_per_batch;

      size_type const rows_per_batch =
        // if this is a validity buffer, each element is a bitmask_type, which
        // corresponds to 32 rows.
        out.valid_count > 0
          ? elements_per_batch * static_cast<size_type>(cudf::detail::size_in_bits<bitmask_type>())
          : elements_per_batch;
      out.num_rows = ((batch_index + 1) * rows_per_batch) > in.num_rows
                       ? in.num_rows - (batch_index * rows_per_batch)
                       : rows_per_batch;

      out.src_element_index = in.src_element_index + (batch_index * elements_per_batch);
      
      out.dst_offset        = in.dst_offset + (batch_index * batch_size);
      // printf("IDO: %lu %d %d\n", in.dst_offset, (int)batch_index, (int)batch_size);

      // out.bytes and out.buf_size are unneeded here because they are only used to
      // calculate real output buffer sizes. the data we are generating here is
      // purely intermediate for the purposes of doing more uniform copying of data
      // underneath the final structure of the output
    });

  /**
   * In the chunked case, this is the code that fixes up the offsets of each batch
   * and prepares each iteration. Given the batches computed before, it figures
   * out the number of batches that will fit in an iteration of `user_buffer_size`.
   *
   * Specifically, offsets for batches are reset to the 0th byte when a new iteration
   * of `user_buffer_size` bytes is needed.
   */
  if (user_buffer_size != 0) {
    // copy the batch offsets back to host
    std::vector<std::size_t> h_offsets(num_batches + 1);
    {
      rmm::device_uvector<std::size_t> offsets(h_offsets.size(), stream, temp_mr);
      auto const batch_byte_size_iter = cudf::detail::make_counting_transform_iterator(
        0, batch_byte_size_function{num_batches, d_batched_dst_buf_info.begin()});

      thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                             batch_byte_size_iter,
                             batch_byte_size_iter + num_batches + 1,
                             offsets.begin());

      CUDF_CUDA_TRY(cudaMemcpyAsync(h_offsets.data(),
                                    offsets.data(),
                                    sizeof(std::size_t) * offsets.size(),
                                    cudaMemcpyDefault,
                                    stream.value()));

      // the next part is working on the CPU, so we want to synchronize here
      stream.synchronize();
    }

    std::vector<std::size_t> num_batches_per_iteration;
    std::vector<std::size_t> size_of_batches_per_iteration;
    std::vector<std::size_t> accum_size_per_iteration;
    std::size_t accum_size = 0;
    {
      auto current_offset_it = h_offsets.begin();
      // figure out how many iterations we need, while fitting batches to iterations
      // with no more than user_buffer_size bytes worth of batches
      while (current_offset_it != h_offsets.end()) {
        // next_iteration_it points to the batch right above the boundary (the batch
        // that didn't fit).
        auto next_iteration_it =
          std::lower_bound(current_offset_it,
                           h_offsets.end(),
                           // We add the cumulative size + 1 because we want to find what would fit
                           // within a buffer of user_buffer_size (up to user_buffer_size).
                           // Since h_offsets is a prefix scan, we add the size we accumulated so
                           // far so we are looking for the next user_buffer_sized boundary.
                           user_buffer_size + accum_size + 1);

        // we subtract 1 from the number of batch here because next_iteration_it points
        // to the batch that didn't fit, so it's one off.
        auto batches_in_iter = std::distance(current_offset_it, next_iteration_it) - 1;

        // to get the amount of bytes in this iteration we get the prefix scan size
        // and subtract the cumulative size so far, leaving the bytes belonging to this
        // iteration
        auto iter_size_bytes = *(current_offset_it + batches_in_iter) - accum_size;
        accum_size += iter_size_bytes;

        num_batches_per_iteration.push_back(batches_in_iter);
        size_of_batches_per_iteration.push_back(iter_size_bytes);
        accum_size_per_iteration.push_back(accum_size);

        if (next_iteration_it == h_offsets.end()) { break; }

        current_offset_it += batches_in_iter;
      }
    }

    // apply changed offset
    {
      auto d_accum_size_per_iteration =
        cudf::detail::make_device_uvector_async(accum_size_per_iteration, stream, temp_mr);

      // we want to update the offset of batches for every iteration, except the first one (because
      // offsets in the first iteration are all 0 based)
      auto num_batches_in_first_iteration = num_batches_per_iteration[0];
      auto const iter     = thrust::make_counting_iterator(num_batches_in_first_iteration);
      auto num_iterations = accum_size_per_iteration.size();
      thrust::for_each(
        rmm::exec_policy(stream, temp_mr),
        iter,
        iter + num_batches - num_batches_in_first_iteration,
        [num_iterations,
         d_batched_dst_buf_info     = d_batched_dst_buf_info.begin(),
         d_accum_size_per_iteration = d_accum_size_per_iteration.begin()] __device__(size_type i) {
          auto prior_iteration_size =
            thrust::upper_bound(thrust::seq,
                                d_accum_size_per_iteration,
                                d_accum_size_per_iteration + num_iterations,
                                d_batched_dst_buf_info[i].dst_offset) -
            1;
          d_batched_dst_buf_info[i].dst_offset -= *prior_iteration_size;
        });
    }
    return std::make_unique<chunk_iteration_state>(std::move(d_batched_dst_buf_info),
                                                   std::move(d_batch_offsets),
                                                   std::move(num_batches_per_iteration),
                                                   std::move(size_of_batches_per_iteration),
                                                   accum_size);

  } else {
    // we instantiate an "iteration state" for the regular single pass contiguous_split
    // consisting of 1 iteration with all of the batches and totalling `total_size` bytes.
    // auto const total_size = std::reduce(h_buf_sizes, h_buf_sizes + num_partitions);
    auto const total_size = h_dst_buf_total_size;

    // 1 iteration with the whole size
    return std::make_unique<chunk_iteration_state>(
      std::move(d_batched_dst_buf_info),
      std::move(d_batch_offsets),
      std::move(std::vector<std::size_t>{static_cast<std::size_t>(num_batches)}),
      std::move(std::vector<std::size_t>{total_size}),
      total_size);
  }
}

/**
 * @brief Create an instance of `chunk_iteration_state` containing 1MB batches of work
 * that are further grouped into chunks or iterations.
 *
 * This function handles both the `chunked_pack` case: when `user_buffer_size` is non-zero,
 * and the single-shot `contiguous_split` case.
 *
 * @param num_bufs num_src_bufs times the number of partitions
 * @param d_dst_buf_info dst_buf_info per partition produced in `compute_splits`
 * @param h_buf_sizes size in bytes of a partition (accessible from host)
 * @param num_partitions the number of partitions (1 meaning no splits)
 * @param user_buffer_size if non-zero, it is the size in bytes that 1MB batches should be
 *        grouped in, as different iterations.
 * @param stream Optional CUDA stream on which to execute kernels
 * @param temp_mr A memory resource for temporary and scratch space
 *
 * @returns new unique pointer to `chunk_iteration_state`
 */
std::unique_ptr<chunk_iteration_state> compute_batches(int num_bufs,
                                                       dst_buf_info* const d_dst_buf_info,
                                                       std::size_t h_dst_buf_total_size,
                                                       std::size_t num_partitions,
                                                       std::size_t user_buffer_size,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref temp_mr)
{
  // Since we parallelize at one block per copy, performance is vulnerable to situations where we
  // have small numbers of copies to do (a combination of small numbers of splits and/or columns),
  // so we will take the actual set of outgoing source/destination buffers and further partition
  // them into much smaller batches in order to drive up the number of blocks and overall
  // occupancy.
  rmm::device_uvector<thrust::pair<std::size_t, std::size_t>> batches(num_bufs, stream, temp_mr);
  thrust::transform(
    rmm::exec_policy(stream, temp_mr),
    d_dst_buf_info,
    d_dst_buf_info + num_bufs,
    batches.begin(),
    cuda::proclaim_return_type<thrust::pair<std::size_t, std::size_t>>(
      [desired_batch_size = desired_batch_size] __device__(
        dst_buf_info const& buf) -> thrust::pair<std::size_t, std::size_t> {
        // Total bytes for this incoming partition
        std::size_t const bytes =
          static_cast<std::size_t>(buf.num_elements) * static_cast<std::size_t>(buf.element_size);

        // This clause handles nested data types (e.g. list or string) that store no data in the row
        // columns, only in their children.
        if (bytes == 0) { return {1, 0}; }

        // The number of batches we want to subdivide this buffer into
        std::size_t const num_batches = size_to_batch_count(bytes);

        // NOTE: leaving batch size as a separate parameter for future tuning
        // possibilities, even though in the current implementation it will be a
        // constant.
        return {num_batches, desired_batch_size};
      }));

  return chunk_iteration_state::create(batches,
                                       num_bufs,
                                       d_dst_buf_info,
                                       h_dst_buf_total_size,
                                       num_partitions,
                                       user_buffer_size,
                                       stream,
                                       temp_mr);
}

void copy_data(int num_batches_to_copy,
               int starting_batch,
               uint8_t const** d_src_bufs,
               uint8_t** d_dst_buf,
               rmm::device_uvector<dst_buf_info>& d_dst_buf_info,
               uint8_t* user_buffer,
               rmm::cuda_stream_view stream)
{
  constexpr size_type block_size = 256;
  if (user_buffer != nullptr) {
    auto index_to_buffer = [user_buffer] __device__(unsigned int) { return user_buffer; };
    copy_partitions<block_size><<<num_batches_to_copy, block_size, 0, stream.value()>>>(
      index_to_buffer, d_src_bufs, d_dst_buf_info.data() + starting_batch);
  } else {
    // there is only ever 1 destination in the shuffle-split case and all offsets into it are absolute
    auto index_to_buffer = [d_dst_buf/*,
                            dst_buf_info = d_dst_buf_info.data(),
                            user_buffer*/] __device__(unsigned int buf_index) {
      // auto const dst_buf_index = dst_buf_info[buf_index].dst_buf_index;
      return d_dst_buf[0];
    };
    copy_partitions<block_size><<<num_batches_to_copy, block_size, 0, stream.value()>>>(
      index_to_buffer, d_src_bufs, d_dst_buf_info.data() + starting_batch);
  }
}

/**
 * @brief Function that checks an input table_view and splits for specific edge cases.
 *
 * It will return true if the input is "empty" (no rows or columns), which means
 * special handling has to happen in the calling code.
 *
 * @param input table_view of source table to be split
 * @param splits the splits specified by the user, or an empty vector if no splits
 * @returns true if the input is empty, false otherwise
 */
bool check_inputs(cudf::table_view const& input, std::vector<size_type> const& splits)
{
  if (input.num_columns() == 0) { return true; }
  if (splits.size() > 0) {
    CUDF_EXPECTS(splits.back() <= input.column(0).size(),
                 "splits can't exceed size of input columns",
                 std::out_of_range);
  }
  size_type begin = 0;
  for (auto end : splits) {
    CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.", std::out_of_range);
    CUDF_EXPECTS(
      end >= begin, "End index cannot be smaller than the starting index.", std::invalid_argument);
    CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.", std::out_of_range);
    begin = end;
  }
  return input.column(0).size() == 0;
}

constexpr size_type type_to_additional_row_counts(cudf::type_id type)
{
  return type == cudf::type_id::STRING ? 1 : 0;
}

__global__ void pack_per_partition_data_kernel(uint8_t* out_buffer,
                                               size_type num_partitions,
                                               size_t columns_per_partition,
                                               src_buf_info const* src_buf_info,
                                               dst_buf_info const* dst_buf_info,
                                               size_type bufs_per_partition,
                                               size_type const* metadata_col_to_buf_index,
                                               size_t const* out_buffer_offsets,
                                               size_type const *char_count_offsets,
                                               shuffle_split_col_data const* col_data)

{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto const threads_per_partition = cudf::util::round_up_safe(columns_per_partition, static_cast<size_t>(cudf::detail::warp_size));
  auto const partition_index = tid / threads_per_partition;
  if(partition_index >= num_partitions){
    return;
  }  
  auto const col_index = tid % threads_per_partition;

  // start of the metadata buffer for this partition
  uint8_t* buf_start = out_buffer + out_buffer_offsets[partition_index];

  // first thread in each partition stores the partition-level row count
  if(col_index == 0){
    size_type partition_num_rows = 0;
    // it is possible to get in here with no columns -or- no rows.
    if(col_index < columns_per_partition){
      auto const src_buf_index = metadata_col_to_buf_index[col_index];      
      auto const dst_buf_index = (partition_index * bufs_per_partition) + src_buf_index;
      partition_num_rows = col_data[col_index].type == cudf::type_id::STRING ? dst_buf_info[dst_buf_index].root_num_rows : dst_buf_info[dst_buf_index].num_rows;
      // printf("CBI: %d %d %d %d\n", (int)col_index, (int)src_buf_index, (int)dst_buf_index, (int)partition_num_rows);
    }
    reinterpret_cast<size_type*>(buf_start)[0] = partition_num_rows;
  }  

  // store char count for strings
  if(col_index < columns_per_partition && col_data[col_index].type == cudf::type_id::STRING){
    auto const src_buf_index = metadata_col_to_buf_index[col_index];
    auto const dst_buf_index = (partition_index * bufs_per_partition) + src_buf_index;

    // char count for this column
    size_type* char_count = reinterpret_cast<size_type*>(buf_start + (char_count_offsets[col_index] * sizeof(size_type)) + 4);
    char_count[0] = dst_buf_info[dst_buf_index].num_rows;      // # of chars
  }  

  // store has-validity bits
  bitmask_type mask = __ballot_sync(0xffffffff, col_index < columns_per_partition ? src_buf_info[metadata_col_to_buf_index[col_index]].is_validity : 0);
  if((col_index % cudf::detail::warp_size == 0) && col_index < columns_per_partition){
    auto const num_char_counts = char_count_offsets[columns_per_partition];
    bitmask_type* has_validity = reinterpret_cast<bitmask_type*>(buf_start + (num_char_counts * sizeof(size_type)) + 4);
    // printf("HV: %d : %d, %d, %d\n", (int)(col_index / cudf::detail::warp_size), (int)mask, (int)col_index, (int)tid);
    has_validity[col_index / cudf::detail::warp_size] = mask;
  }
}

// the partition header consists of:
// - an array of size_type elements, representing row counts, corresponding to a column in the global metadata.
//   - string columns contain two row counts (the column row count and the number of chars)
//   - all other columns contain 1 row count
//
// - 1 bit per column in the metadata corresponding to whether or not the column contains validity. rounded up
//   to the nearest byte at the last element
//
// - final padding out to 8 bytes (the minimum alignment needed for the re-assembly step on the receiver side)
//
void pack_per_partition_data(cz_metadata_internal const& metadata,
                             rmm::device_buffer& out_buffer,
                             rmm::device_uvector<size_t> const& out_buffer_offsets,
                             src_buf_info const* d_src_buf_info,
                             dst_buf_info const* d_dst_buf_info,
                             int bufs_per_partition,
                             size_type const* d_metadata_col_to_buf_index,
                             rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto const metadata_size = metadata.global_metadata.col_info.size();

  // compute offset for each char count for each string column in the input
  rmm::device_uvector<shuffle_split_col_data> d_col_data = cudf::detail::make_device_uvector_async(metadata.global_metadata.col_info, stream, temp_mr);
  rmm::device_uvector<size_type> char_count_offsets(metadata_size + 1, stream, temp_mr);
  auto char_count_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([d_col_data = d_col_data.begin(), num_cols = d_col_data.size()] __device__ (size_type i) -> size_type {
    return i >= num_cols ? 0 : (d_col_data[i].type == cudf::type_id::STRING ? 1 : 0);
  }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         char_count_iter,
                         char_count_iter + metadata_size + 1,
                         char_count_offsets.begin());
  // print_vector(char_count_offsets);
  
  // pack the row counts and validity info
  auto const num_partitions = out_buffer_offsets.size();
  
  // we want a multiple of full warps per partition
  size_type const thread_count_per_partition = cudf::util::round_up_safe(metadata_size, static_cast<size_t>(cudf::detail::warp_size));
  cudf::detail::grid_1d const grid{thread_count_per_partition * static_cast<size_type>(num_partitions), 128};
  pack_per_partition_data_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(reinterpret_cast<uint8_t*>(out_buffer.data()),
                                                                                                     num_partitions,
                                                                                                     metadata_size,
                                                                                                     d_src_buf_info,
                                                                                                     d_dst_buf_info,
                                                                                                     bufs_per_partition,
                                                                                                     d_metadata_col_to_buf_index,
                                                                                                     out_buffer_offsets.begin(),
                                                                                                     char_count_offsets.begin(),
                                                                                                     d_col_data.begin());

  /*
  {
    stream.synchronize();
    std::vector<uint8_t> h_partitions = cudf::detail::make_std_vector_sync(cudf::device_span<uint8_t const>{reinterpret_cast<uint8_t const*>(out_buffer.data()), out_buffer.size()}, stream);
    std::vector<size_t> h_partition_offsets = cudf::detail::make_std_vector_sync(out_buffer_offsets, stream);
    std::vector<size_type> h_row_count_offsets = cudf::detail::make_std_vector_sync(row_count_offsets, stream);

    for(int p_idx=0; p_idx<num_partitions; p_idx++){
      size_type const* row_counts = reinterpret_cast<size_type const*>(h_partitions.data() + h_partition_offsets[p_idx]);
      bitmask_type const* has_validity = reinterpret_cast<bitmask_type const*>(row_counts + row_count_offsets[metadata_size]);
      for(int idx=0; idx<metadata_size; idx++){
      }
    }
  }
  */

#if 0
  // store the row counts and has-validity
  auto iter = thrust::make_counting_iterator(0);
  auto const num_partitions = out_buffer_offsets.size();
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr),
                   iter,
                   iter + (metadata_size * num_partitions),
                   [num_partitions,
                    metadata_size,
                    d_src_buf_info,
                    d_dst_buf_info,
                    src_bufs_per_partition,
                    d_metadata_col_to_buf_index,
                    out_buffer = reinterpret_cast<uint8_t*>(out_buffer.data()),
                    out_buffer_offsets = out_buffer_offsets.begin(),
                    row_count_offsets = row_count_offsets.begin(),
                    d_col_data = d_col_data.begin()] __device__ (size_type i){

    auto const partition_index = i / num_partitions;
    auto const col_index = i % num_partitions;    
    
    // where we're getting the row counts from
    auto const buf_index = (partition_index * src_bufs_per_partition) + d_metadata_col_to_buf_index[col_index];

    // start of the metadata buffer for this partition
    uint8_t* buf_start = out_buffer + out_buffer_offsets[partition_index];

    // start of row counts for this column
    size_type* row_count = reinterpret_cast<size_type*>(buf_start + (row_count_offsets[col_index] * sizeof(size_type)));
    if(d_col_data[col_index].type == cudf::type_id::STRING){
      row_count[0] = d_dst_buf_info[buf_index].root_num_rows;
      row_count[1] = d_dst_buf_info[buf_index].num_rows;      // # of chars
    } else {
      // all other columns write out just 1 row count. everything else can be reconstructed from there 
      // on the assemble side (eg offsets)
      row_count[0] = d_dst_buf_info[buf_index].num_rows;
    }

    /*
    bitmask_type mask = d_src_buf_info[buf_index].is_validity ? (1 << (col_index % 32)) : 0;
    auto const num_row_counts = row_count_offsets[metadata_size];
    bitmask_type* has_validity = reinterpret_cast<bitmask_type*>(buf_start + (num_row_counts * sizeof(size_type)));
    atomicOr(&has_validity[col_index / 32], mask);
    */
  });

  // compute the has-validity bits. this is being done as a full kernel because:
  // - we don't want to have to initialize the big rmm::device_buffer that contains all the partition data
  // - because of this, the value of all the has-validity bits will be random
  // - if we just used thrust, all we could do is call atomicOr which wouldn't handle the case where 
  //   the uninitialized memory is 1 but we want to set it to 0
#endif
}

};  // anonymous namespace

namespace detail {

/**
 * @brief A helper struct containing the state of contiguous_split, whether the caller
 * is using the single-pass contiguous_split or chunked_pack.
 *
 * It exposes an iterator-like pattern where contiguous_split_state::has_next()
 * returns true when there is work to be done, and false otherwise.
 *
 * contiguous_split_state::contiguous_split() performs a single-pass contiguous_split
 * and is valid iff contiguous_split_state is instantiated with 0 for the user_buffer_size.
 *
 * contiguous_split_state::contiguous_split_chunk(device_span) is only valid when
 * user_buffer_size > 0. It should be called as long as has_next() returns true. The
 * device_span passed to contiguous_split_chunk must be allocated in stream `stream` by
 * the user.
 *
 * None of the methods are thread safe.
 */
struct contiguous_split_state {
  contiguous_split_state(cudf::table_view const& input,
                         std::size_t user_buffer_size,
                         rmm::cuda_stream_view stream,
                         std::optional<rmm::device_async_resource_ref> mr,
                         rmm::device_async_resource_ref temp_mr)
    : contiguous_split_state(input, {}, user_buffer_size, stream, mr, temp_mr)
  {
  }

  contiguous_split_state(cudf::table_view const& input,
                         std::vector<size_type> const& splits,
                         rmm::cuda_stream_view stream,
                         std::optional<rmm::device_async_resource_ref> mr,
                         rmm::device_async_resource_ref temp_mr)
    : contiguous_split_state(input, splits, 0, stream, mr, temp_mr)
  {
  }

  bool has_next() const { return !is_empty && chunk_iter_state->has_more_copies(); }

  std::size_t get_total_contiguous_size() const
  {
    return is_empty ? 0 : chunk_iter_state->total_size;
  }

  std::pair<shuffle_split_result, shuffle_split_metadata> contiguous_split()
  {
    CUDF_EXPECTS(user_buffer_size == 0, "Cannot contiguous split with a user buffer");
    if (is_empty || input.num_columns() == 0) { 
      return {shuffle_split_result{std::make_unique<rmm::device_buffer>(std::move(out_buffer)), std::move(out_buffer_offsets)},
              shuffle_split_metadata{std::move(metadata.global_metadata.col_info)}};
    }

    auto const num_batches_total =
      std::get<1>(chunk_iter_state->get_current_starting_index_and_buff_count());

    // perform the copy.
    copy_data(num_batches_total,
              0 /* starting at buffer for single-shot 0*/,
              src_and_dst_pointers->d_src_bufs,
              src_and_dst_pointers->d_dst_buf,
              chunk_iter_state->d_batched_dst_buf_info,
              nullptr,
              stream);

    // debug
    stream.synchronize();

    // these "orig" dst_buf_info pointers describe the prior-to-batching destination
    // buffers per partition
    auto d_orig_dst_buf_info = partition_buf_size_and_dst_buf_info->d_dst_buf_info;
    auto h_orig_dst_buf_info = partition_buf_size_and_dst_buf_info->h_dst_buf_info;

    // postprocess valid_counts: apply the valid counts computed by copy_data for each
    // batch back to the original dst_buf_infos
    auto const keys = cudf::detail::make_counting_transform_iterator(
      0, out_to_in_index_function{chunk_iter_state->d_batch_offsets.begin(), (int)num_bufs});

    auto values = thrust::make_transform_iterator(
      chunk_iter_state->d_batched_dst_buf_info.begin(),
      cuda::proclaim_return_type<size_type>(
        [] __device__(dst_buf_info const& info) { return info.valid_count; }));

    thrust::reduce_by_key(rmm::exec_policy(stream, temp_mr),
                          keys,
                          keys + num_batches_total,
                          values,
                          thrust::make_discard_iterator(),
                          dst_valid_count_output_iterator{d_orig_dst_buf_info});

    CUDF_CUDA_TRY(cudaMemcpyAsync(h_orig_dst_buf_info,
                                  d_orig_dst_buf_info,
                                  partition_buf_size_and_dst_buf_info->dst_buf_info_size,
                                  cudaMemcpyDefault,
                                  stream.value()));

    stream.synchronize();

    // not necessary for the non-chunked case, but it makes it so further calls to has_next
    // return false, just in case
    chunk_iter_state->advance_iteration();
        
    // std::pair<shuffle_split_result, shuffle_split_metadata>
    return {shuffle_split_result{std::make_unique<rmm::device_buffer>(std::move(out_buffer)), std::move(out_buffer_offsets)},
            shuffle_split_metadata{std::move(metadata.global_metadata.col_info)}};
  }

  /*
  std::unique_ptr<std::vector<uint8_t>> build_packed_column_metadata()
  {
    CUDF_EXPECTS(num_partitions == 1, "build_packed_column_metadata supported only without splits");

    if (input.num_columns() == 0) { return std::unique_ptr<std::vector<uint8_t>>(); }

    if (is_empty) {
      // this is a bit ugly, but it was done to re-use make_empty_packed_table between the
      // regular contiguous_split and chunked_pack cases.
      auto empty_packed_tables = std::move(make_empty_packed_table().front());
      return std::move(empty_packed_tables.data.metadata);
    }

    auto& h_dst_buf_info  = partition_buf_size_and_dst_buf_info->h_dst_buf_info;
    auto cur_dst_buf_info = h_dst_buf_info;
    detail::metadata_builder mb{input.num_columns()};

    populate_metadata(input.begin(), input.end(), cur_dst_buf_info, mb);

    return std::make_unique<std::vector<uint8_t>>(std::move(mb.build()));
  }
  */

 private:
  contiguous_split_state(cudf::table_view const& input,
                         std::vector<size_type> const& splits,
                         std::size_t user_buffer_size,
                         rmm::cuda_stream_view stream,
                         std::optional<rmm::device_async_resource_ref> mr,
                         rmm::device_async_resource_ref temp_mr)
    : input(input),
      user_buffer_size(user_buffer_size),
      stream(stream),
      mr(mr),
      temp_mr(temp_mr),
      is_empty{check_inputs(input, splits)},
      num_partitions{splits.size() + 1},
      num_src_bufs{count_src_bufs(input.begin(), input.end())},
      num_bufs{num_src_bufs * num_partitions}
  {
    // compute metadata, even if the input is empty.
    metadata = compute_metadata(input);

    // if the table we are about to contig split is empty, no additional
    // work is necessary.
    if (is_empty) { return; }

    // debug
    stream.synchronize();

    // First pass over the source tables to generate a `dst_buf_info` per split and column buffer
    // (`num_bufs`). After this, contiguous_split uses `dst_buf_info` to further subdivide the work
    // into 1MB batches in `compute_batches`
    std::tie(partition_buf_size_and_dst_buf_info, partition_split_indices_and_src_buf_info) =
       compute_splits(input, splits, num_partitions, num_src_bufs, num_bufs, metadata, stream, temp_mr);

    // debug
    stream.synchronize();

    // generate output offsets from the partition buf sizes
    out_buffer_offsets = rmm::device_uvector<size_t>(num_partitions, stream, mr.value_or(cudf::get_current_device_resource()));
    auto size_iter = cudf::detail::make_counting_transform_iterator(0, partition_buf_size_func{{partition_buf_size_and_dst_buf_info->d_partition_sizes, num_partitions}});
    thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                           size_iter,
                           size_iter + num_partitions,
                           out_buffer_offsets.begin());

    // debug
    stream.synchronize();

    // one big output buffer
    out_buffer = rmm::device_buffer(partition_buf_size_and_dst_buf_info->h_dst_buf_total_size, stream, mr.value_or(cudf::get_current_device_resource()));

    /*
    void pack_per_partition_data(cz_metadata_internal const& metadata,
                             rmm::device_buffer& out_buffer,
                             rmm::device_uvector<size_t> const& out_buffer_offsets,
                             src_buf_info const* d_src_buf_info,
                             dst_buf_info const* d_dst_buf_info,
                             int src_bufs_per_partition,
                             size_type* d_metadata_col_to_buf_index,
                             rmm::cuda_stream_view stream)
                             */

    // pack the output metadata buffers
    pack_per_partition_data(metadata,
                            out_buffer,
                            out_buffer_offsets,
                            partition_split_indices_and_src_buf_info->d_src_buf_info,
                            partition_buf_size_and_dst_buf_info->d_dst_buf_info,
                            num_src_bufs,
                            partition_split_indices_and_src_buf_info->d_metadata_col_to_buf_index,
                            stream);

    // debug
    stream.synchronize();

    // Second pass: uses `dst_buf_info` to break down the work into 1MB batches.
    chunk_iter_state = compute_batches(num_bufs,
                                       partition_buf_size_and_dst_buf_info->d_dst_buf_info,
                                       partition_buf_size_and_dst_buf_info->h_dst_buf_total_size,
                                       num_partitions,
                                       user_buffer_size,
                                       stream,
                                       temp_mr);
    
    // debug
    stream.synchronize();
    
    CUDF_EXPECTS(user_buffer_size == 0, "Chunked mode not supported yet.");
    // allocate output partition buffers, in the non-chunked case
    /*
    if (user_buffer_size == 0) {
      out_buffers.reserve(num_partitions);
      auto h_buf_sizes = partition_buf_size_and_dst_buf_info->h_buf_sizes;
      std::transform(h_buf_sizes,
                     h_buf_sizes + num_partitions,
                     std::back_inserter(out_buffers),
                     [stream = stream,
                      mr = mr.value_or(rmm::mr::get_current_device_resource())](std::size_t bytes) {
                       return rmm::device_buffer{bytes, stream, mr};
                     });
    }
    */    

    src_and_dst_pointers = std::move(setup_src_and_dst_pointers(
      input, num_partitions, num_src_bufs, out_buffer, stream, temp_mr));
  }

  /*
  std::vector<packed_table> make_packed_tables()
  {
    if (input.num_columns() == 0) { return std::vector<packed_table>(); }
    if (is_empty) { return make_empty_packed_table(); }
    std::vector<packed_table> result;
    result.reserve(num_partitions);
    std::vector<column_view> cols;
    cols.reserve(input.num_columns());

    auto& h_dst_buf_info = partition_buf_size_and_dst_buf_info->h_dst_buf_info;
    auto& h_dst_bufs     = src_and_dst_pointers->h_dst_bufs;

    auto cur_dst_buf_info = h_dst_buf_info;
    detail::metadata_builder mb(input.num_columns());

    for (std::size_t idx = 0; idx < num_partitions; idx++) {
      // traverse the buffers and build the columns.
      cur_dst_buf_info = build_output_columns(input.begin(),
                                              input.end(),
                                              cur_dst_buf_info,
                                              std::back_inserter(cols),
                                              h_dst_bufs[idx],
                                              mb);

      // pack the columns
      result.emplace_back(packed_table{
        cudf::table_view{cols},
        packed_columns{std::make_unique<std::vector<uint8_t>>(mb.build()),
                       std::make_unique<rmm::device_buffer>(std::move(out_buffers[idx]))}});

      cols.clear();
      mb.clear();
    }

    return result;
  }
  */

 /*
  std::vector<packed_table> make_empty_packed_table()
  {
    // sanitize the inputs (to handle corner cases like sliced tables)
    std::vector<cudf::column_view> empty_column_views;
    empty_column_views.reserve(input.num_columns());
    std::transform(input.begin(),
                   input.end(),
                   std::back_inserter(empty_column_views),
                   [](column_view const& col) { return cudf::empty_like(col)->view(); });

    table_view empty_inputs(empty_column_views);

    // build the empty results
    std::vector<packed_table> result;
    result.reserve(num_partitions);
    auto const iter = thrust::make_counting_iterator(0);
    std::transform(iter,
                   iter + num_partitions,
                   std::back_inserter(result),
                   [&empty_inputs](int partition_index) {
                     return packed_table{empty_inputs,
                                         packed_columns{std::make_unique<std::vector<uint8_t>>(
                                                          pack_metadata(empty_inputs, nullptr, 0)),
                                                        std::make_unique<rmm::device_buffer>()}};
                   });

    return result;
  }
  */

  cudf::table_view const input;        ///< The input table_view to operate on
  std::size_t const user_buffer_size;  ///< The size of the user buffer for the chunked_pack case
  rmm::cuda_stream_view const stream;
  std::optional<rmm::device_async_resource_ref const> mr;  ///< The resource for any data returned

  // this resource defaults to `mr` for the contiguous_split case, but it can be useful for the
  // `chunked_pack` case to allocate scratch/temp memory in a pool
  rmm::device_async_resource_ref const temp_mr;  ///< The memory resource for scratch/temp space

  // whether the table was empty to begin with (0 rows or 0 columns) and should be metadata-only
  bool const is_empty;  ///< True if the source table has 0 rows or 0 columns

  // This can be 1 if `contiguous_split` is just packing and not splitting
  std::size_t const num_partitions;  ///< The number of partitions to produce

  size_type const num_src_bufs;  ///< Number of source buffers including children

  std::size_t const num_bufs;  ///< Number of source buffers including children * number of splits

  std::unique_ptr<packed_partition_buf_size_and_dst_buf_info>
    partition_buf_size_and_dst_buf_info;  ///< Per-partition buffer size and destination buffer info

  std::unique_ptr<packed_split_indices_and_src_buf_info>
    partition_split_indices_and_src_buf_info;  ///< Per-partition buffer size and destination buffer info

  std::unique_ptr<packed_src_and_dst_pointers>
    src_and_dst_pointers;  ///< Src. and dst. pointers for `copy_partition`

  //
  // State around the chunked pattern
  //

  // chunked_pack will have 1 or more "chunks" to iterate on, defined in chunk_iter_state
  // contiguous_split will have a single "chunk" in chunk_iter_state, so no iteration.
  std::unique_ptr<chunk_iteration_state>
    chunk_iter_state;  ///< State object for chunk iteration state

  // Two API usages are allowed:
  //  - `chunked_pack`: for this mode, the user will provide a buffer that must be at least 1MB.
  //    The behavior is "chunked" in that it will contiguously copy up until the user specified
  //    `user_buffer_size` limit, exposing a next() call for the user to invoke. Note that in this
  //    mode, no partitioning occurs, hence the name "pack".
  //
  //  - `contiguous_split` (default): when the user doesn't provide their own buffer,
  //    `contiguous_split` will allocate a buffer per partition and will place contiguous results in
  //    each buffer.
  //
  //std::vector<rmm::device_buffer>
//    out_buffers;  ///< Buffers allocated for a regular `contiguous_split`
  rmm::device_buffer              out_buffer{};
  rmm::device_uvector<size_t>     out_buffer_offsets{0, cudf::get_default_stream()};
  cz_metadata_internal metadata;  
};

};  // namespace detail

std::pair<shuffle_split_result, shuffle_split_metadata> shuffle_split(cudf::table_view const& input,
                                                                      std::vector<size_type> const& splits,
                                                                      rmm::cuda_stream_view stream,
                                                                      rmm::device_async_resource_ref mr)
{
  // for now, we don't allow strings, lists or columns with validity
  CUDF_EXPECTS(std::all_of(input.begin(), input.end(), [](cudf::column_view const& col){
    return col.type().id() != cudf::type_id::STRING && 
           col.type().id() != cudf::type_id::LIST &&
           !col.nullable();
  }), "Unsupported column type (for now)");

  // `temp_mr` is the same as `mr` for contiguous_split as it allocates all
  // of its memory from the default memory resource in cuDF
  auto temp_mr = mr;
  auto state   = detail::contiguous_split_state(input, splits, stream, mr, temp_mr);
  return state.contiguous_split();
}

namespace detail {

#define OUTPUT_ITERATOR(__name, __T, __field_name)                                                  \
  template<typename __T>                                                                            \
  struct __name##generic_output_iter {                                                              \
    __T* c;                                                                                         \
    using value_type        = decltype(__T::__field_name);                                          \
    using difference_type   = size_t;                                                               \
    using pointer           = decltype(__T::__field_name)*;                                         \
    using reference         = decltype(__T::__field_name)&;                                         \
    using iterator_category = thrust::output_device_iterator_tag;                                   \
                                                                                                    \
    __name##generic_output_iter operator+ __host__ __device__(int i) { return {c + i}; }            \
                                                                                                    \
    __name##generic_output_iter& operator++ __host__ __device__()                                   \
    {                                                                                               \
      c++;                                                                                          \
      return *this;                                                                                 \
    }                                                                                               \
                                                                                                    \
    reference operator[] __device__(int i) { return dereference(c + i); }                           \
    reference operator* __device__() { return dereference(c); }                                     \
                                                                                                    \
  private:                                                                                          \
    reference __device__ dereference(__T* c) { return c->__field_name; }                            \
  };                                                                                                \
  using __name = __name##generic_output_iter<__T>

// per-flattened-column information
struct assemble_column_info {
  cudf::type_id         type;
  bool                  has_validity;
  size_type             num_rows, num_chars;
  size_type             null_count;
  size_type             num_children;
};
OUTPUT_ITERATOR(assemble_column_info_num_rows_output_iter, assemble_column_info, num_rows);
OUTPUT_ITERATOR(assemble_column_info_has_validity_output_iter, assemble_column_info, has_validity);

// a copy batch. 1 per block.
struct assemble_batch {
  __device__ assemble_batch(int8_t const* _src, int8_t* _dst, size_t _size, bool _validity, int _value_shift, int _bit_shift):
    src(_src), dst(_dst), size(_size), validity(_validity), value_shift(_value_shift), bit_shift(_bit_shift){}

  int8_t const* src;
  int8_t* dst;
  size_t              size;
  bool                validity; // whether or not this is a validity buffer
  int value_shift;              // amount to shift values down by (for offset buffers)
  int bit_shift;                // # of bits to shift right by (for validity buffers)
  size_type valid_count = 0;    // (output) validity count for this block of work
};

struct assemble_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    auto const data = col->has_validity ? buffer + 1 : buffer;
    buffer = data + 1;

    out.push_back(std::make_unique<cudf::column>(cudf::data_type{col->type},
                  col->num_rows,
                  std::move(*data),
                  col->has_validity ? std::move(*validity) : rmm::device_buffer{},
                  col->null_count));
    
    return {col + 1, buffer};
  }
  
  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    buffer = col->has_validity ? buffer + 1 : buffer;

    // build children
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(col->num_children);
    auto next = col + 1;
    for(size_type i=0; i<col->num_children; i++){
      std::tie(next, buffer) = cudf::type_dispatcher(cudf::data_type{next->type},
                                                     detail::assemble_column_functor{stream, mr},
                                                     next,
                                                     buffer,
                                                     children);
    }    

    out.push_back(cudf::make_structs_column(col->num_rows,
                                            std::move(children),
                                            col->null_count,
                                            col->has_validity ? std::move(*validity) : rmm::device_buffer{},
                                            stream,
                                            mr));
    return {next, buffer};
  }
    
    /*
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  size_t operator()(size_t cur, host_span<assemble_column_info const> assemble_data, host_span<rmm::device_buffer> buffers, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto col = assemble_data[cur];
    auto validity = cur;
    auto offsets = col.has_validity ? cur + 1 : cur;
    cur = offsets + 1;

    // build offsets
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      col.num_rows + 1,
                                                      std::move(buffers[offsets]),
                                                      rmm::device_buffer{},
                                                      0);

    // build the child
    std::vector<std::unique_ptr<cudf::column>> child_col;
    cur = cudf::type_dispatcher(cudf::data_type{col.type},
                                *this,
                                cur,
                                assemble_data,
                                buffers,
                                child_col);
    
    // build the final column
    out.push_back(cudf::make_lists_column(col.num_rows,
                                          std::move(offsets_col),
                                          std::move(child_col.back()),
                                          col.null_count,
                                          col.has_validity ? std::move(buffers[validity]) : rmm::device_buffer{},
                                          stream,
                                          mr));
    return cur;
  }  
  */

  // template <typename T, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::list_view> and !std::is_same_v<T, cudf::struct_view>)>
  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::struct_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    CUDF_FAIL("Unsupported type in shuffle_assemble");
  }
};

struct assemble_buffer_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col, std::vector<rmm::device_buffer>& out)
  {
    // validity
    if(col.has_validity){
      out.push_back(alloc_validity(col.num_rows));
    }

    // data
    auto const data_size = cudf::util::round_up_safe(cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows, shuffle_split_partition_data_align);
    out.push_back(rmm::device_buffer(data_size, stream, mr));
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  void operator()(assemble_column_info const& col, std::vector<rmm::device_buffer>& out)
  { 
    // validity
    if(col.has_validity){
      out.push_back(alloc_validity(col.num_rows));
    }

    // offsets
    auto const offsets_size = cudf::util::round_up_safe(sizeof(size_type) * (col.num_rows + 1), shuffle_split_partition_data_align);
    out.push_back(rmm::device_buffer(offsets_size, stream, mr));
  } 

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  void operator()(assemble_column_info const& col, std::vector<rmm::device_buffer>& out)
  { 
    // validity
    if(col.has_validity){
      out.push_back(alloc_validity(col.num_rows));
    }    
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  void operator()(assemble_column_info const& col, std::vector<rmm::device_buffer>& out)
  { 
    // validity
    if(col.has_validity){
      out.push_back(alloc_validity(col.num_rows));
    }

    // chars
    auto const chars_size = cudf::util::round_up_safe(sizeof(int8_t) * (col.num_chars + 1), shuffle_split_partition_data_align);
    out.push_back(rmm::device_buffer(chars_size, stream, mr));

    // offsets
    auto const offsets_size = cudf::util::round_up_safe(sizeof(size_type) * (col.num_rows + 1), shuffle_split_partition_data_align);
    out.push_back(rmm::device_buffer(offsets_size, stream, mr));
  }

  template <typename T, CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> && 
                                       !std::is_same_v<T, cudf::list_view> && 
                                       !std::is_same_v<T, cudf::string_view> && 
                                       !cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col, std::vector<rmm::device_buffer>& out)
  { 
    CUDF_FAIL("Unsupported type in assemble_buffer_functor");
  }
 
private:
  rmm::device_buffer alloc_validity(size_type num_rows)
  {
    return rmm::device_buffer(bitmask_allocation_size_bytes(num_rows, shuffle_split_partition_data_align), stream, mr);
  }
};

// Computes required allocation size of a bitmask
__device__ std::size_t device_bitmask_allocation_size_bytes(size_type number_of_bits, std::size_t padding_boundary)
{
  auto necessary_bytes = cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes = padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                                           necessary_bytes, padding_boundary);
  return padded_bytes;
}

// Important: this returns the size of the buffer -without- padding. just the size of
// the raw bytes containing the actual data.
struct assemble_buffer_size_functor {
  template <typename T, typename OutputIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter out)
  {
    // validity
    if(col.has_validity){
      *out++ = device_bitmask_allocation_size_bytes(col.num_rows, 1);
    }

    // data
    *out++ = cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows;
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter out)
  { 
    // validity
    if(col.has_validity){
      *out++ = device_bitmask_allocation_size_bytes(col.num_rows, 1);
    }

    // offsets
    *out++ = sizeof(size_type) * (col.num_rows + 1);
  } 

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter out)
  { 
    // validity
    if(col.has_validity){
      *out++ = device_bitmask_allocation_size_bytes(col.num_rows, 1);
    }
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter out)
  { 
    // validity
    if(col.has_validity){
      *out++ = device_bitmask_allocation_size_bytes(col.num_rows, 1);
    }

    // chars
    *out++ = sizeof(int8_t) * (col.num_chars + 1);

    // offsets
    *out++ = sizeof(size_type) * (col.num_rows + 1);
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> && 
                                       !std::is_same_v<T, cudf::list_view> && 
                                       !std::is_same_v<T, cudf::string_view> && 
                                       !cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter out)
  {
  }
};

struct assemble_metadata_offset_functor {
  template <typename T, typename OutputIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter out, size_t offset)
  {
    // validity
    if(col.has_validity){
      *out += offset;
      *out++;
    }

    // data
    *out += offset;
    out++;
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter out, size_t offset)
  { 
    // validity
    if(col.has_validity){
      *out += offset;
      *out++;
    }

    // offsets
    (*out++) += offset;
    *out++;
  } 

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter out, size_t offset)
  { 
    // validity
    if(col.has_validity){
      *out += offset;
      *out++;
    }
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter out, size_t offset)
  { 
    // validity
    if(col.has_validity){
      *out += offset;
      *out++;
    }

    // chars
    *out += offset;
    *out++;

    // offsets
    *out += offset;
    *out++;
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> && 
                                       !std::is_same_v<T, cudf::list_view> && 
                                       !std::is_same_v<T, cudf::string_view> && 
                                       !cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter out, size_t offset)
  {
  }
};

// returns:
// - a vector of assemble_column_info structs representing the destination column data.
//   the vector is of length global_metadata.col_info.size()  that is, the flattened list of columns in the table.
//
// - the same vector as above, but in host memory. 
//
// - a vector of assemble_column_info structs, representing the source column data.
//   the vector is of length global_metadata.col_info.size() * the # of partitions. 
//
std::tuple<rmm::device_uvector<assemble_column_info>,
           std::vector<assemble_column_info>,
           rmm::device_uvector<assemble_column_info>,
           size_t>
assemble_build_column_info(shuffle_split_metadata const& h_global_metadata,
                           cudf::device_span<int8_t const> partitions, 
                           cudf::device_span<size_t const> partition_offsets,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<shuffle_split_col_data> global_metadata = cudf::detail::make_device_uvector_async(h_global_metadata.col_info, stream, temp_mr);

  // "columns" here means the number of flattened columns in the entire source table, not just the
  // number of columns at the top level
  auto const num_columns = global_metadata.size();
  size_type const num_partitions = partition_offsets.size();
  auto const num_column_instances = num_columns * num_partitions;

  // generate per-column data ------------------------------------------------------
  rmm::device_uvector<assemble_column_info> column_info(num_columns, stream, temp_mr);  

  // compute:
  //  - indices into the char count data for string columns
  //  - offset into the partition data where has-validity begins
  rmm::device_uvector<size_type> char_count_indices(num_columns + 1, stream, temp_mr);
  auto cc_index_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([global_metadata = global_metadata.begin(), num_columns] __device__ (size_type i) {
    return i >= num_columns ? 0 : (global_metadata[i].type == cudf::type_id::STRING ? 1 : 0);
  }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr), cc_index_iter, cc_index_iter + num_columns + 1, char_count_indices.begin());
  size_type const per_partition_num_char_counts = char_count_indices.back_element(stream);
  // the +1 is for the per-partition overall row count at the very beginning
  auto const has_validity_offset = (per_partition_num_char_counts + 1) * sizeof(size_type);

  /*  
  {
    auto h_char_count_indices = cudf::detail::make_std_vector_sync(char_count_indices, stream);
    printf("per_partition_num_char_counts : %d\n", per_partition_num_char_counts);
    printf("has_validity_offset : %lu\n", has_validity_offset);
    for(size_t idx=0; idx<h_char_count_indices.size(); idx++){
      printf("h_char_count_indices(%lu): %d\n", idx, h_char_count_indices[idx]);
    }
  } 
  */ 

  // compute has-validity
  // note that we are iterating vertically -> horizontally here, with each column's individual piece per partition first.
  auto column_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([num_partitions] __device__ (size_type i){
    return i / num_partitions;
  }));  
  auto has_validity_values = cudf::detail::make_counting_transform_iterator(0, 
    cuda::proclaim_return_type<bool>([num_partitions,
                                      has_validity_offset,
                                      partitions = partitions.data(),
                                      partition_offsets = partition_offsets.begin()]
                                      __device__ (int i) -> bool {
      auto const partition_index = i % num_partitions;
      bitmask_type const*const has_validity_buf = reinterpret_cast<bitmask_type const*>(partitions + partition_offsets[partition_index] + has_validity_offset);
      auto const col_index = i / num_partitions;
      // printf("HVV: %d, %d, %d, %d, %d\n", (int)partition_index, (int)partition_offsets[partition_index], (int)has_validity_offset, (int)col_index, (int)has_validity_buf[col_index / 32]);
      return has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
    })
  );
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        column_keys,
                        column_keys +  num_column_instances,
                        has_validity_values,
                        thrust::make_discard_iterator(),
                        assemble_column_info_has_validity_output_iter{column_info.begin()},
                        thrust::equal_to<size_type>{},
                        thrust::logical_or<bool>{});
  /*
  {
    auto h_column_info = cudf::detail::make_std_vector_sync(column_info, stream);
    for(size_t idx=0; idx<h_column_info.size(); idx++){
      printf("h_column_info(%lu): has_validity = %d\n", idx, (int)(h_column_info[idx].has_validity ? 1 : 0));
    }
  }
  */

  //print_span(cudf::device_span<size_t const>(partition_offsets));

  // compute overall row count
  auto row_count_values = cudf::detail::make_counting_transform_iterator(0,
    cuda::proclaim_return_type<cudf::size_type>([num_partitions,
                                                 partitions = partitions.data(),
                                                 partition_offsets = partition_offsets.begin()]
                                                 __device__ (int i){
                                                  return reinterpret_cast<size_type const*>(partitions + partition_offsets[i])[0];
                                                 }));
  size_t const row_count =  thrust::reduce(rmm::exec_policy_nosync(stream, temp_mr),
                                            row_count_values,
                                            row_count_values + num_partitions);
  
  // compute char counts for strings
  // note that we are iterating vertically -> horizontally here, with each column's individual piece per partition first.
  // TODO: use an output iterator and write directly to the outgoing assembly_info structs
  auto cc_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<cudf::size_type>([num_partitions] __device__ (int i){
    return i / num_partitions;
  }));
  auto char_count_values = cudf::detail::make_counting_transform_iterator(0,
    cuda::proclaim_return_type<cudf::size_type>([num_partitions,
                                                 partitions = partitions.data(),
                                                 partition_offsets = partition_offsets.begin(),
                                                 global_metadata = global_metadata.begin()]
                                                 __device__ (int i){
      auto const partition_index = i % num_partitions;
      auto const col_index = i / num_partitions;

      // non-string columns don't have a char count
      auto const column_type = global_metadata[col_index].type;
      if(column_type != cudf::type_id::STRING){
        return 0;
      }

      // string columns
      size_type const*const char_counts = reinterpret_cast<size_type const*>(partitions + partition_offsets[partition_index] + 4);
      // printf("RCI %d : %d, partition_index = %d\n", (int)col_index, char_counts[col_index], (int)partition_index);
      return char_counts[col_index];
    })
  );
  rmm::device_uvector<size_type> char_counts(num_columns, stream, temp_mr);
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        cc_keys, 
                        cc_keys + num_column_instances,
                        char_count_values,
                        thrust::make_discard_iterator(),
                        char_counts.begin());
  // print_span(static_cast<cudf::device_span<size_type const>>(char_counts));
  
  // copy type and summed row counts
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr), iter, iter + num_columns, [row_count,
                                                                                        column_info = column_info.begin(),
                                                                                        global_metadata = global_metadata.begin(),
                                                                                        char_count_indices = char_count_indices.begin(),
                                                                                        char_counts = char_counts.begin()]
                                                                                        __device__ (size_type col_index){
    auto const& metadata = global_metadata[col_index];
    auto& cinfo = column_info[col_index];
    
    cinfo.type = metadata.type;
    cinfo.null_count = 0; // TODO
    cinfo.num_children = metadata.num_children;
    
    cinfo.num_rows = row_count;
    
    // string columns store the char count separately
    cinfo.num_chars = cinfo.type == cudf::type_id::STRING ? char_counts[char_count_indices[col_index]] : 0;
  });
  
  /*
  {
    auto h_column_info = cudf::detail::make_std_vector_sync(column_info, stream);
    for(size_t idx=0; idx<h_column_info.size(); idx++){
      printf("col_info[%lu]: type = %d has_validity = %d num_rows = %d num_chars = %d null_count = %d\n", idx,
        (int)h_column_info[idx].type, h_column_info[idx].has_validity ? 1 : 0, h_column_info[idx].num_rows, h_column_info[idx].num_chars, h_column_info[idx].null_count);
    }
  } 
  */ 

  // generate per-column-instance data ------------------------------------------------------

  // has-validity, type, row count
  rmm::device_uvector<assemble_column_info> column_instance_info(num_column_instances, stream, temp_mr);
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr), iter, iter + num_column_instances, [char_count_indices = char_count_indices.begin(),
                                                                                                 column_instance_info = column_instance_info.begin(),
                                                                                                 global_metadata = global_metadata.begin(),
                                                                                                 partitions = partitions.data(),
                                                                                                 partition_offsets = partition_offsets.begin(),
                                                                                                 num_columns,
                                                                                                 has_validity_offset]
                                                                                                 __device__ (size_type i){
    auto const partition_index = i / num_columns;
    auto const col_index = i % num_columns;
    auto const col_instance_index = (partition_index * num_columns) + col_index;

    auto const& metadata = global_metadata[col_index];
    auto& cinstance_info = column_instance_info[col_instance_index];

    uint8_t const*const pheader = reinterpret_cast<uint8_t const*>(partitions + partition_offsets[partition_index]);

    bitmask_type const*const has_validity_buf = reinterpret_cast<bitmask_type const*>(pheader + has_validity_offset);
    cinstance_info.has_validity = has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
    
    cinstance_info.type = metadata.type;
    cinstance_info.null_count = 0; // TODO
    cinstance_info.num_children = metadata.num_children;
    
    cinstance_info.num_rows = reinterpret_cast<size_type const*>(pheader)[0];
    
    // string columns store the char count separately
    if(metadata.type == cudf::type_id::STRING){
      size_type const*const char_counts = reinterpret_cast<size_type const*>(pheader + 4);
      cinstance_info.num_chars = char_counts[char_count_indices[col_index]];
    }
  });
  
  /*
  {
    auto h_column_instance_info = cudf::detail::make_std_vector_sync(column_instance_info, stream);
    for(size_t idx=0; idx<h_column_instance_info.size(); idx++){
      size_type const partition_index = idx / num_columns;
      size_type const col_index = idx % num_columns;
      size_type const col_instance_index = (partition_index * num_columns) + col_index;

      printf("col_info[%d, %d, %d]: type = %d has_validity = %d num_rows = %d num_chars = %d null_count = %d\n",
        partition_index, col_index, col_instance_index,
        (int)h_column_instance_info[idx].type, h_column_instance_info[idx].has_validity ? 1 : 0, h_column_instance_info[idx].num_rows, h_column_instance_info[idx].num_chars, h_column_instance_info[idx].null_count);
    }
  } 
  */  

  // compute per-partition metadata size
  size_t const metadata_rc_size = ((per_partition_num_char_counts + 1) * sizeof(size_type));
  size_t const metadata_has_validity_size = (cudf::util::round_up_safe(num_columns, size_t{32}) / size_t{32}) * sizeof(bitmask_type);
  size_t const per_partition_metadata_size = cudf::util::round_up_safe(metadata_rc_size + metadata_has_validity_size, shuffle_split_partition_data_align);

  return {std::move(column_info), cudf::detail::make_std_vector_sync(column_info, stream), std::move(column_instance_info), per_partition_metadata_size};
}

template<typename SizeIterator, typename GroupFunction>
rmm::device_uvector<std::invoke_result_t<GroupFunction>> transform_expand(SizeIterator first,
                                                                          SizeIterator last,
                                                                          GroupFunction op,
                                                                          rmm::cuda_stream_view stream,
                                                                          rmm::device_async_resource_ref mr)
{ 
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto value_count = std::distance(first, last);
  auto size_wrapper = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([value_count, first] __device__ (size_t i){
    return i >= value_count ? 0 : first[i];
  }));
  rmm::device_uvector<size_t> group_offsets(value_count + 1, stream, temp_mr);
  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                         size_wrapper,
                         size_wrapper + group_offsets.size(),
                         group_offsets.begin());
  size_t total_size = group_offsets.back_element(stream); // note memcpy and device sync
  
  using OutputType = std::invoke_result_t<GroupFunction>;
  rmm::device_uvector<OutputType> result(total_size, stream, mr);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + total_size,
                    result.begin(),
                    cuda::proclaim_return_type<OutputType>([op, group_offsets_begin = group_offsets.begin(), group_offsets_end = group_offsets.end()] __device__ (size_t i){
                      auto const group_index = thrust::lower_bound(thrust::seq, group_offsets_begin, group_offsets_end, i) - group_offsets_begin;
                      auto const intra_group_index = i - group_offsets_begin[group_index];
                      return op(group_index, intra_group_index);
                    }));

  return result;
};

// returns destination buffers
std::pair<std::vector<rmm::device_buffer>, rmm::device_uvector<assemble_batch>> assemble_build_buffers(rmm::device_uvector<assemble_column_info> const& column_info,
                                                                                                       rmm::device_uvector<assemble_column_info> const& column_instance_info,
                                                                                                       cudf::device_span<int8_t const> partitions,
                                                                                                       size_t num_partitions,
                                                                                                       size_t per_partition_metadata_size,
                                                                                                       rmm::cuda_stream_view stream,
                                                                                                       rmm::device_async_resource_ref mr)
{
  auto h_column_info = cudf::detail::make_std_vector_sync(column_info, stream);
  auto temp_mr = cudf::get_current_device_resource_ref();  
  
  // allocate output buffers ----------------------------------
  std::vector<rmm::device_buffer> assemble_buffers;
  assemble_buffers.reserve(h_column_info.size() * 3); // worst case, every column has 3 buffers.
  // mapping of column index to first-buffer index
  std::vector<size_type> h_column_to_buffer_map(h_column_info.size());  
  for(size_t idx=0; idx<h_column_info.size(); idx++){
    h_column_to_buffer_map[idx] = assemble_buffers.size();
    cudf::type_dispatcher(cudf::data_type{h_column_info[idx].type},
                          detail::assemble_buffer_functor{stream, mr},
                          h_column_info[idx],
                          assemble_buffers);

  }
  std::vector<int8_t*> h_dst_buffers(assemble_buffers.size());
  std::transform(assemble_buffers.begin(), assemble_buffers.end(), h_dst_buffers.begin(), [](rmm::device_buffer& buf){
    return reinterpret_cast<int8_t*>(buf.data());
  });
  auto dst_buffers = cudf::detail::make_device_uvector_async(h_dst_buffers, stream, temp_mr);
  auto column_to_buffer_map = cudf::detail::make_device_uvector_async(h_column_to_buffer_map, stream, cudf::get_current_device_resource_ref());
  // print_span(cudf::device_span<size_type const>{column_to_buffer_map});

  // generate copy batches ------------------------------------

  // compute:
  // - unpadded sizes of the source buffers
  // - offsets into the partition data where each source buffer starts
  // - offsets into the destination buffers where each source buffer starts writing
  size_t const buffers_per_partition = assemble_buffers.size();
  size_t const num_src_buffers = buffers_per_partition * num_partitions;
  rmm::device_uvector<size_t> src_sizes_unpadded(num_src_buffers, stream, mr);
  rmm::device_uvector<size_t> src_offsets(num_src_buffers, stream, mr);
  rmm::device_uvector<size_t> dst_offsets(num_src_buffers, stream, mr);
  {
    // generate unpadded sizes of the source buffers
    auto const num_column_instances = column_instance_info.size();
    auto iter = thrust::make_counting_iterator(0);
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + num_column_instances,
                    [buffers_per_partition,
                     num_columns = column_info.size(),
                     column_to_buffer_map = column_to_buffer_map.begin(),
                     column_instance_info = column_instance_info.begin(),
                     src_sizes_unpadded = src_sizes_unpadded.begin()] __device__ (size_type i){

      auto const partition_index = i / num_columns;
      auto const col_index = i % num_columns;
      auto const col_instance_index = (partition_index * num_columns) + col_index;

      auto const& cinfo_instance = column_instance_info[col_instance_index];
      auto const buf_index = column_to_buffer_map[col_index] + (partition_index * buffers_per_partition);
      cudf::type_dispatcher(cudf::data_type{cinfo_instance.type},
                            detail::assemble_buffer_size_functor{},
                            cinfo_instance,
                            &src_sizes_unpadded[buf_index]);
      // printf("SSU: %d %d (%d %d), %lu\n", (int)partition_index, (int)buf_index, (int)col_index, (int)column_to_buffer_map[col_index], src_sizes_unpadded[buf_index]);
    });
    // print_span(cudf::device_span<size_t const>{src_sizes_unpadded});
    
    // scan to source offsets. include padding for the buffers themselves
    auto padded_sizes = thrust::make_transform_iterator(src_sizes_unpadded.begin(), cuda::proclaim_return_type<size_t>([shuffle_split_partition_data_align = shuffle_split_partition_data_align] __device__ (size_t size_unpadded){
      return cudf::util::round_up_safe(size_unpadded, shuffle_split_partition_data_align);
    }));
    thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                           padded_sizes,
                           padded_sizes + src_sizes_unpadded.size(),
                           src_offsets.begin());
    // print_span(cudf::device_span<size_t const>{src_offsets});
    
    // add metadata header offset
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + num_column_instances,
                    [num_columns = column_info.size(),
                     buffers_per_partition,
                     column_to_buffer_map = column_to_buffer_map.begin(),
                     column_instance_info = column_instance_info.begin(),
                     src_offsets = src_offsets.begin(),
                     per_partition_metadata_size] __device__ (size_type i){

      auto const partition_index = i / num_columns;
      auto const metadata_offset = (partition_index + 1) * per_partition_metadata_size;
      auto const col_index = i % num_columns;
      auto const col_instance_index = (partition_index * num_columns) + col_index;
      auto const& cinfo_instance = column_instance_info[col_instance_index];
      auto const buf_index = column_to_buffer_map[col_index] + (partition_index * buffers_per_partition);

      // printf("MHO: %d, partition_index = %d, metadata_offset = %d, col_index = %d, col_instance_index = %d, buf_index = %d\n", i, (int)partition_index, (int)metadata_offset, (int)col_index, (int)col_instance_index, (int)buf_index);
      cudf::type_dispatcher(cudf::data_type{cinfo_instance.type},
                            detail::assemble_metadata_offset_functor{},
                            cinfo_instance,
                            &src_offsets[buf_index],
                            metadata_offset);
    });
    // print_span(cudf::device_span<size_t const>{src_offsets});

    // generate destination buffer offsets
    // Note: vertical iteration
    auto dst_buf_key = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([num_partitions] __device__ (size_t i){
      return i / num_partitions;
    }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([src_sizes_unpadded = src_sizes_unpadded.begin(), num_partitions, buffers_per_partition] __device__ (size_t i){
      auto const dst_buf_index = i / num_partitions;
      auto const partition_index = i % num_partitions;
      auto const src_buf_index = (partition_index * buffers_per_partition) + dst_buf_index;
      return src_sizes_unpadded[src_buf_index];
    }));
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                  dst_buf_key,
                                  dst_buf_key + num_src_buffers,
                                  size_iter,
                                  dst_offsets.begin());
    // print_span(cudf::device_span<size_t const>{dst_offsets});
  }

  // generate batches
  auto batch_count_iter = cudf::detail::make_counting_transform_iterator(0, 
                                                                         cuda::proclaim_return_type<size_t>([src_sizes_unpadded = src_sizes_unpadded.begin()] __device__ (size_t i){
                                                                           return size_to_batch_count(src_sizes_unpadded[i]);
                                                                         }));
  auto copy_batches = transform_expand(batch_count_iter, 
                                       batch_count_iter + src_sizes_unpadded.size(),
                                       cuda::proclaim_return_type<assemble_batch>([dst_buffers = dst_buffers.begin(),
                                                                                   dst_offsets = dst_offsets.begin(),
                                                                                   partitions = partitions.data(),
                                                                                   buffers_per_partition,
                                                                                   num_partitions,
                                                                                   src_sizes_unpadded = src_sizes_unpadded.begin(),
                                                                                   src_offsets = src_offsets.begin(),
                                                                                   desired_batch_size = desired_batch_size] __device__ (size_t src_buf_index, size_t batch_index){
                                         auto const batch_offset = batch_index * desired_batch_size;
                                         auto const partition_index = src_buf_index / buffers_per_partition;
                                         
                                         auto const src_offset = src_offsets[src_buf_index];
                                        
                                         auto const dst_buf_index = src_buf_index % buffers_per_partition;
                                         auto const dst_offset_index = (dst_buf_index * num_partitions) + partition_index;
                                         auto const dst_offset = dst_offsets[dst_offset_index];

                                         auto const bytes = std::min(src_sizes_unpadded[src_buf_index] - batch_offset, desired_batch_size);

                                         /*                                         
                                         printf("ET: partition_index=%lu, src_buf_index=%lu, dst_buf_index=%lu, batch_index=%lu, src_offset=%lu, dst_offset=%lu bytes=%lu\n", 
                                           partition_index,
                                           src_buf_index,
                                           dst_buf_index,
                                           batch_index,
                                           src_offset + batch_offset,
                                           dst_offset + batch_offset,
                                           bytes);
                                           */

                                         return assemble_batch {
                                          partitions + src_offset + batch_offset,
                                          dst_buffers[dst_buf_index] + dst_offset + batch_offset,
                                          bytes,
                                          0,  // TODO: handle offsets
                                          0,  // TODO: handle validity shifting
                                          0};
                                         }),
                                       stream,
                                       mr);

  return {std::move(assemble_buffers), std::move(copy_batches)};
}

void assemble_copy(rmm::device_uvector<assemble_batch> const& batches, rmm::cuda_stream_view stream)
{
  auto input_iter = thrust::make_transform_iterator(batches.begin(), cuda::proclaim_return_type<void*>([] __device__ (assemble_batch const& batch){
    return reinterpret_cast<void*>(const_cast<int8_t*>(batch.src));
  }));
  auto output_iter = thrust::make_transform_iterator(batches.begin(), cuda::proclaim_return_type<void*>([] __device__ (assemble_batch const& batch){
    return reinterpret_cast<void*>(batch.dst);
  }));
  auto size_iter = thrust::make_transform_iterator(batches.begin(), cuda::proclaim_return_type<size_t>([] __device__ (assemble_batch const& batch){
    return batch.size;
  }));

  size_t temp_storage_bytes;
  cub::DeviceMemcpy::Batched(nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, batches.size(), stream);
  rmm::device_buffer temp_storage(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  cub::DeviceMemcpy::Batched(temp_storage.data(), temp_storage_bytes, input_iter, output_iter, size_iter, batches.size(), stream);

  stream.synchronize();
}

// assemble all the columns and the final table from the intermediate buffers
std::unique_ptr<cudf::table> build_table(std::vector<assemble_column_info> const& assembly_data,
                                         std::vector<rmm::device_buffer>& assembly_buffers,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto column = assembly_data.begin();
  auto buffer = assembly_buffers.begin();  
  while(column != assembly_data.end()){    
    std::tie(column, buffer) = cudf::type_dispatcher(cudf::data_type{column->type},
                                                     detail::assemble_column_functor{stream, mr},
                                                     column,
                                                     buffer,
                                                     columns);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace detail

std::unique_ptr<table> shuffle_assemble(shuffle_split_metadata const& global_metadata,
                                        cudf::device_span<int8_t const> partitions,
                                        cudf::device_span<size_t const> partition_offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  // generate the info structs representing the flattened column hierarchy. the total number of assembled rows, null counts, etc
  auto [column_info, h_column_info, column_instance_info, per_partition_metadata_size] = detail::assemble_build_column_info(global_metadata, partitions, partition_offsets, stream, mr);

  // generate the (empty) output buffers based on the column info. note that is not a 1:1 mapping between column info
  // and buffers, since some columns will have validity and some will not.
  auto [dst_buffers, batches] = detail::assemble_build_buffers(column_info, column_instance_info, partitions, partition_offsets.size(), per_partition_metadata_size, stream, mr);  
  
  // copy the data. note that this does not sync.
  detail::assemble_copy(batches, stream);
  
  // return the final assembled table
  return build_table(h_column_info, dst_buffers, stream, mr);
}

};  // namespace spark_rapids_jni
