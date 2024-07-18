/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_scan.cuh>

#include <cstdint>

#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// TODO(elstehle) replace with DeviceScan interface once https://github.com/NVIDIA/cccl/issues/50 is addressed
// Temporary wrapper that allows specializing the DeviceScan algorithm for different offset types
template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, typename OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch_scan_wrapper(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ScanOpT scan_op,
  InitValueT init_value,
  OffsetT num_items,
  cudaStream_t stream = 0)
{
  using init_value_t = cub::detail::InputValue<InitValueT>;
  init_value_t init_value_wrapper{init_value};

  return cub::DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, init_value_t, OffsetT>::Dispatch(
    d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, init_value_wrapper, num_items, stream);
}

DECLARE_LAUNCH_WRAPPER(dispatch_scan_wrapper, dispatch_exclusive_scan);

// %PARAM% TEST_LAUNCH lid 0:1:2

// TODO(elstehle) replace with actual offset types, once https://github.com/NVIDIA/cccl/issues/50 is addresed
// List of offset types to be used for testing large number of items
using offset_types = c2h::type_list<std::int64_t, std::uint64_t, std::int32_t, std::uint32_t>;

template <typename ItemT>
struct expected_sum_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ ItemT operator()(const uint64_t index) const
  {
    uint64_t sum_per_full_segment = (segment_size * (segment_size - 1)) / 2;
    uint64_t full_segments        = index / segment_size;
    uint64_t index_within_segment = index % segment_size;

    uint64_t sum_within_partial_segment = (index_within_segment * (index_within_segment - 1)) / 2;
    uint64_t sum_over_full_segments     = full_segments * sum_per_full_segment;
    return static_cast<ItemT>(sum_within_partial_segment + sum_over_full_segments);
  }
};

template <typename ItemT>
struct mod_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ uint64_t operator()(const uint64_t index) const
  {
    return static_cast<ItemT>(index % segment_size);
  }
};

CUB_TEST("DeviceScan works for very large number of items", "[scan][device]", offset_types)
try
{
  using op_t     = cub::Sum;
  using item_t   = std::uint32_t;
  using index_t  = std::uint64_t;
  using offset_t = typename c2h::get<0, TestType>;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_max_ull =
    std::min(static_cast<std::size_t>(::cuda::std::numeric_limits<offset_t>::max()),
             ::cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  offset_t num_items_max = static_cast<offset_t>(num_items_max_ull);
  offset_t num_items_min =
    num_items_max_ull > 10000 ? static_cast<offset_t>(num_items_max_ull - 10000ULL) : offset_t{0};
  // TODO(elstehle) remove single-item size, once https://github.com/NVIDIA/cccl/issues/50 is addresed
  offset_t num_items =
    GENERATE_COPY(values({num_items_max, static_cast<offset_t>(num_items_max - 1), static_cast<offset_t>(1)}),
                  take(2, random(num_items_min, num_items_max)));

  // Prepare input
  constexpr index_t segment_size = 1000;
  auto index_it                  = thrust::make_counting_iterator(index_t{});
  auto items_it                  = thrust::make_transform_iterator(index_it, mod_op<item_t>{segment_size});

  // Output memory allocation
  c2h::device_vector<item_t> d_items_out(num_items);
  auto d_items_out_it = thrust::raw_pointer_cast(d_items_out.data());

  // Run test
  dispatch_exclusive_scan(items_it, d_items_out_it, op_t{}, item_t{}, num_items);

  // Ensure that we created the correct output
  auto expected_out_it =
    thrust::make_transform_iterator(index_it, expected_sum_op<item_t>{static_cast<index_t>(segment_size)});
  bool all_results_correct = thrust::equal(d_items_out.cbegin(), d_items_out.cend(), expected_out_it);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}
