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

#include <cub/detail/rfa.cuh>

#include <thrust/device_vector.h>

#include <algorithm>
#include <numeric>

#include "c2h/vector.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"
#include "cub/block/block_load.cuh"
#include "cub/device/dispatch/dispatch_reduce_deterministic.cuh"
#include "cub/thread/thread_load.cuh"
#include <catch2/catch.hpp>

// %PARAM% TEST_LAUNCH lid 0:1:2

using float_type_list = c2h::type_list<float, double>;

template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
CUB_RUNTIME_FUNCTION static cudaError_t DeterministicSum(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  NumItemsT num_items,
  cudaStream_t stream = 0)
{
  CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::DeterministicSum");

  // Signed integer type for global offsets
  using OffsetT = cub::detail::choose_offset_t<NumItemsT>;

  // The output value type
  using OutputT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::value_t<InputIteratorT>>;

  using InitT = OutputT;

  return cub::detail::DeterministicDispatchReduce<InputIteratorT, OutputIteratorT, OffsetT>::Dispatch(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_out,
    static_cast<OffsetT>(num_items),
    InitT{}, // zero-initialize
    stream);
}

template <int NOMINAL_BLOCK_THREADS_4B, int NOMINAL_ITEMS_PER_THREAD_4B>
struct AgentReducePolicy
{
  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = 4;

  /// Cooperative block-wide reduction algorithm to use
  static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING;

  /// Cache load modifier for reading input elements
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::CacheLoadModifier::LOAD_DEFAULT;
  constexpr static int ITEMS_PER_THREAD                 = NOMINAL_ITEMS_PER_THREAD_4B;
  constexpr static int BLOCK_THREADS                    = NOMINAL_BLOCK_THREADS_4B;
};

template <int ItemsPerThread, int BlockSize>
struct hub_t
{
  struct Policy : cub::ChainedPolicy<300, Policy, Policy>
  {
    constexpr static int ITEMS_PER_THREAD = ItemsPerThread;

    using DeterministicReducePolicy = AgentReducePolicy<BlockSize, ItemsPerThread>;

    // SingleTilePolicy
    using SingleTilePolicy = DeterministicReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = DeterministicReducePolicy;
  };

  using MaxPolicy = Policy;
};

template <typename type>
void deterministic_reduce_gpu(const int N)
{
  const int num_items = N;
  c2h::device_vector<type> input(num_items);
  const type min_val = static_cast<type>(0.0f);
  const type max_val = static_cast<type>(1000.0f);
  c2h::gen(CUB_SEED(2), input, min_val, max_val);
  c2h::device_vector<type> output_p1(1);
  c2h::device_vector<type> output_p2(1);

  const type* d_input = thrust::raw_pointer_cast(input.data());

  std::size_t temp_storage_bytes{};

  using deterministic_dispatch_t_p1 =
    cub::detail::DeterministicDispatchReduce<decltype(d_input), decltype(output_p1.begin()), int, hub_t<4, 256>>;

  using deterministic_dispatch_t_p2 =
    cub::detail::DeterministicDispatchReduce<decltype(d_input), decltype(output_p1.begin()), int, hub_t<4, 128>>;

  deterministic_dispatch_t_p1::Dispatch(nullptr, temp_storage_bytes, d_input, output_p1.begin(), num_items);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);

  deterministic_dispatch_t_p1::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, d_input, output_p1.begin(), num_items);

  type const res_p1 = output_p1[0];

  deterministic_dispatch_t_p2::Dispatch(nullptr, temp_storage_bytes, d_input, output_p2.begin(), num_items);

  deterministic_dispatch_t_p2::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, d_input, output_p2.begin(), num_items);

  type const res_p2 = output_p1[0];

  REQUIRE(res_p1 == res_p2);
}

DECLARE_LAUNCH_WRAPPER(DeterministicSum, deterministic_sum);

template <typename type>
void deterministic_reduce_heterogenous(const int N)
{
  const int num_items = N;
  c2h::device_vector<type> input_device(num_items, 1.0f);
  const type min_val = static_cast<type>(0.0f);
  const type max_val = static_cast<type>(1000.0f);
  c2h::gen(CUB_SEED(2), input_device, min_val, max_val);
  c2h::host_vector<type> input_host = input_device;

  cub::detail::rfa_detail::deterministic_sum_t<type> op{};
  cub::detail::rfa_detail::ReproducibleFloatingAccumulator<type> res = std::accumulate(
    input_host.begin(), input_host.end(), cub::detail::rfa_detail::ReproducibleFloatingAccumulator<type>{}, op);

  c2h::device_vector<type> output_device(1);

  const type* device_input = thrust::raw_pointer_cast(input_device.data());

  deterministic_sum(device_input, output_device.begin(), num_items);

  type const res_device = output_device[0];

  REQUIRE(res.conv() == res_device);
}

CUB_TEST("Deterministic Device reduce works with float and double on gpu", "[reduce][deterministic]", float_type_list)
{
  using type          = typename c2h::get<0, TestType>;
  const int num_items = 1 << 28;
  c2h::device_vector<type> input(num_items, 1.0f);
  c2h::device_vector<type> output(1);

  const type* d_input = thrust::raw_pointer_cast(input.data());

  deterministic_sum(d_input, output.begin(), num_items);

  type const res = output[0];

  REQUIRE(res == num_items);
}

CUB_TEST("Deterministic Device reduce works with float and double on gpu with known result",
         "[reduce][deterministic]",
         float_type_list)
{
  using type                     = typename c2h::get<0, TestType>;
  c2h::device_vector<type> input = {1.0f, 2.0f, 3.0f, 4.0f};
  const int num_items            = input.size();
  c2h::device_vector<type> output(1);

  const type* d_input = thrust::raw_pointer_cast(input.data());

  deterministic_sum(d_input, output.begin(), num_items);

  type const res = output[0];

  REQUIRE(res == 10.0f);
}

CUB_TEST("Deterministic Device reduce works with float and double on cpu", "[reduce][deterministic]", float_type_list)
{
  using type          = typename c2h::get<0, TestType>;
  const int num_items = 42;
  c2h::host_vector<type> input(num_items, 1.0f);

  cub::detail::rfa_detail::deterministic_sum_t<type> op{};
  cub::detail::rfa_detail::ReproducibleFloatingAccumulator<type> res =
    std::accumulate(input.begin(), input.end(), cub::detail::rfa_detail::ReproducibleFloatingAccumulator<type>{}, op);

  REQUIRE(res.conv() == num_items);
}

CUB_TEST("Deterministic Device reduce works with float and double on cpu with known result",
         "[reduce][deterministic]",
         float_type_list)
{
  using type                   = typename c2h::get<0, TestType>;
  c2h::host_vector<type> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  cub::detail::rfa_detail::deterministic_sum_t<type> op{};
  cub::detail::rfa_detail::ReproducibleFloatingAccumulator<type> res =
    std::accumulate(input.begin(), input.end(), cub::detail::rfa_detail::ReproducibleFloatingAccumulator<type>{}, op);

  REQUIRE(res.conv() == 15.0f);
}

CUB_TEST("Deterministic Device reduce works with float and double and is deterministic on gpu with different policies ",
         "[reduce][deterministic]",
         float_type_list)
{
  using type              = typename c2h::get<0, TestType>;
  constexpr int max_items = 50000;
  constexpr int min_items = 1;

  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  CAPTURE(num_items);
  deterministic_reduce_gpu<type>(num_items);
}

CUB_TEST("Deterministic Device reduce works with float and double on cpu and gpu and compare result",
         "[reduce][deterministic]",
         float_type_list)
{
  using type              = typename c2h::get<0, TestType>;
  constexpr int max_items = 50000;
  constexpr int min_items = 1;

  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  CAPTURE(num_items);
  deterministic_reduce_heterogenous<type>(num_items);
}
