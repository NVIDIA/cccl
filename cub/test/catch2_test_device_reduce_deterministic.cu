/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/dispatch_reduce_deterministic.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>

#include <numeric>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/generators.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

using float_type_list = c2h::type_list<float, double>;

template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT, typename InitT>
CUB_RUNTIME_FUNCTION static cudaError_t DeterministicSum(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  NumItemsT num_items,
  InitT init_value,
  cudaStream_t stream = 0)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::DeterministicSum");

  // Signed integer type for global offsets
  using OffsetT = cub::detail::choose_offset_t<NumItemsT>;

  // The output value type
  using OutputT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>;

  return cub::detail::DispatchReduceDeterministic<InputIteratorT, OutputIteratorT, OffsetT, InitT>::Dispatch(
    d_temp_storage, temp_storage_bytes, d_in, d_out, static_cast<OffsetT>(num_items), init_value, stream);
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

    using ReducePolicy = AgentReducePolicy<BlockSize, ItemsPerThread>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy;
};

DECLARE_LAUNCH_WRAPPER(DeterministicSum, deterministic_sum);

// TODO (srinivasyadav18): Replace with macro `REQUIRE_APPROX_EQ_EPSILON` once the PR
// https://github.com/NVIDIA/cccl/pull/4842 is merged
template <typename T>
bool approx_eq(const T& expected, const T& actual, const double tolerance = 0.01)
{
  double diff     = std::abs(static_cast<double>(expected) - static_cast<double>(actual));
  double rel_diff = diff / std::abs(static_cast<double>(expected));
  return rel_diff < tolerance;
}

C2H_TEST("Deterministic Device reduce works with float and double on gpu", "[reduce][deterministic]", float_type_list)
{
  using type          = typename c2h::get<0, TestType>;
  const int num_items = 1 << 20;
  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

  c2h::device_vector<type> d_output(1);

  const type* d_input_ptr = thrust::raw_pointer_cast(d_input.data());

  deterministic_sum(d_input_ptr, d_output.begin(), num_items, type{});

  c2h::host_vector<type> h_input = d_input;

  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  // with the device RFA result.
  // NOTE: `std::reduce` is not equivalent
  const type h_expected           = std::accumulate(h_input.begin(), h_input.end(), type{}, ::cuda::std::plus<type>());
  c2h::host_vector<type> h_output = d_output;

  REQUIRE(approx_eq(h_expected, h_output[0]));
}

C2H_TEST("Deterministic Device reduce works with float and double and is deterministic on gpu with different policies ",
         "[reduce][deterministic]",
         float_type_list)
{
  using type              = typename c2h::get<0, TestType>;
  constexpr int min_items = 1;
  constexpr int max_items = 50000;

  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  CAPTURE(num_items);

  c2h::device_vector<type> input(num_items);

  const type min_val = static_cast<type>(-1000.0f);
  const type max_val = static_cast<type>(1000.0f);

  c2h::gen(C2H_SEED(2), input, min_val, max_val);
  c2h::device_vector<type> output_p1(1);
  c2h::device_vector<type> output_p2(1);

  using input_it_t   = const type*;
  input_it_t d_input = thrust::raw_pointer_cast(input.data());

  using output_it_t = decltype(output_p1.begin());

  using init_t      = type;
  using accum_t     = type;
  using transform_t = ::cuda::std::identity;

  using deterministic_dispatch_t_p1 =
    cub::detail::DispatchReduceDeterministic<input_it_t, output_it_t, int, init_t, transform_t, accum_t, hub_t<1, 128>>;

  using deterministic_dispatch_t_p2 =
    cub::detail::DispatchReduceDeterministic<input_it_t, output_it_t, int, init_t, transform_t, accum_t, hub_t<2, 256>>;

  std::size_t temp_storage_bytes{};

  auto error =
    deterministic_dispatch_t_p1::Dispatch(nullptr, temp_storage_bytes, d_input, output_p1.begin(), num_items);
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<std::uint8_t> temp_storage_p1(temp_storage_bytes);

  error = deterministic_dispatch_t_p1::Dispatch(
    thrust::raw_pointer_cast(temp_storage_p1.data()), temp_storage_bytes, d_input, output_p1.begin(), num_items);
  REQUIRE(error == cudaSuccess);

  type const res_p1 = output_p1[0];

  error = deterministic_dispatch_t_p2::Dispatch(nullptr, temp_storage_bytes, d_input, output_p2.begin(), num_items);
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<std::uint8_t> temp_storage_p2(temp_storage_bytes);

  error = deterministic_dispatch_t_p2::Dispatch(
    thrust::raw_pointer_cast(temp_storage_p2.data()), temp_storage_bytes, d_input, output_p2.begin(), num_items);
  REQUIRE(error == cudaSuccess);

  type const res_p2 = output_p2[0];

  c2h::host_vector<type> h_input = input;
  const type h_expected          = std::accumulate(h_input.begin(), h_input.end(), type{}, ::cuda::std::plus<type>());

  // device RFA result should be approximately equal to host result
  REQUIRE(approx_eq(h_expected, res_p1));

  // Both device RFA results should be strictly equal, as RFA is deterministic
  REQUIRE(res_p1 == res_p2);
}

C2H_TEST("Deterministic Device reduce works with float and double on gpu with different iterators",
         "[reduce][deterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 1 << 10;

  SECTION("device_vector iterators")
  {
    c2h::device_vector<type> d_input(num_items);
    c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

    c2h::device_vector<type> d_output(1);

    deterministic_sum(d_input.begin(), d_output.begin(), num_items, type{});

    c2h::host_vector<type> h_input = d_input;

    // Requires `std::accumulate` to produce deterministic result which is required for comparison
    // with the device RFA result.
    // NOTE: `std::reduce` is not equivalent
    const type h_expected = std::accumulate(h_input.begin(), h_input.end(), type{}, ::cuda::std::plus<type>());
    c2h::host_vector<type> h_output = d_output;

    REQUIRE(approx_eq(h_expected, h_output[0]));
  }

  SECTION("constant iterator")
  {
    thrust::constant_iterator<type> input(1.0f);
    c2h::device_vector<type> d_output(1);

    deterministic_sum(input, d_output.begin(), num_items, type{});

    // Requires `std::accumulate` to produce deterministic result which is required for comparison
    // with the device RFA result.
    // NOTE: `std::reduce` is not equivalent
    const type h_expected           = std::accumulate(input, input + num_items, type{}, ::cuda::std::plus<type>());
    c2h::host_vector<type> h_output = d_output;

    REQUIRE(approx_eq(h_expected, h_output[0]));
  }
}

template <class T>
struct square_t
{
  _CCCL_HOST_DEVICE T operator()(int x) const
  {
    return static_cast<T>(x * x);
  }
};

C2H_TEST("Deterministic Device reduce works with float and double on gpu with different transform operators",
         "[reduce][deterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 1 << 10;

  using input_it_t = thrust::counting_iterator<int>;
  auto input       = input_it_t(1);
  c2h::device_vector<type> d_output(1);

  using output_it_t = decltype(d_output.begin());
  using init_t      = type;
  using accum_t     = type;
  using transform_t = square_t<type>;

  using deterministic_dispatch_t =
    cub::detail::DispatchReduceDeterministic<input_it_t, output_it_t, int, init_t, transform_t, accum_t>;

  std::size_t temp_storage_bytes{};

  auto error = deterministic_dispatch_t::Dispatch(nullptr, temp_storage_bytes, input, d_output.begin(), num_items);
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);

  error = deterministic_dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, input, d_output.begin(), num_items);
  REQUIRE(error == cudaSuccess);

  type const h_output = d_output[0];

  auto h_input = thrust::make_transform_iterator(input, transform_t{});
  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  const type h_expected = std::accumulate(h_input, h_input + num_items, type{}, ::cuda::std::plus<type>());

  // device RFA result should be approximately equal to host result
  REQUIRE(approx_eq(h_expected, h_output));
}

C2H_TEST("Deterministic Device reduce works with float and double on gpu with different init values",
         "[reduce][deterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 1 << 10;

  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

  c2h::device_vector<type> d_output(1);

  type init_value = GENERATE_COPY(
    static_cast<type>(42), ::cuda::std::numeric_limits<type>::max(), ::cuda::std::numeric_limits<type>::min());

  deterministic_sum(d_input.begin(), d_output.begin(), num_items, init_value);

  c2h::host_vector<type> h_input = d_input;

  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  // with the device RFA result.
  // NOTE: `std::reduce` is not equivalent
  const type h_expected = std::accumulate(h_input.begin(), h_input.end(), init_value, ::cuda::std::plus<type>());
  c2h::host_vector<type> h_output = d_output;

  REQUIRE(approx_eq(h_expected, h_output[0]));
}
