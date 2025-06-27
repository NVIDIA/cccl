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

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <numeric>

#include "catch2_test_device_reduce.cuh"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <c2h/generators.h>

using float_type_list = c2h::type_list<float, double>;

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

C2H_TEST("Deterministic Device reduce works with float and double on gpu", "[reduce][deterministic]", float_type_list)
{
  using type          = typename c2h::get<0, TestType>;
  const int num_items = 1 << 20;
  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

  c2h::device_vector<type> d_output(1);

  const type* d_input_ptr = thrust::raw_pointer_cast(d_input.data());

  const auto env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);
  cub::DeviceReduce::Reduce(d_input_ptr, d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);

  c2h::host_vector<type> h_input = d_input;

  c2h::host_vector<type> h_expected(1);
  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  // with the device RFA result.
  // NOTE: `std::reduce` is not equivalent
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, ::cuda::std::plus<type>());

  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output, type{0.01});
}

C2H_TEST("Deterministic Device reduce works with float and double and is deterministic on gpu with different policies ",
         "[reduce][deterministic]",
         float_type_list)
{
  using type              = typename c2h::get<0, TestType>;
  constexpr int min_items = 1;
  constexpr int max_items = 1 << 20;

  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  CAPTURE(num_items);

  c2h::device_vector<type> d_input(num_items);

  const type min_val = static_cast<type>(-1000.0f);
  const type max_val = static_cast<type>(1000.0f);

  c2h::gen(C2H_SEED(2), d_input, min_val, max_val);
  c2h::device_vector<type> d_output_p1(1);
  c2h::device_vector<type> d_output_p2(1);

  auto env1 = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::gpu_to_gpu), cuda::execution::__tune(hub_t<1, 128>{})};

  auto env2 = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::gpu_to_gpu), cuda::execution::__tune(hub_t<2, 256>{})};

  cub::DeviceReduce::Reduce(d_input.begin(), d_output_p1.begin(), num_items, cuda::std::plus<type>{}, type{}, env1);

  cub::DeviceReduce::Reduce(d_input.begin(), d_output_p2.begin(), num_items, cuda::std::plus<type>{}, type{}, env2);

  c2h::host_vector<type> h_input = d_input;
  c2h::host_vector<type> h_expected(1);
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, ::cuda::std::plus<type>());

  // device RFA result should be approximately equal to host result
  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output_p1, type{0.01});

  // Both device RFA results should be strictly equal, as RFA is deterministic
  REQUIRE(d_output_p1 == d_output_p2);
}

C2H_TEST("Deterministic Device reduce works with float and double on gpu with NaN",
         "[reduce][deterministic]",
         float_type_list)
{
  using type     = typename c2h::get<0, TestType>;
  using limits_t = cuda::std::numeric_limits<type>;

  constexpr int num_items = 1 << 20;

  constexpr auto min_op = cuda::minimum<type>{};
  constexpr auto init   = cuda::std::numeric_limits<type>::max();

  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

  const int num_indices = GENERATE_COPY(take(2, random(1 << 10, num_items)));

  CAPTURE(num_indices, num_items);

  // generate random indices to scatter NaNs
  c2h::device_vector<int> indices(num_indices);
  for (int i = 0; i < 2; ++i)
  {
    const type nan_val = i == 0 ? limits_t::signaling_NaN() : limits_t::quiet_NaN();

    c2h::gen(C2H_SEED(2), indices, 0, num_items - 1);
    auto begin = thrust::make_constant_iterator(nan_val);
    auto end   = begin + num_indices;

    // sprinkle some NaNs randomly throughout the input
    thrust::scatter(c2h::device_policy, begin, end, indices.cbegin(), d_input.begin());
  }

  c2h::device_vector<type> d_output_p1(1);
  c2h::device_vector<type> d_output_p2(1);

  auto env1 = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::gpu_to_gpu), cuda::execution::__tune(hub_t<1, 128>{})};

  auto env2 = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::gpu_to_gpu), cuda::execution::__tune(hub_t<2, 256>{})};

  cub::DeviceReduce::Reduce(d_input.begin(), d_output_p1.begin(), num_items, min_op, init, env1);
  cub::DeviceReduce::Reduce(d_input.begin(), d_output_p2.begin(), num_items, min_op, init, env2);

  REQUIRE_EQ_WITH_NAN_MATCHING(d_output_p1, d_output_p2);
}

C2H_TEST("Deterministic Device reduce works with float and double on gpu with different iterators",
         "[reduce][deterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = 1 << 10;
  const auto env      = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);

  SECTION("device_vector iterators")
  {
    c2h::device_vector<type> d_input(num_items);
    c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

    c2h::device_vector<type> d_output(1);

    cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);

    c2h::host_vector<type> h_input = d_input;

    c2h::host_vector<type> h_expected(1);
    // Requires `std::accumulate` to produce deterministic result which is required for comparison
    // with the device RFA result.
    // NOTE: `std::reduce` is not equivalent
    h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, ::cuda::std::plus<type>());
    c2h::host_vector<type> h_output = d_output;

    REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
  }

  SECTION("constant iterator")
  {
    thrust::constant_iterator<type> input(1.0f);
    c2h::device_vector<type> d_output(1);

    cub::DeviceReduce::Reduce(input, d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);

    c2h::host_vector<type> h_expected(1);
    // Requires `std::accumulate` to produce deterministic result which is required for comparison
    // with the device RFA result.
    // NOTE: `std::reduce` is not equivalent
    h_expected[0] = std::accumulate(input, input + num_items, type{}, ::cuda::std::plus<type>());

    c2h::host_vector<type> h_output = d_output;
    REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
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

  auto h_input = thrust::make_transform_iterator(input, transform_t{});

  c2h::host_vector<type> h_expected(1);
  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  h_expected[0] = std::accumulate(h_input, h_input + num_items, type{}, ::cuda::std::plus<type>());

  // device RFA result should be approximately equal to host result
  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output, type{0.01});
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

  const auto env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);
  cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, init_value, env);

  c2h::host_vector<type> h_input = d_input;
  c2h::host_vector<type> h_expected(1);
  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  // with the device RFA result.
  // NOTE: `std::reduce` is not equivalent
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), init_value, ::cuda::std::plus<type>());

  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output, type{0.01});
}

using test_types = c2h::type_list<
  int8_t,
  int32_t,
  int64_t,
  float,
  double
#if TEST_HALF_T()
  ,
  half_t,
  __half
#endif // TEST_HALF_T()
#if TEST_BF_T()
  ,
  bfloat16_t,
  __nv_bfloat16
#endif // TEST_BF_T()
#if TEST_INT128()
  ,
  __int128_t
#endif
  >;

C2H_TEST("Deterministic Device reduce works with various types on gpu with different input types and reduction op",
         "[reduce][deterministic]",
         test_types)
{
  using type = typename c2h::get<0, TestType>;

  const auto env          = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);
  constexpr int num_items = 1 << 10;

  type min_value{}, max_value{};

  // for nvbfloat16_t with CTK 12.0 needs float type value in constructor
  if constexpr (cuda::std::is_same_v<type, __nv_bfloat16> || cuda::std::is_same_v<type, bfloat16_t>)
  {
    min_value = type{-50.0f};
    max_value = type{50.0f};
  }
  else
  {
    min_value = static_cast<type>(-50);
    max_value = static_cast<type>(50);
  }

  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input, min_value, max_value);

  SECTION("plus")
  {
    // test with `cuda::std::plus` reduction operator only if the type is integral
    // as integral types are always deterministic
    // NOTE: we do not test with extended floating types because they are not deterministic
    // with `cuda::std::plus` operator
    if constexpr (::cuda::std::is_integral_v<type>)
    {
      c2h::device_vector<type> d_output(1);

      cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);

      c2h::host_vector<type> h_input = d_input;

      c2h::host_vector<type> h_expected(1);
      // Requires `std::accumulate` to produce deterministic result which is required for comparison
      // with the device RFA result.
      // NOTE: `std::reduce` is not equivalent
      h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, cuda::std::plus<type>{});

      c2h::host_vector<type> h_output = d_output;
      REQUIRE(h_expected == h_output);
    }
  }

  SECTION("minimum")
  {
    c2h::device_vector<type> d_output(1);

    using init_t = cub::detail::it_value_t<decltype(unwrap_it(thrust::raw_pointer_cast(d_input.data())))>;
    init_t init_value{::cuda::std::numeric_limits<init_t>::max()};

    cub::DeviceReduce::Reduce(
      unwrap_it(thrust::raw_pointer_cast(d_input.data())),
      unwrap_it(thrust::raw_pointer_cast(d_output.data())),
      num_items,
      ::cuda::minimum<init_t>{},
      init_value,
      env);

    c2h::host_vector<type> h_input = d_input;
    c2h::host_vector<type> h_expected(1);
    h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{init_value}, ::cuda::minimum<>{});

    c2h::host_vector<type> h_output = d_output;
    if constexpr (::cuda::is_floating_point_v<type>)
    {
      REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
    }
    else
    {
      REQUIRE(h_expected == h_output);
    }
  }

  SECTION("maximum")
  {
    c2h::device_vector<type> d_output(1);

    using init_t = cub::detail::it_value_t<decltype(unwrap_it(thrust::raw_pointer_cast(d_input.data())))>;
    init_t init_value{::cuda::std::numeric_limits<init_t>::min()};

    cub::DeviceReduce::Reduce(
      unwrap_it(thrust::raw_pointer_cast(d_input.data())),
      unwrap_it(thrust::raw_pointer_cast(d_output.data())),
      num_items,
      ::cuda::maximum<>{},
      init_value,
      env);

    c2h::host_vector<type> h_input = d_input;
    c2h::host_vector<type> h_expected(1);
    h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{init_value}, ::cuda::maximum<>{});

    c2h::host_vector<type> h_output = d_output;
    if constexpr (::cuda::is_floating_point_v<type>)
    {
      REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
    }
    else
    {
      REQUIRE(h_expected == h_output);
    }
  }
}
