// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_reduce_nondeterministic.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <numeric>

#include "catch2_test_device_reduce.cuh"
#include <c2h/catch2_test_helper.h>
#include <c2h/generators.h>

using float_type_list =
  c2h::type_list<float
#if _CCCL_PTX_ARCH() >= 600
                 ,
                 double
#endif
                 >;

template <int NOMINAL_BLOCK_THREADS_4B, int NOMINAL_ITEMS_PER_THREAD_4B>
struct AgentReducePolicy
{
  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = 4;

  /// Cooperative block-wide reduction algorithm to use
  static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM =
    cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;

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

    // ReduceNondeterministicPolicy
    using ReduceNondeterministicPolicy = ReducePolicy;
  };

  using MaxPolicy = Policy;
};

C2H_TEST("Nondeterministic Device reduce works with float and double on gpu",
         "[reduce][nondeterministic]",
         float_type_list)
{
  using type          = typename c2h::get<0, TestType>;
  const int num_items = GENERATE_COPY(values({0, 1, 20, 100, 2000, 1 << 20}));
  c2h::device_vector<type> d_input(num_items, thrust::no_init);

  type amplitude = static_cast<type>(1);
  c2h::gen(C2H_SEED(2), d_input, -amplitude / (num_items + 1), 2 * amplitude / (num_items + 1));

  c2h::device_vector<type> d_output(1);

  const auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env));

  c2h::host_vector<type> h_input = d_input;

  c2h::host_vector<type> h_expected(1);
  // TODO: Use std::reduce once we drop support for GCC 7 and 8
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, cuda::std::plus<type>());

  // relative round-off error of recursive summation is proportional to n * type::epsilon,
  // see https://epubs.siam.org/doi/epdf/10.1137/19M1257780

  type relative_err = std::min((num_items + 1) * std::numeric_limits<type>::epsilon(), static_cast<type>(1));
  c2h::host_vector<type> h_actual = d_output;

  REQUIRE_APPROX_EQ_EPSILON(h_expected, h_actual, relative_err);
}

C2H_TEST("Nondeterministic Device reduce works with float and double on gpu with NaN",
         "[reduce][nondeterministic]",
         float_type_list)
{
  using type     = typename c2h::get<0, TestType>;
  using limits_t = cuda::std::numeric_limits<type>;

  const int num_items = GENERATE_COPY(values({0, 1, 20, 100, 2000, 1 << 20}));

  constexpr auto min_op = cuda::minimum<type>{};
  constexpr auto init   = cuda::std::numeric_limits<type>::max();

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

  int num_indices = 0;
  if (num_items > 0)
  {
    num_indices = GENERATE_COPY(take(2, random(1, num_items + 1)));

    CAPTURE(num_indices, num_items);

    // generate random indices to scatter NaNs (no duplicates)
    c2h::device_vector<int> indices(num_items);
    thrust::sequence(c2h::device_policy, indices.begin(), indices.end());
    thrust::shuffle(c2h::device_policy, indices.begin(), indices.end(), thrust::default_random_engine{});

    // Take only the first num_indices elements for scattering
    indices.resize(num_indices);
    for (int i = 0; i < 2; ++i)
    {
      const type nan_val = i == 0 ? limits_t::signaling_NaN() : limits_t::quiet_NaN();

      auto begin = thrust::make_constant_iterator(nan_val);
      auto end   = begin + num_indices;

      // sprinkle some NaNs randomly throughout the input
      thrust::scatter(c2h::device_policy, begin, end, indices.cbegin(), d_input.begin());
    }
  }
  else
  {
    CAPTURE(num_indices, num_items);
  }

  c2h::device_vector<type> d_output_p1(1);
  c2h::device_vector<type> d_output_p2(1);

  auto env1 = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed), cuda::execution::__tune(hub_t<1, 128>{})};

  auto env2 = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed), cuda::execution::__tune(hub_t<2, 256>{})};

  REQUIRE(
    cudaSuccess == cub::DeviceReduce::Reduce(d_input.begin(), d_output_p1.begin(), num_items, min_op, init, env1));
  REQUIRE(
    cudaSuccess == cub::DeviceReduce::Reduce(d_input.begin(), d_output_p2.begin(), num_items, min_op, init, env2));

  REQUIRE_EQ_WITH_NAN_MATCHING(d_output_p1, d_output_p2);
}

C2H_TEST("Nondeterministic Device reduce works with float and double on gpu with different iterators",
         "[reduce][nondeterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(values({0, 1, 20, 100, 2000, 1 << 10}));
  const auto env      = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  SECTION("device_vector iterators")
  {
    c2h::device_vector<type> d_input(num_items, thrust::no_init);
    c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

    c2h::device_vector<type> d_output(1);

    REQUIRE(
      cudaSuccess
      == cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env));

    c2h::host_vector<type> h_input = d_input;

    c2h::host_vector<type> h_expected(1);
    // TODO: Use std::reduce once we drop support for GCC 7 and 8
    h_expected[0]                   = std::accumulate(h_input.begin(), h_input.end(), type{}, cuda::std::plus<type>());
    c2h::host_vector<type> h_output = d_output;

    REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
  }

  SECTION("constant iterator")
  {
    thrust::constant_iterator<type> input(1.0f);
    c2h::device_vector<type> d_output(1);

    REQUIRE(cudaSuccess
            == cub::DeviceReduce::Reduce(input, d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env));

    c2h::host_vector<type> h_output = d_output;
    REQUIRE(h_output[0] == static_cast<type>(num_items));
  }
}

// Transform that composes casting with thrust::square
template <class T>
struct square_t
{
  __host__ __device__ T operator()(int x) const
  {
    return thrust::square<T>{}(static_cast<T>(x));
  }
};

C2H_TEST("Nondeterministic Device reduce works with float and double on gpu with different transform operators",
         "[reduce][nondeterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(values({0, 1, 20, 100, 2000, 1 << 10}));

  using input_it_t = cuda::counting_iterator<int>;
  auto input       = input_it_t(1);
  c2h::device_vector<type> d_output(1);

  auto* raw_ptr = thrust::raw_pointer_cast(d_output.data());

  using output_it_t = decltype(raw_ptr);
  using init_t      = type;
  using accum_t     = type;
  using transform_t = square_t<type>;

  using nondeterministic_dispatch_t = cub::detail::
    DispatchReduceNondeterministic<input_it_t, output_it_t, int, cuda::std::plus<type>, init_t, accum_t, transform_t>;

  std::size_t temp_storage_bytes{};

  auto error = nondeterministic_dispatch_t::Dispatch(
    nullptr, temp_storage_bytes, input, raw_ptr, num_items, cuda::std::plus<type>{});
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);

  error = nondeterministic_dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    input,
    raw_ptr,
    num_items,
    cuda::std::plus<type>{});
  REQUIRE(error == cudaSuccess);

  auto h_input = thrust::make_transform_iterator(input, transform_t{});

  c2h::host_vector<type> h_expected(1);
  // TODO: Use std::reduce once we drop support for GCC 7 and 8
  h_expected[0] = std::accumulate(h_input, h_input + num_items, type{}, cuda::std::plus<type>());

  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output, type{0.01});
}

C2H_TEST("Nondeterministic Device reduce works with float and double on gpu with different init values",
         "[reduce][nondeterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(values({0, 1, 20, 100, 2000, 1 << 10}));

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_input, static_cast<type>(-1000.0), static_cast<type>(1000.0));

  c2h::device_vector<type> d_output(1);

  const type init_value = static_cast<type>(GENERATE(42, -42, 0));

  const auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Reduce(
            d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, init_value, env));

  c2h::host_vector<type> h_input = d_input;
  c2h::host_vector<type> h_expected(1);
  // TODO: Use std::reduce once we drop support for GCC 7 and 8
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), init_value, cuda::std::plus<type>());

  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output, type{0.01});
}

using test_types =
  c2h::type_list<int32_t,
                 unsigned int,
                 float
#if _CCCL_PTX_ARCH() >= 600
                 ,
                 double
#endif
                 >;

C2H_TEST("Nondeterministic Device reduce works with various types on gpu with different input types",
         "[reduce][nondeterministic]",
         test_types)
{
  using type = typename c2h::get<0, TestType>;

  const auto env      = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  const int num_items = GENERATE_COPY(values({0, 1, 20, 100, 2000, 1 << 10}));

  type min_value{}, max_value{};

  if constexpr (cuda::std::is_unsigned_v<type>)
  {
    min_value = type{0};
    max_value = type{100};
  }
  else
  {
    min_value = type{-50};
    max_value = type{50};
  }

  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_input, min_value, max_value);

  c2h::device_vector<type> d_output(1);

  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env));

  c2h::host_vector<type> h_input = d_input;

  c2h::host_vector<type> h_expected(1);
  // TODO: Use std::reduce once we drop support for GCC 7 and 8
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, cuda::std::plus<type>{});

  c2h::host_vector<type> h_output = d_output;
  if constexpr (cuda::std::is_integral_v<type>)
  {
    REQUIRE(h_expected == h_output);
  }
  else
  {
    REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
  }
}
