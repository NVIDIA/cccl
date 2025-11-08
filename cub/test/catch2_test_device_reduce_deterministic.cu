// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/dispatch_reduce_deterministic.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

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
  auto error =
    cub::DeviceReduce::Reduce(d_input_ptr, d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);
  REQUIRE(error == cudaSuccess);

  c2h::host_vector<type> h_input = d_input;

  c2h::host_vector<type> h_expected(1);
  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  // with the device RFA result.
  // NOTE: `std::reduce` is not equivalent
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, cuda::std::plus<type>());

  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output, type{0.01});
}

template <typename FType, typename Iter>
struct cyclic_chunk_accessor
{
  Iter d_chunk;
  size_t chunk_size;

  _CCCL_HOST_DEVICE cyclic_chunk_accessor(Iter d_chunk, size_t chunk_size)
      : d_chunk(d_chunk)
      , chunk_size(chunk_size)
  {}

  _CCCL_HOST_DEVICE FType operator()(size_t idx) const
  {
    return d_chunk[idx % chunk_size];
  }
};

using large_offset_type_list = c2h::type_list<double>;
C2H_TEST("Deterministic Device reduce works with float and double on gpu with large offset types and num_items",
         "[reduce][deterministic]",
         large_offset_type_list)
{
  using type                    = typename c2h::get<0, TestType>;
  const size_t random_num_items = static_cast<size_t>(cuda::std::numeric_limits<::cuda::std::int32_t>::max())
                                + GENERATE_COPY(take(1, random(1, 1000)));

  const size_t half_chunk_size = GENERATE_COPY(take(1, random(1, 128)));

  // make the chunk size even, so that we can split it in half
  const size_t chunk_size = half_chunk_size * 2;

  const size_t num_chunks = random_num_items / chunk_size;
  const size_t num_items  = num_chunks * chunk_size;

  c2h::device_vector<type> d_chunk(chunk_size);
  thrust::sequence(thrust::device, d_chunk.begin(), d_chunk.end(), 0);

  // overwrite the second half of the chunk with negative values from the first half
  thrust::transform(
    thrust::device,
    d_chunk.begin(),
    d_chunk.begin() + half_chunk_size,
    d_chunk.begin() + half_chunk_size,
    ::cuda::std::negate<type>{});

  cyclic_chunk_accessor<type, decltype(d_chunk.data())> wrapper{d_chunk.data(), chunk_size};
  auto d_input = thrust::make_transform_iterator(thrust::counting_iterator<size_t>{}, wrapper);
  c2h::device_vector<type> d_output(1);

  const auto env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);
  auto error = cub::DeviceReduce::Reduce(d_input, d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);
  REQUIRE(error == cudaSuccess);

  // expected sum must be zero, as there would be equal number of positive and negative values
  // in the input and they will cancel each other out
  c2h::host_vector<type> h_expected(1, type{});
  c2h::host_vector<type> h_output = d_output;

  // output should be exactly equal to expected i.e 0.0
  REQUIRE_APPROX_EQ_ABS(h_expected, h_output, type{1e-10});
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

  auto error1 =
    cub::DeviceReduce::Reduce(d_input.begin(), d_output_p1.begin(), num_items, cuda::std::plus<type>{}, type{}, env1);
  REQUIRE(error1 == cudaSuccess);

  auto error2 =
    cub::DeviceReduce::Reduce(d_input.begin(), d_output_p2.begin(), num_items, cuda::std::plus<type>{}, type{}, env2);
  REQUIRE(error2 == cudaSuccess);

  c2h::host_vector<type> h_input = d_input;
  c2h::host_vector<type> h_expected(1);
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{}, cuda::std::plus<type>());

  // device RFA result should be approximately equal to host result
  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output_p1, type{0.05});

  // Both device RFA results should be strictly equal, as RFA is deterministic
  REQUIRE(d_output_p1 == d_output_p2);
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

    auto error =
      cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);
    REQUIRE(error == cudaSuccess);

    c2h::host_vector<type> h_input = d_input;

    c2h::host_vector<type> h_expected(1);
    // Requires `std::accumulate` to produce deterministic result which is required for comparison
    // with the device RFA result.
    // NOTE: `std::reduce` is not equivalent
    h_expected[0]                   = std::accumulate(h_input.begin(), h_input.end(), type{}, cuda::std::plus<type>());
    c2h::host_vector<type> h_output = d_output;

    REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
  }

  SECTION("constant iterator")
  {
    thrust::constant_iterator<type> input(1.0f);
    c2h::device_vector<type> d_output(1);

    auto error = cub::DeviceReduce::Reduce(input, d_output.begin(), num_items, cuda::std::plus<type>{}, type{}, env);
    REQUIRE(error == cudaSuccess);

    c2h::host_vector<type> h_expected(1);
    // Requires `std::accumulate` to produce deterministic result which is required for comparison
    // with the device RFA result.
    // NOTE: `std::reduce` is not equivalent
    h_expected[0] = std::accumulate(input, input + num_items, type{}, cuda::std::plus<type>());

    c2h::host_vector<type> h_output = d_output;
    REQUIRE_APPROX_EQ_EPSILON(h_expected, h_output, type{0.01});
  }
}

template <class T>
struct square_t
{
  __host__ __device__ T operator()(int x) const
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
  h_expected[0] = std::accumulate(h_input, h_input + num_items, type{}, cuda::std::plus<type>());

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
    static_cast<type>(42), cuda::std::numeric_limits<type>::max(), cuda::std::numeric_limits<type>::min());

  const auto env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);
  auto error =
    cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, init_value, env);
  REQUIRE(error == cudaSuccess);

  c2h::host_vector<type> h_input = d_input;
  c2h::host_vector<type> h_expected(1);
  // Requires `std::accumulate` to produce deterministic result which is required for comparison
  // with the device RFA result.
  // NOTE: `std::reduce` is not equivalent
  h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), init_value, cuda::std::plus<type>());

  REQUIRE_APPROX_EQ_EPSILON(h_expected, d_output, type{0.01});
}

using test_types =
  c2h::type_list<float,
                 double,
                 int8_t,
                 int32_t,
                 int64_t
#if TEST_INT128()
                 ,
                 __int128_t
#endif
                 >;

C2H_TEST("Deterministic Device reduce works with integral types on gpu with different reduction operators",
         "[reduce][deterministic]",
         test_types)
{
  using type   = typename c2h::get<0, TestType>;
  using init_t = type;

  const auto env          = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);
  constexpr int num_items = 1 << 10;

  type min_value{}, max_value{};

  min_value = static_cast<type>(-50);
  max_value = static_cast<type>(50);

  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(2), d_input, min_value, max_value);

  if constexpr (::cuda::std::is_integral_v<type>)
  {
    SECTION("plus")
    {
      c2h::device_vector<type> d_output(1);

      auto error =
        cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::std::plus<type>{}, init_t{}, env);
      REQUIRE(error == cudaSuccess);

      c2h::host_vector<type> h_input = d_input;

      c2h::host_vector<type> h_expected(1);
      // Requires `std::accumulate` to produce deterministic result which is required for comparison
      // with the device RFA result.
      // NOTE: `std::reduce` is not equivalent
      h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), init_t{}, cuda::std::plus<type>{});

      c2h::host_vector<type> h_output = d_output;
      REQUIRE(h_expected == h_output);
    }

    SECTION("bitwise xor")
    {
      c2h::device_vector<type> d_output(1);

      init_t init_value{};

      auto error = cub::DeviceReduce::Reduce(
        d_input.begin(), d_output.begin(), num_items, cuda::std::bit_xor<>{}, init_value, env);
      REQUIRE(error == cudaSuccess);

      c2h::host_vector<type> h_input = d_input;
      c2h::host_vector<type> h_expected(1);
      h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{init_value}, cuda::std::bit_xor<type>{});

      c2h::host_vector<type> h_output = d_output;
      REQUIRE(h_expected == h_output);
    }

    SECTION("logical or")
    {
      c2h::device_vector<type> d_output(1);

      init_t init_value{};

      auto error = cub::DeviceReduce::Reduce(
        d_input.begin(), d_output.begin(), num_items, cuda::std::logical_or<>{}, init_value, env);
      REQUIRE(error == cudaSuccess);

      c2h::host_vector<type> h_input = d_input;
      c2h::host_vector<type> h_expected(1);
      h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{init_value}, cuda::std::logical_or<>{});

      c2h::host_vector<type> h_output = d_output;
      REQUIRE(h_expected == h_output);
    }
  }

  SECTION("minimum")
  {
    c2h::device_vector<type> d_output(1);

    init_t init_value{cuda::std::numeric_limits<init_t>::max()};

    auto error =
      cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::minimum<init_t>{}, init_value, env);
    REQUIRE(error == cudaSuccess);

    c2h::host_vector<type> h_input = d_input;
    c2h::host_vector<type> h_expected(1);
    h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{init_value}, cuda::minimum<>{});

    c2h::host_vector<type> h_output = d_output;
    REQUIRE(h_expected == h_output);
  }

  SECTION("maximum")
  {
    c2h::device_vector<type> d_output(1);

    init_t init_value{cuda::std::numeric_limits<init_t>::min()};

    auto error =
      cub::DeviceReduce::Reduce(d_input.begin(), d_output.begin(), num_items, cuda::maximum<init_t>{}, init_value, env);
    REQUIRE(error == cudaSuccess);

    c2h::host_vector<type> h_input = d_input;
    c2h::host_vector<type> h_expected(1);
    h_expected[0] = std::accumulate(h_input.begin(), h_input.end(), type{init_value}, cuda::maximum<>{});

    c2h::host_vector<type> h_output = d_output;
    REQUIRE(h_expected == h_output);
  }
}
