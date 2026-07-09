// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/stream>

#include <sstream>

#include <c2h/catch2_test_helper.h>

using namespace thrust::placeholders;

struct stream_convertible
{
  cudaStream_t stream;

  operator cudaStream_t() const noexcept
  {
    return stream;
  }
};
struct stream_convertible_non_copyable
{
  cudaStream_t stream;

  stream_convertible_non_copyable(cudaStream_t stream)
      : stream(stream)
  {}

  stream_convertible_non_copyable(const stream_convertible_non_copyable&)                    = delete;
  auto operator=(const stream_convertible_non_copyable&) -> stream_convertible_non_copyable& = delete;
  stream_convertible_non_copyable(stream_convertible_non_copyable&&)                         = default;
  auto operator=(stream_convertible_non_copyable&&) -> stream_convertible_non_copyable&      = default;

  operator cudaStream_t() const noexcept
  {
    return stream;
  }
};

struct with_stream_method
{
  cudaStream_t str;

  auto stream() const noexcept
  {
    return str;
  }
};

struct with_get_stream_method
{
  cudaStream_t stream;

  auto get_stream() const noexcept
  {
    return stream;
  }
};

template <typename F>
void check_graph_nodes_with_different_streams(F call_cub_api)
{
  // create stream and begin capture
  cuda::stream stream{cuda::devices[0]};
  // REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  REQUIRE(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal) == cudaSuccess);

  // test various streams
  SECTION("cudaStream_t")
  {
    call_cub_api(stream.get());
  }
  SECTION("cuda::stream_ref")
  {
    call_cub_api(cuda::stream_ref{stream});
  }
  SECTION("stream_convertible")
  {
    call_cub_api(stream_convertible{stream.get()});
  }
  SECTION("stream_convertible_non_copyable")
  {
    call_cub_api(stream_convertible_non_copyable{stream.get()});
  }
  SECTION("with_stream_method")
  {
    call_cub_api(with_stream_method{stream.get()});
  }
  SECTION("with_get_stream_method")
  {
    call_cub_api(with_get_stream_method{stream.get()});
  }
  SECTION("environment with cuda::stream_ref")
  {
    call_cub_api(cuda::std::execution::env{cuda::stream_ref{stream}});
  }
  SECTION("environment with prop with cudaStream_t")
  {
    // MSVC has trouble nesting two aggregate initializations with CTAD
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream, stream.get()};
    call_cub_api(cuda::std::execution::env{cuda::std::move(stream_prop)});
  }
  SECTION("environment with prop with cuda::stream_ref")
  {
    // MSVC has trouble nesting two aggregate initializations with CTAD
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream, cuda::stream_ref{stream}};
    call_cub_api(cuda::std::execution::env{cuda::std::move(stream_prop)});
  }

  // end capture and check that we captured 1 node
  cudaGraph_t graph;
  REQUIRE(cudaGraphCreate(&graph, 0) == cudaSuccess);
  REQUIRE(cudaStreamEndCapture(stream.get(), &graph) == cudaSuccess);
  size_t num_nodes = 0;
  REQUIRE(cudaGraphGetNodes(graph, nullptr, &num_nodes) == cudaSuccess);
  CHECK(num_nodes == 1);

  // run the graph so we can check the results later
  cudaGraphExec_t exec{};
  REQUIRE(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
  REQUIRE(cudaGraphLaunch(exec, stream.get()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  // tear down
  REQUIRE(cudaGraphDestroy(graph) == cudaSuccess);
}

C2H_TEST("DeviceTransform::Transform custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::constant_iterator<type> a{13};
  cuda::counting_iterator<type> b{42};
  c2h::device_vector<type> result(num_items, thrust::no_init);

  check_graph_nodes_with_different_streams([&](auto streamish) {
    cub::DeviceTransform::Transform(cuda::std::make_tuple(a, b), result.begin(), num_items, _1 + _2, streamish);
  });

  CHECK(thrust::equal(result.begin(), result.end(), cuda::counting_iterator<type>{42 + 13}));
}

C2H_TEST("DeviceTransform::Transform (single argument) custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::counting_iterator<type> a{42};
  c2h::device_vector<type> result(num_items, thrust::no_init);

  check_graph_nodes_with_different_streams([&](auto streamish) {
    cub::DeviceTransform::Transform(a, result.begin(), num_items, _1 + 13, streamish);
  });

  CHECK(thrust::equal(result.begin(), result.end(), cuda::counting_iterator<type>{42 + 13}));
}

C2H_TEST("DeviceTransform::Generate custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  auto generator      = cub::detail::__return_constant<type>{1337};
  c2h::device_vector<type> result(num_items, thrust::no_init);

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  check_graph_nodes_with_different_streams([&](auto streamish) {
    cub::DeviceTransform::Generate(result.begin(), num_items, generator, streamish);
  });

  CHECK(thrust::equal(result.begin(), result.end(), cuda::constant_iterator<type>{1337}));
}

C2H_TEST("DeviceTransform::Fill custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  c2h::device_vector<type> result(num_items, thrust::no_init);

  check_graph_nodes_with_different_streams([&](auto streamish) {
    cub::DeviceTransform::Fill(result.begin(), num_items, 0xBAD, streamish);
  });

  CHECK(thrust::equal(result.begin(), result.end(), cuda::constant_iterator<type>{0xBAD}));
}

struct reference_func
{
  _CCCL_HOST_DEVICE int operator()(int i) const
  {
    const auto sum = i + 13;
    return sum > 1000 ? sum : 1337;
  }
};

C2H_TEST("DeviceTransform::TransformIf custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::constant_iterator<type> a{13};
  cuda::counting_iterator<type> b{42};
  c2h::device_vector<type> result(num_items, 1337);

  check_graph_nodes_with_different_streams([&](auto streamish) {
    cub::DeviceTransform::TransformIf(
      cuda::std::make_tuple(a, b), result.begin(), num_items, (_1 + _2) > 1000, _1 + _2, streamish);
  });

  auto reference_it = cuda::transform_iterator{cuda::counting_iterator{42}, reference_func{}};
  CHECK(thrust::equal(result.begin(), result.end(), reference_it));
}

C2H_TEST("DeviceTransform::TransformIf (single argument) custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::counting_iterator<type> a{42};
  c2h::device_vector<type> result(num_items, 1337);

  check_graph_nodes_with_different_streams([&](auto streamish) {
    cub::DeviceTransform::TransformIf(a, result.begin(), num_items, (_1 + 13) > 1000, _1 + 13, streamish);
  });

  auto reference_it = cuda::transform_iterator{cuda::counting_iterator{42}, reference_func{}};
  CHECK(thrust::equal(result.begin(), result.end(), reference_it));
}

C2H_TEST("DeviceTransform::TransformStableArgumentAddresses custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::constant_iterator<type> a{13};
  cuda::counting_iterator<type> b{42};
  c2h::device_vector<type> result(num_items, thrust::no_init);

  check_graph_nodes_with_different_streams([&](auto streamish) {
    cub::DeviceTransform::TransformStableArgumentAddresses(
      cuda::std::make_tuple(a, b), result.begin(), num_items, _1 + _2, streamish);
  });

  CHECK(thrust::equal(result.begin(), result.end(), cuda::counting_iterator<type>{42 + 13}));
}

// use a policy selector that prescribes to run with exactly 8 threads per block and 3 items per thread
struct my_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::TransformPolicy
  {
    constexpr int min_bytes_in_flight = 64 * 1024;
    constexpr auto algorithm          = cub::TransformAlgorithm::prefetch;
    constexpr auto policy             = cub::TransformPrefetchPolicy{8, 3, 3, 3};
    return {min_bytes_in_flight, algorithm, policy, {}, {}};
  }
};

struct get_thread_id
{
  _CCCL_DEVICE auto operator()() const -> unsigned
  {
    return threadIdx.x;
  }
};

C2H_TEST("DeviceTransform::Transform can be tuned", "[reduce][device]")
{
  c2h::device_vector<unsigned> result(3 * 8, thrust::no_init);

  auto env = cuda::execution::tune(my_policy_selector{});
  REQUIRE(cudaSuccess
          == cub::DeviceTransform::Transform(cuda::std::tuple{}, result.data(), result.size(), get_thread_id{}, env));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::device_vector<unsigned> expected{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  REQUIRE(result == expected);
}

C2H_TEST("DeviceTransform::Transform can be tuned with custom stream", "[reduce][device]")
{
  c2h::device_vector<unsigned> result(3 * 8, thrust::no_init);

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, cuda::execution::tune(my_policy_selector{})};
  REQUIRE(cudaSuccess
          == cub::DeviceTransform::Transform(cuda::std::tuple{}, result.data(), result.size(), get_thread_id{}, env));
  stream.sync();

  c2h::device_vector<unsigned> expected{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  REQUIRE(result == expected);
}

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
C2H_TEST("Test TransformPolicy properties", "[transform][device]")
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::TransformPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::TransformPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::TransformPrefetchPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::TransformPrefetchPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::TransformVectorizedPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::TransformVectorizedPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::TransformAsyncCopyPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::TransformAsyncCopyPolicy>);

  // aggregate init
  constexpr auto p1_prefetch   = cub::TransformPrefetchPolicy{256, 2, 1, 32, 128, 0};
  constexpr auto p1_vectorized = cub::TransformVectorizedPolicy{256, 8, 4};
  constexpr auto p1_async_copy = cub::TransformAsyncCopyPolicy{256, 1, 32, 1};
  constexpr auto p1 =
    cub::TransformPolicy{64 * 1024, cub::TransformAlgorithm::prefetch, p1_prefetch, p1_vectorized, p1_async_copy};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2_prefetch = cub::TransformPrefetchPolicy{
    .threads_per_block         = 256,
    .items_per_thread_no_input = 2,
    .min_items_per_thread      = 1,
    .max_items_per_thread      = 32,
    .prefetch_byte_stride      = 128,
    .unroll_factor             = 0};
  constexpr auto p2_vectorized =
    cub::TransformVectorizedPolicy{.threads_per_block = 256, .items_per_thread = 8, .vec_size = 4};
  constexpr auto p2_async_copy = cub::TransformAsyncCopyPolicy{
    .threads_per_block = 256, .min_items_per_thread = 1, .max_items_per_thread = 32, .unroll_factor = 1};
  constexpr auto p2 = cub::TransformPolicy{
    .min_bytes_in_flight = 64 * 1024,
    .algorithm           = cub::TransformAlgorithm::prefetch,
    .prefetch            = p2_prefetch,
    .vectorized          = p2_vectorized,
    .async_copy          = p2_async_copy};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2_prefetch   = p1_prefetch;
  constexpr auto p2_vectorized = p1_vectorized;
  constexpr auto p2_async_copy = p1_async_copy;
  constexpr auto p2            = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1_prefetch == p2_prefetch);
  STATIC_REQUIRE_FALSE(p1_prefetch != p2_prefetch);

  STATIC_REQUIRE(p1_vectorized == p2_vectorized);
  STATIC_REQUIRE_FALSE(p1_vectorized != p2_vectorized);

  STATIC_REQUIRE(p1_async_copy == p2_async_copy);
  STATIC_REQUIRE_FALSE(p1_async_copy != p2_async_copy);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1_prefetch)
          == "TransformPrefetchPolicy { .threads_per_block = 256, .items_per_thread_no_input = 2"
             ", .min_items_per_thread = 1, .max_items_per_thread = 32"
             ", .prefetch_byte_stride = 128, .unroll_factor = 0 }");
  REQUIRE(to_string(p1_vectorized)
          == "TransformVectorizedPolicy { .threads_per_block = 256, .items_per_thread = 8, .vec_size = 4 }");
  REQUIRE(to_string(p1_async_copy)
          == "TransformAsyncCopyPolicy { .threads_per_block = 256, .min_items_per_thread = 1"
             ", .max_items_per_thread = 32, .unroll_factor = 1, .store_vec_size = 0 }");
  REQUIRE(
    to_string(p1)
    == "TransformPolicy { .min_bytes_in_flight = 65536"
       ", .algorithm = TransformAlgorithm::prefetch"
       ", .prefetch = TransformPrefetchPolicy { .threads_per_block = 256"
       ", .items_per_thread_no_input = 2, .min_items_per_thread = 1, .max_items_per_thread = 32"
       ", .prefetch_byte_stride = 128, .unroll_factor = 0 }"
       ", .vectorized = TransformVectorizedPolicy { .threads_per_block = 256"
       ", .items_per_thread = 8, .vec_size = 4 }"
       ", .async_copy = TransformAsyncCopyPolicy { .threads_per_block = 256"
       ", .min_items_per_thread = 1, .max_items_per_thread = 32, .unroll_factor = 1, .store_vec_size = 0 } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
