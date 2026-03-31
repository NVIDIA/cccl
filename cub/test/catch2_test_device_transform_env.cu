// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/stream>

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
  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> cub::detail::transform::transform_policy
  {
    constexpr int min_bytes_in_flight = 64 * 1024;
    constexpr auto algorithm          = cub::detail::transform::Algorithm::prefetch;
    constexpr auto policy             = cub::detail::transform::prefetch_policy{8, 3, 3, 3};
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

  auto env = cuda::execution::__tune(my_policy_selector{});
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
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, cuda::execution::__tune(my_policy_selector{})};
  REQUIRE(cudaSuccess
          == cub::DeviceTransform::Transform(cuda::std::tuple{}, result.data(), result.size(), get_thread_id{}, env));
  stream.sync();

  c2h::device_vector<unsigned> expected{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  REQUIRE(result == expected);
}
