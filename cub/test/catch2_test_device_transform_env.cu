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
  SECTION("with_stream_method")
  {
    call_cub_api(with_stream_method{stream.get()});
  }
  SECTION("with_get_stream_method")
  {
    call_cub_api(with_get_stream_method{stream.get()});
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
