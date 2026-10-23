// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/count.h>

#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/std/tuple>
#include <cuda/stream>

#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/buffer_utils.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/device_and_stream.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

// The API checks stream forwarding independently of the wrapper implementation.
[[maybe_unused]] static cudaStream_t expected_host_stream{};

template <class T>
__global__ void prepare_input_kernel(T* d_in, T* d_out, int num_items)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < num_items)
  {
    d_in[i]  = T{21};
    d_out[i] = T{-1};
  }
}

template <class T>
__global__ void cub_api_example_x2_0_kernel(const T* d_in, T* d_out, int num_items)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  if (i < num_items)
  {
    d_out[i] = d_in[i] * T{2};
  }
}

template <class T>
__global__ void cub_api_example_x0_5_kernel(const T* d_in, T* d_out, int num_items)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  if (i < num_items)
  {
    d_out[i] = d_in[i] / T{2};
  }
}

struct cub_api_example_t
{
  static constexpr int threads_in_block = 256;

  template <class T, class KernelT>
  CUB_RUNTIME_FUNCTION static cudaError_t invoke(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KernelT kernel,
    const T* d_in,
    T* d_out,
    int num_items,
    cudaStream_t stream = nullptr)
  {
    constexpr bool should_be_invoked_on_device = (TEST_LAUNCH == 1);

    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (if (should_be_invoked_on_device) { return cudaErrorLaunchFailure; }),
                      (if (!should_be_invoked_on_device) { return cudaErrorLaunchFailure; }));

    // For host execution (TEST_LAUNCH != 1), tests set expected_host_stream to
    // the stream passed to the wrapper. Graph capture substitutes a temporary
    // stream when the default stream is requested, so skip equality in that case.
    // Device execution (TEST_LAUNCH == 1) must receive the device-side default stream.
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      ({
#if TEST_LAUNCH == 2
        // Default-stream graph execution captures on a temporary non-default stream.
        if (expected_host_stream != cudaStream_t{} && stream != expected_host_stream)
#else
        if (stream != expected_host_stream)
#endif
        {
          return cudaErrorInvalidResourceHandle;
        }
      }),
      ({
        if (stream != cudaStream_t{})
        {
          return cudaErrorInvalidResourceHandle;
        }
      }));

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = static_cast<std::size_t>(num_items);
      return cudaSuccess;
    }

    if (temp_storage_bytes != static_cast<std::size_t>(num_items))
    {
      return cudaErrorInvalidValue;
    }

#if TEST_LAUNCH == 2
    NV_IF_TARGET(NV_IS_HOST, ({
                   cudaStreamCaptureStatus status{};
                   cudaStreamIsCapturing(stream, &status);
                   if (status != cudaStreamCaptureStatusActive)
                   {
                     return cudaErrorLaunchFailure;
                   }
                 }));
#endif

    const int blocks_in_grid = (num_items + threads_in_block - 1) / threads_in_block;

    return thrust::cuda_cub::detail::triple_chevron(blocks_in_grid, threads_in_block, 0, stream)
      .doit(kernel, d_in, d_out, num_items);
  }

  template <class T>
  CUB_RUNTIME_FUNCTION static cudaError_t
  x2_0(std::uint8_t* d_temp_storage,
       std::size_t& temp_storage_bytes,
       const T* d_in,
       T* d_out,
       int num_items,
       cudaStream_t stream = nullptr)
  {
    return invoke(d_temp_storage, temp_storage_bytes, cub_api_example_x2_0_kernel<T>, d_in, d_out, num_items, stream);
  }

  template <class T>
  CUB_RUNTIME_FUNCTION static cudaError_t
  x0_5(std::uint8_t* d_temp_storage,
       std::size_t& temp_storage_bytes,
       const T* d_in,
       T* d_out,
       int num_items,
       cudaStream_t stream = nullptr)
  {
    return invoke(d_temp_storage, temp_storage_bytes, cub_api_example_x0_5_kernel<T>, d_in, d_out, num_items, stream);
  }
};

DECLARE_LAUNCH_WRAPPER(cub_api_example_t::x2_0, x2_0);
DECLARE_LAUNCH_WRAPPER(cub_api_example_t::x0_5, x0_5);
DECLARE_TMPL_LAUNCH_WRAPPER(cub_api_example_t::x2_0, x2_0_templated, ESCAPE_LIST(class T), ESCAPE_LIST(T));

CUB_TEST("Launch wrapper works with predefined invocables", "[test][utils]", CUB_SMALL)
{
  INFO("Launch = " << TEST_LAUNCH);

  const int n = 42;
  c2h::device_vector<int> in(n, 21);
  c2h::device_vector<int> out(n);

  const int* d_in = thrust::raw_pointer_cast(in.data());
  int* d_out      = thrust::raw_pointer_cast(out.data());

  {
    x2_0(d_in, d_out, n);

    const auto actual   = static_cast<std::size_t>(thrust::count(c2h::device_policy, out.begin(), out.end(), 42));
    const auto expected = static_cast<std::size_t>(n);

    REQUIRE(actual == expected);
  }

  {
    x0_5(d_out, d_out, n);

    const auto actual   = static_cast<std::size_t>(thrust::count(c2h::device_policy, out.begin(), out.end(), 21));
    const auto expected = static_cast<std::size_t>(n);

    REQUIRE(actual == expected);
  }
}

struct custom_x2_0_invocable
{
  template <class T>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    const T* d_in,
    T* d_out,
    int num_items,
    cudaStream_t stream = nullptr)
  {
    return cub_api_example_t::x2_0(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  }
};

struct custom_x0_5_invocable
{
  template <class T>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    const T* d_in,
    T* d_out,
    int num_items,
    cudaStream_t stream = nullptr)
  {
    return cub_api_example_t::x0_5(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  }
};

CUB_TEST("Launch wrapper works with custom invocables", "[test][utils]", CUB_SMALL)
{
  const int n = 42;
  c2h::device_vector<int> in(n, 21);
  c2h::device_vector<int> out(n);

  const int* d_in = thrust::raw_pointer_cast(in.data());
  int* d_out      = thrust::raw_pointer_cast(out.data());

  {
    launch(custom_x2_0_invocable{}, d_in, d_out, n);

    const auto actual   = static_cast<std::size_t>(thrust::count(c2h::device_policy, out.begin(), out.end(), 42));
    const auto expected = static_cast<std::size_t>(n);

    REQUIRE(actual == expected);
  }

  {
    launch(custom_x0_5_invocable{}, d_out, d_out, n);

    const auto actual   = static_cast<std::size_t>(thrust::count(c2h::device_policy, out.begin(), out.end(), 21));
    const auto expected = static_cast<std::size_t>(n);

    REQUIRE(actual == expected);
  }
}

CUB_TEST("Launch wrapper accepts trailing stream-like arguments", "[test][utils]", CUB_SMALL)
{
  INFO("Launch = " << TEST_LAUNCH);

  constexpr int num_items = 42;
  const auto stream       = c2h::make_current_device_stream();
  c2h::device_vector<int> in(num_items);
  c2h::device_vector<int> out(num_items);
  auto* d_in  = thrust::raw_pointer_cast(in.data());
  auto* d_out = thrust::raw_pointer_cast(out.data());

  const auto test_stream = [&](auto&& stream_arg) {
    expected_host_stream = stream.get();
    prepare_input_kernel<<<1, 64, 0, stream.get()>>>(d_in, d_out, num_items);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());

    SECTION("predefined invocable")
    {
      x2_0(d_in, d_out, num_items, stream_arg);
    }
    SECTION("templated invocable")
    {
      x2_0_templated<int>(d_in, d_out, num_items, stream_arg);
    }
    SECTION("custom invocable")
    {
      launch(custom_x2_0_invocable{}, d_in, d_out, num_items, stream_arg);
    }

    REQUIRE(cudaSuccess == cudaStreamQuery(stream.get()));
    REQUIRE(thrust::count(c2h::device_policy, out.begin(), out.end(), 42) == num_items);
    expected_host_stream = cudaStream_t{};
  };

  test_with_custom_streams(test_stream, stream);
}

CUB_TEST("Launch wrapper accepts an explicit default stream", "[test][utils]", CUB_SMALL)
{
  INFO("Launch = " << TEST_LAUNCH);

  constexpr int num_items = 42;
  const auto device       = c2h::current_device();
  const auto stream       = c2h::default_stream_ref();
  auto in                 = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);
  auto out                = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);
  auto actual             = c2h::make_host_buffer<int>(stream, device, num_items, cuda::no_init);
  auto* d_in              = in.data();
  auto* d_out             = out.data();
  expected_host_stream    = cudaStream_t{};

  // Preparation is deliberately not synchronized before invoking the wrapper.
  const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<64>());
  cuda::launch(stream, config, prepare_input_kernel<int>, d_in, d_out, num_items);

  SECTION("raw stream")
  {
    x2_0(d_in, d_out, num_items, cudaStream_t{});
  }
  SECTION("stream reference")
  {
    x2_0(d_in, d_out, num_items, cuda::stream_ref{cudaStream_t{}});
  }
  SECTION("null stream")
  {
    x2_0(d_in, d_out, num_items, nullptr);
  }
  SECTION("templated invocable")
  {
    x2_0_templated<int>(d_in, d_out, num_items, cudaStream_t{});
  }
  SECTION("custom invocable")
  {
    launch(custom_x2_0_invocable{}, d_in, d_out, num_items, cudaStream_t{});
  }

  REQUIRE(stream.is_done());
  c2h::copy_to_host(stream, out, actual);
  for (const auto value : actual)
  {
    REQUIRE(value == 42);
  }
}
