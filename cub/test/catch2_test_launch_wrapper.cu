// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Environment launch wrappers require this factory to check stream usage.
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include <cub/config.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_resource>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/execution>
#include <cuda/std/optional>
#include <cuda/stream>

#include <nv/target>

#include <cstddef>
#include <cstdint>

#include <cub_test_macros.h>
#include <cuda_runtime.h>

#include <c2h/buffer_utils.cuh>
#include <c2h/catch2_test_helper.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/device_and_stream.h>
#include <catch2_test_custom_streams.cuh>
#include <catch2_test_launch_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

// Check stream forwarding independently of the wrapper. nullopt expects a helper-owned non-default stream.
[[maybe_unused]] static cuda::std::optional<cudaStream_t> expected_host_stream{cudaStream_t{}};

class expected_stream_scope
{
public:
  explicit expected_stream_scope(cuda::std::optional<cudaStream_t> expected) noexcept
      : previous(expected_host_stream)
  {
    expected_host_stream = expected;
  }

  expected_stream_scope(const expected_stream_scope&)            = delete;
  expected_stream_scope& operator=(const expected_stream_scope&) = delete;

  ~expected_stream_scope() noexcept
  {
    expected_host_stream = previous;
  }

private:
  const cuda::std::optional<cudaStream_t> previous;
};

// This readback synchronizes stream; callers check wrapper completion before invoking it.
template <class T>
void require_all_equal(cuda::stream_ref stream, const cuda::device_buffer<T>& actual, const T& expected)
{
  auto host = c2h::make_host_buffer<T>(stream, c2h::current_device(), actual.size(), cuda::no_init);
  c2h::copy_to_host(stream, actual, host);
  for (std::size_t i = 0; i < host.size(); ++i)
  {
    INFO("Element index: " << i);
    REQUIRE(host[i] == expected);
  }
}

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
struct launch_test_data
{
  cuda::device_buffer<T> in;
  cuda::device_buffer<T> out;

  launch_test_data(cuda::stream_ref stream, int num_items)
      : in(c2h::make_device_buffer<T>(stream, c2h::current_device(), num_items, cuda::no_init))
      , out(c2h::make_device_buffer<T>(stream, c2h::current_device(), num_items, cuda::no_init))
  {}

  // Enqueue initialization; the caller controls when preparation must complete.
  void prepare(cuda::stream_ref stream)
  {
    constexpr int block_size = 64;
    const auto num_items     = static_cast<int>(in.size());
    const auto grid_size     = (num_items + block_size - 1) / block_size;
    const auto config        = cuda::make_config(cuda::grid_dims(grid_size), cuda::block_dims<block_size>());
    cuda::launch(stream, config, prepare_input_kernel<T>, in.data(), out.data(), num_items);
  }
};

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

  CUB_RUNTIME_FUNCTION static std::size_t required_storage_bytes(int num_items)
  {
    return static_cast<std::size_t>(num_items);
  }

  CUB_RUNTIME_FUNCTION static cudaError_t check_execution_target()
  {
    constexpr bool should_be_invoked_on_device = (TEST_LAUNCH == 1);
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (if (should_be_invoked_on_device) { return cudaErrorLaunchFailure; }),
                      (if (!should_be_invoked_on_device) { return cudaErrorLaunchFailure; }));
    return cudaSuccess;
  }

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
    if (const auto error = check_execution_target(); error != cudaSuccess)
    {
      return error;
    }

    // Host execution may receive a helper-owned stream when an environment omits its stream
    // or when a two-phase graph launch requests the default stream.
    // Device execution (TEST_LAUNCH == 1) must receive the device-side default stream.
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      ({
        const bool requires_helper_stream =
          !expected_host_stream || ((TEST_LAUNCH == 2) && *expected_host_stream == cudaStream_t{});
        const bool expected_stream =
          (requires_helper_stream ? stream != cudaStream_t{} : stream == *expected_host_stream);
        if (!expected_stream)
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
      temp_storage_bytes = required_storage_bytes(num_items);
      return cudaSuccess;
    }

    if (temp_storage_bytes != required_storage_bytes(num_items))
    {
      return cudaErrorInvalidValue;
    }

#if TEST_LAUNCH == 2
    NV_IF_TARGET(NV_IS_HOST, ({
                   cudaStreamCaptureStatus status{};
                   const auto error = cudaStreamIsCapturing(stream, &status);
                   if (error != cudaSuccess)
                   {
                     return error;
                   }
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

  template <class T, class KernelT, class Env>
  CUB_RUNTIME_FUNCTION static cudaError_t invoke(KernelT kernel, const T* d_in, T* d_out, int num_items, Env env)
  {
    if (const auto error = check_execution_target(); error != cudaSuccess)
    {
      return error;
    }

    // The host wrapper's allocation-failure probe need not provide a stream.
    const auto stream = cuda::std::execution::__query_or(env, cuda::get_stream_t{}, cuda::stream_ref{cudaStream_t{}});
    auto mr           = cuda::mr::get_memory_resource(env);
    auto bytes        = required_storage_bytes(num_items);
    void* storage{};
    _CCCL_TRY
    {
      storage = mr.allocate(stream, bytes, alignof(std::uint8_t));
    }
    _CCCL_CATCH_ALL
    {
      return cudaErrorMemoryAllocation;
    }

    const auto error = invoke(static_cast<std::uint8_t*>(storage), bytes, kernel, d_in, d_out, num_items, stream.get());
    mr.deallocate(stream, storage, bytes, alignof(std::uint8_t));
    return error;
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

  template <class T, class Env>
  CUB_RUNTIME_FUNCTION static cudaError_t x2_0(const T* d_in, T* d_out, int num_items, Env env)
  {
    return invoke(cub_api_example_x2_0_kernel<T>, d_in, d_out, num_items, env);
  }

  template <class T, class Env>
  CUB_RUNTIME_FUNCTION static cudaError_t x0_5(const T* d_in, T* d_out, int num_items, Env env)
  {
    return invoke(cub_api_example_x0_5_kernel<T>, d_in, d_out, num_items, env);
  }
};

DECLARE_LAUNCH_WRAPPER(cub_api_example_t::x2_0, x2_0);
DECLARE_LAUNCH_WRAPPER(cub_api_example_t::x0_5, x0_5);
DECLARE_TMPL_LAUNCH_WRAPPER(cub_api_example_t::x2_0, x2_0_templated, ESCAPE_LIST(class T), ESCAPE_LIST(T));
DECLARE_LAUNCH_WRAPPER_ENV(cub_api_example_t::x2_0, x2_0_env);
DECLARE_LAUNCH_WRAPPER_ENV(cub_api_example_t::x0_5, x0_5_env);
DECLARE_TMPL_LAUNCH_WRAPPER_ENV(cub_api_example_t::x2_0, x2_0_env_templated, ESCAPE_LIST(class T), ESCAPE_LIST(T));

CUB_TEST("Launch wrapper works with predefined invocables", "[test][utils]", CUB_SMALL)
{
  INFO("Launch = " << TEST_LAUNCH);

  constexpr int num_items = 42;
  const auto stream       = c2h::default_stream_ref();
  launch_test_data<int> data{stream, num_items};
  const auto* d_in = data.in.data();
  auto* d_out      = data.out.data();
  data.prepare(stream);

  {
    x2_0(d_in, d_out, num_items);

    REQUIRE(stream.is_done());
    require_all_equal(stream, data.out, 42);
  }

  {
    x0_5(d_out, d_out, num_items);

    REQUIRE(stream.is_done());
    require_all_equal(stream, data.out, 21);
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
  constexpr int num_items = 42;
  const auto stream       = c2h::default_stream_ref();
  launch_test_data<int> data{stream, num_items};
  const auto* d_in = data.in.data();
  auto* d_out      = data.out.data();
  data.prepare(stream);

  {
    launch(custom_x2_0_invocable{}, d_in, d_out, num_items);

    REQUIRE(stream.is_done());
    require_all_equal(stream, data.out, 42);
  }

  {
    launch(custom_x0_5_invocable{}, d_out, d_out, num_items);

    REQUIRE(stream.is_done());
    require_all_equal(stream, data.out, 21);
  }
}

CUB_TEST("Launch wrapper accepts trailing stream-like arguments", "[test][utils]", CUB_SMALL)
{
  INFO("Launch = " << TEST_LAUNCH);

  constexpr int num_items = 42;
  const auto stream       = c2h::make_current_device_stream();
  launch_test_data<int> data{stream, num_items};
  auto* d_in  = data.in.data();
  auto* d_out = data.out.data();

  const auto test_stream = [&](auto&& stream_arg) {
    const expected_stream_scope expected_scope{stream.get()};

    data.prepare(stream);

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

    REQUIRE(stream.is_done());
    require_all_equal(stream, data.out, 42);
  };

  test_with_custom_streams(test_stream, stream);
}

CUB_TEST("Launch wrapper accepts an explicit default stream", "[test][utils]", CUB_SMALL)
{
  INFO("Launch = " << TEST_LAUNCH);

  constexpr int num_items = 42;
  const auto stream       = c2h::default_stream_ref();
  launch_test_data<int> data{stream, num_items};
  auto* d_in  = data.in.data();
  auto* d_out = data.out.data();

  const expected_stream_scope expected_scope{cudaStream_t{}};

  // Preparation is deliberately not synchronized before invoking the wrapper.
  data.prepare(stream);

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
  require_all_equal(stream, data.out, 42);
}

CUB_TEST("Launch environment wrapper orders preparation and completes kernels", "[test][utils]", CUB_SMALL)
{
  INFO("Launch = " << TEST_LAUNCH);

  // Exercise the shared dispatch with more than one block.
  constexpr int num_items = 513;
  const auto stream       = c2h::make_current_device_stream();

  const auto test_env = [&](auto env) {
    const auto setup_stream = cuda::std::execution::__query_or(env, cuda::get_stream_t{}, c2h::default_stream_ref());
    launch_test_data<int> data{setup_stream, num_items};

    data.prepare(setup_stream);
    cuda::std::optional<cudaStream_t> expected_stream;
    if constexpr (cuda::std::execution::__queryable_with<decltype(env), cuda::get_stream_t>)
    {
      expected_stream = setup_stream.get();
    }
    else
    {
      // Without a stream property, the caller must complete preparation itself.
      setup_stream.sync();
      expected_stream = cuda::std::nullopt;
    }
    const expected_stream_scope expected_scope{expected_stream};

    SECTION("predefined invocable")
    {
      x2_0_env(data.in.data(), data.out.data(), num_items, env);
    }
    SECTION("templated invocable")
    {
      x2_0_env_templated<int>(data.in.data(), data.out.data(), num_items, env);
    }

    REQUIRE(setup_stream.is_done());
    require_all_equal(setup_stream, data.out, 42);

    x0_5_env(data.out.data(), data.out.data(), num_items, env);
    REQUIRE(setup_stream.is_done());
    require_all_equal(setup_stream, data.out, 21);
  };

  SECTION("non-default stream with unsynchronized preparation")
  {
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
    test_env(cuda::std::execution::env{expected_allocation_size(num_items), stream_prop});
  }
  // The environment graph wrapper requires a non-default capture stream.
#if TEST_LAUNCH != 2
  SECTION("explicit default stream with unsynchronized preparation")
  {
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream_t{}, c2h::default_stream_ref()};
    test_env(cuda::std::execution::env{expected_allocation_size(num_items), stream_prop});
  }
#endif // TEST_LAUNCH != 2
  SECTION("no stream property")
  {
    test_env(cuda::std::execution::env{expected_allocation_size(num_items)});
  }
}
