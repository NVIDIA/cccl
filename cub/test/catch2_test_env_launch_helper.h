// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/optional>

#include <c2h/catch2_test_helper.h>

//! @file
//! This file contains utilities for device-scope API tests of environment APIs.
//!
//! Device-scope API in CUB can be launched from the host, device, or as part of cuda graph.
//! Utilities in this file facilitate testing in all cases.
//!
//! ```
//! // Add PARAM to make CMake generate a test for all launch modes:
//! // %PARAM% TEST_LAUNCH lid 0:1:2
//!
//! // Declare CDP wrapper for CUB API. The wrapper will accept the same
//! // arguments as the CUB API. The wrapper name is provided as the second argument.
//! DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, cub_reduce_sum);
//!
//! C2H_TEST("Reduce test", "[device][reduce]")
//! {
//!   // ...
//!   // Invoke the wrapper from the test. It'll allocate temporary storage and
//!   // invoke the CUB API on the host or device side while checking return
//!   // codes and launch errors.
//!   cub_reduce_sum(d_in, d_out, n, env);
//! }
//!
//! ```
//!
//! Consult with `test/catch2_test_launch_wrapper.cu` for more usage examples.

#if !defined(TEST_LAUNCH)
#  error Test file should contain %PARAM% TEST_LAUNCH lid 0:1:2
#endif

struct get_expected_allocation_size_t
{};

__host__ __device__ static cuda::std::execution::prop<get_expected_allocation_size_t, size_t>
expected_allocation_size(size_t expected)
{
  return cuda::std::execution::prop{get_expected_allocation_size_t{}, expected};
}

struct get_allowed_kernels_t
{};

__host__ __device__ static cuda::std::execution::prop<get_allowed_kernels_t, cuda::std::span<void*>>
allowed_kernels(cuda::std::span<void*> allowed_kernels)
{
  return cuda::std::execution::prop{get_allowed_kernels_t{}, allowed_kernels};
}

struct stream_registry_factory_state_t
{
  cuda::std::optional<cudaStream_t> m_stream;
  cuda::std::span<void*> m_kernels;
};

static CUB_RUNTIME_FUNCTION stream_registry_factory_state_t* get_stream_registry_factory_state()
{
  stream_registry_factory_state_t* ptr{};
  NV_IF_ELSE_TARGET(NV_IS_HOST, (static stream_registry_factory_state_t state; ptr = &state;), (ptr = nullptr;));
  return ptr;
}

struct kernel_launcher_t : thrust::cuda_cub::detail::triple_chevron
{
  template <class... Args>
  CUB_RUNTIME_FUNCTION kernel_launcher_t(Args... args)
      : thrust::cuda_cub::detail::triple_chevron(args...)
  {}

  template <class K, class... Args>
  CUB_RUNTIME_FUNCTION cudaError_t doit(K kernel, Args const&... args) const
  {
    NV_IF_TARGET(NV_IS_HOST, ({
                   auto& kernels = get_stream_registry_factory_state()->m_kernels;
                   if (!kernels.empty())
                   {
                     if (cuda::std::find(kernels.begin(), kernels.end(), reinterpret_cast<void*>(kernel))
                         == kernels.end())
                     {
                       FAIL("Kernel is not allowed: " << c2h::type_name<K>());
                     }
                   }
                 }));
    return thrust::cuda_cub::detail::triple_chevron::doit(kernel, args...);
  }
};

struct stream_registry_factory_t
{
  CUB_RUNTIME_FUNCTION kernel_launcher_t
  operator()(dim3 grid, dim3 block, size_t shared_mem, cudaStream_t stream, bool dependent_launch = false) const
  {
    NV_IF_TARGET(NV_IS_HOST, (if (get_stream_registry_factory_state()->m_stream) {
                   REQUIRE(stream == get_stream_registry_factory_state()->m_stream);
                 }));
    return kernel_launcher_t(grid, block, shared_mem, stream, dependent_launch);
  }

  CUB_RUNTIME_FUNCTION cudaError_t PtxVersion(int& version)
  {
    return cub::PtxVersion(version);
  }

  CUB_RUNTIME_FUNCTION cudaError_t PtxArchId(::cuda::arch_id& arch_id) const
  {
    return cub::detail::ptx_arch_id(arch_id);
  }

  CUB_RUNTIME_FUNCTION cudaError_t MultiProcessorCount(int& sm_count) const
  {
    int device_ordinal;
    cudaError_t error = cudaGetDevice(&device_ordinal);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get SM count
    return cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal);
  }

  template <typename Kernel>
  CUB_RUNTIME_FUNCTION cudaError_t
  MaxSmOccupancy(int& sm_occupancy, Kernel kernel_ptr, int block_size, int dynamic_smem_bytes = 0)
  {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, kernel_ptr, block_size, dynamic_smem_bytes);
  }

  CUB_RUNTIME_FUNCTION cudaError_t MaxGridDimX(int& max_grid_dim_x) const
  {
    int device_ordinal;
    cudaError_t error = cudaGetDevice(&device_ordinal);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get max grid dimension
    return cudaDeviceGetAttribute(&max_grid_dim_x, cudaDevAttrMaxGridDimX, device_ordinal);
  }
};

struct stream_scope
{
  stream_scope(cudaStream_t stream)
  {
    get_stream_registry_factory_state()->m_stream = stream;
  }

  ~stream_scope()
  {
    get_stream_registry_factory_state()->m_stream = cuda::std::nullopt;
  }
};

struct kernel_scope
{
  kernel_scope(cuda::std::span<void*> allowed_kernels)
  {
    get_stream_registry_factory_state()->m_kernels = allowed_kernels;
  }

  ~kernel_scope()
  {
    get_stream_registry_factory_state()->m_kernels = {};
  }
};

struct device_memory_resource : cub::detail::device_memory_resource
{
  cudaStream_t target_stream = 0;
  size_t* bytes_allocated    = nullptr;
  size_t* bytes_deallocated  = nullptr;

  void* allocate_sync(size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous allocation");
    return nullptr;
  }

  void deallocate_sync(void* /* ptr */, size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous deallocation");
  }

  void* allocate(cuda::stream_ref stream, size_t bytes, size_t /* alignment */)
  {
    return allocate(stream, bytes);
  }

  void* allocate(cuda::stream_ref stream, size_t bytes)
  {
    REQUIRE(target_stream == stream.get());

    if (bytes_allocated)
    {
      *bytes_allocated += bytes;
    }
    return cub::detail::device_memory_resource::allocate(stream, bytes);
  }

  void deallocate(const cuda::stream_ref stream, void* ptr, size_t bytes, size_t /* alignment */)
  {
    deallocate(stream, ptr, bytes);
  }

  void deallocate(const cuda::stream_ref stream, void* ptr, size_t bytes)
  {
    REQUIRE(target_stream == stream.get());

    if (bytes_deallocated)
    {
      *bytes_deallocated += bytes;
    }
    cub::detail::device_memory_resource::deallocate(stream, ptr, bytes);
  }

  bool operator==(const device_memory_resource& rhs) const
  {
    return target_stream == rhs.target_stream && bytes_allocated == rhs.bytes_allocated
        && bytes_deallocated == rhs.bytes_deallocated;
  }
  bool operator!=(const device_memory_resource& rhs) const
  {
    return !(*this == rhs);
  }
};
static_assert(::cuda::mr::resource<device_memory_resource>);

struct throwing_memory_resource
{
  void* allocate_sync(size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous allocation");
    return nullptr;
  }

  void deallocate_sync(void* /* ptr */, size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous deallocation");
  }

  void* allocate(cuda::stream_ref /* stream */, size_t /* bytes */, size_t /* alignment */)
  {
    throw "test";
  }

  void* allocate(cuda::stream_ref /* stream */, size_t /* bytes */)
  {
    throw "test";
  }

  void deallocate(cuda::stream_ref /* stream */, void* /* ptr */, size_t /* bytes */, size_t /* alignment*/)
  {
    throw "test";
  }

  void deallocate(cuda::stream_ref /* stream */, void* /* ptr */, size_t /* bytes */)
  {
    throw "test";
  }

  bool operator==(const throwing_memory_resource&) const
  {
    return true;
  }
  bool operator!=(const throwing_memory_resource&) const
  {
    return false;
  }
};
static_assert(::cuda::mr::resource<throwing_memory_resource>);

struct device_side_memory_resource
{
  void* ptr{};
  size_t* bytes_allocated   = nullptr;
  size_t* bytes_deallocated = nullptr;

  __host__ __device__ void* allocate_sync(size_t /* bytes */, size_t /* alignment */)
  {
    cuda::std::terminate();
  }

  __host__ __device__ void deallocate_sync(void* /* ptr */, size_t /* bytes */, size_t /* alignment */)
  {
    cuda::std::terminate();
  }

  __host__ __device__ void* allocate(cuda::stream_ref stream, size_t bytes, size_t /* alignment */)
  {
    return allocate(stream, bytes);
  }

  __host__ __device__ void* allocate(cuda::stream_ref /* stream */, size_t bytes)
  {
    if (bytes_allocated)
    {
      *bytes_allocated += bytes;
    }
    return static_cast<void*>(static_cast<char*>(ptr) + *bytes_allocated);
  }

  __host__ __device__ void deallocate(const cuda::stream_ref /* stream */, void* /* ptr */, size_t bytes)
  {
    if (bytes_deallocated)
    {
      *bytes_deallocated += bytes;
    }
  }

  __host__ __device__ void
  deallocate(const cuda::stream_ref /* stream */, void* /* ptr */, size_t bytes, size_t /* alignment */)
  {
    if (bytes_deallocated)
    {
      *bytes_deallocated += bytes;
    }
  }

  bool operator==(const device_side_memory_resource& rhs) const
  {
    return ptr == rhs.ptr && bytes_allocated == rhs.bytes_allocated && bytes_deallocated == rhs.bytes_deallocated;
  }
  bool operator!=(const device_side_memory_resource& rhs) const
  {
    return !(*this == rhs);
  }
};
static_assert(::cuda::mr::resource<device_side_memory_resource>);

template <size_t... Is, class TplT, class EnvT>
auto replace_back(cuda::std::integer_sequence<size_t, Is...>, TplT tpl, EnvT env)
{
  return cuda::std::make_tuple(cuda::std::get<Is>(tpl)..., env);
}

#define DECLARE_INVOCABLE(API, WRAPPED_API_NAME, TMPL_HEAD_OPT, TMPL_ARGS_OPT) \
  TMPL_HEAD_OPT                                                                \
  struct WRAPPED_API_NAME##_invocable_t                                        \
  {                                                                            \
    template <class... Ts>                                                     \
    CUB_RUNTIME_FUNCTION cudaError_t operator()(Ts... args) const              \
    {                                                                          \
      return API TMPL_ARGS_OPT(args...);                                       \
    }                                                                          \
  }

#define DECLARE_LAUNCH_WRAPPER(API, WRAPPED_API_NAME)           \
  DECLARE_INVOCABLE(API, WRAPPED_API_NAME, , );                 \
  [[maybe_unused]] inline constexpr struct WRAPPED_API_NAME##_t \
  {                                                             \
    template <class... As>                                      \
    void operator()(As... args) const                           \
    {                                                           \
      launch(WRAPPED_API_NAME##_invocable_t{}, args...);        \
    }                                                           \
  } WRAPPED_API_NAME

#define ESCAPE_LIST(...) __VA_ARGS__

// TODO(bgruber): make the following macro also produce a global instance of a functor, but to pass the template
// arguments, we need variable templates from C++14.
#define DECLARE_TMPL_LAUNCH_WRAPPER(API, WRAPPED_API_NAME, TMPL_PARAMS, TMPL_ARGS)                         \
  DECLARE_INVOCABLE(API, WRAPPED_API_NAME, ESCAPE_LIST(template <TMPL_PARAMS>), ESCAPE_LIST(<TMPL_ARGS>)); \
  template <TMPL_PARAMS, class... As>                                                                      \
  static void WRAPPED_API_NAME(As... args)                                                                 \
  {                                                                                                        \
    launch(WRAPPED_API_NAME##_invocable_t<TMPL_ARGS>{}, args...);                                          \
  }

#if TEST_LAUNCH == 2

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  // Environment is always last
  constexpr size_t env_idx = sizeof...(Args) - 1;

  // Extract environment from the argument list
  using tpl_t = cuda::std::tuple<Args...>;
  using env_t = cuda::std::tuple_element_t<env_idx, tpl_t>;
  tpl_t tuple(args...);
  env_t env = cuda::std::get<env_idx>(tuple);

  // Environment-based API should use default stream if not specified in the environment
  cudaStream_t stream{0};

  if constexpr (cuda::std::execution::__queryable_with<env_t, cuda::get_stream_t>)
  {
    // Retrieve stream from the environment if present
    stream = cuda::get_stream(env).get();
  }
  else
  {
    // Create new stream one otherwise
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  }

  // cuda graphs do not support default stream
  REQUIRE(stream != cudaStream_t{0});

  size_t bytes_allocated{};
  size_t bytes_deallocated{};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::__get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");
  auto mr         = device_memory_resource{{}, stream, &bytes_allocated, &bytes_deallocated};
  auto mr_env     = cuda::std::execution::prop{cuda::mr::__get_memory_resource_t{}, mr};
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto fixed_env  = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  cuda::std::apply(
    [stream, action](auto... args) {
      // Make sure specified stream is used
      stream_scope scope(stream);
      cudaError_t error = action(args...);
      REQUIRE(cudaSuccess == error);
    },
    fixed_args);

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

  // Make sure there are no memory leaks
  REQUIRE(bytes_deallocated == bytes_allocated);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));

  if constexpr (!cuda::std::execution::__queryable_with<env_t, cuda::get_stream_t>)
  {
    REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
  }

  size_t expected_bytes_allocated = fixed_env.query(get_expected_allocation_size_t{});
  REQUIRE(expected_bytes_allocated == bytes_allocated);
}

#elif TEST_LAUNCH == 1

template <class ActionT, class... Args>
__global__ void device_side_api_launch_kernel(cudaError_t* d_error, ActionT action, Args... args)
{
  *d_error = action(args...);
}

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  // Environment is always last
  constexpr size_t env_idx = sizeof...(Args) - 1;

  // Extract environment from the argument list
  using tpl_t = cuda::std::tuple<Args...>;
  using env_t = cuda::std::tuple_element_t<env_idx, tpl_t>;
  tpl_t tuple(args...);
  env_t env = cuda::std::get<env_idx>(tuple);

  size_t expected_bytes_allocated = env.query(get_expected_allocation_size_t{});

  c2h::device_vector<cudaError_t> d_error(1, cudaErrorInvalidValue);
  c2h::device_vector<std::size_t> d_temp_storage(expected_bytes_allocated);
  c2h::device_vector<std::size_t> d_allocated(1, 0);
  c2h::device_vector<std::size_t> d_deallocated(1, 0);

  // Host-side stream is unusable in device code, force it to be 0
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{cudaStream_t{}}};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::__get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");
  auto mr = device_side_memory_resource{
    thrust::raw_pointer_cast(d_temp_storage.data()),
    thrust::raw_pointer_cast(d_allocated.data()),
    thrust::raw_pointer_cast(d_deallocated.data())};
  auto mr_env    = cuda::std::execution::prop{cuda::mr::__get_memory_resource_t{}, mr};
  auto fixed_env = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

  cuda::std::apply(
    [&](auto... args) {
      device_side_api_launch_kernel<<<1, 1>>>(thrust::raw_pointer_cast(d_error.data()), action, args...);
      REQUIRE(cudaSuccess == d_error[0]);
    },
    fixed_args);

  REQUIRE(d_allocated[0] == expected_bytes_allocated);
  REQUIRE(d_allocated[0] == d_deallocated[0]);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

#else // TEST_LAUNCH == 0

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  // Environment is always last
  constexpr size_t env_idx = sizeof...(Args) - 1;

  // Extract environment from the argument list
  using tpl_t = cuda::std::tuple<Args...>;
  using env_t = cuda::std::tuple_element_t<env_idx, tpl_t>;
  tpl_t tuple(args...);
  env_t env = cuda::std::get<env_idx>(tuple);

  // Environment-based API should use default stream if not specified in the environment
  cudaStream_t stream{0};

  if constexpr (cuda::std::execution::__queryable_with<env_t, cuda::get_stream_t>)
  {
    // Retrieve stream from the environment if present
    stream = cuda::get_stream(env).get();
  }
  else
  {
    // Create new stream one otherwise
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  }

  size_t bytes_allocated{};
  size_t bytes_deallocated{};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::__get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");

  {
    auto mr         = throwing_memory_resource{};
    auto mr_env     = cuda::std::execution::prop{cuda::mr::__get_memory_resource_t{}, mr};
    auto fixed_env  = cuda::std::execution::env{mr_env, env};
    auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

    cuda::std::apply(
      [action](auto... args) {
        REQUIRE(cudaErrorMemoryAllocation == action(args...));
      },
      fixed_args);
  }

  auto mr         = device_memory_resource{{}, stream, &bytes_allocated, &bytes_deallocated};
  auto mr_env     = cuda::std::execution::prop{cuda::mr::__get_memory_resource_t{}, mr};
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto fixed_env  = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);
  auto kernels    = cuda::std::execution::__query_or(env, get_allowed_kernels_t{}, cuda::std::span<void*>{});

  cuda::std::apply(
    [stream, kernels, action](auto... args) {
      // Make sure specified stream and kernels are used
      stream_scope allowed_stream(stream);
      kernel_scope allowed_kernels(kernels);
      cudaError_t error = action(args...);
      REQUIRE(cudaSuccess == error);
    },
    fixed_args);

  // Make sure there are no memory leaks
  REQUIRE(bytes_deallocated == bytes_allocated);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  if constexpr (!cuda::std::execution::__queryable_with<env_t, cuda::get_stream_t>)
  {
    REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
  }

  size_t expected_bytes_allocated = fixed_env.query(get_expected_allocation_size_t{});
  REQUIRE(expected_bytes_allocated == bytes_allocated);
}

#endif // TEST_LAUNCH == 0

// Helper relies on the fact that CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY is `stream_registry_factory_t`
static_assert(cuda::std::is_same_v<CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY, stream_registry_factory_t>);
