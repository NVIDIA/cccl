// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/__algorithm/copy.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/execution>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <stream_registry_factory.h>

#include <c2h/catch2_test_macros.h>
#include <c2h/checked_memory_resource.cuh>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2_test_cuda_utils.cuh>
#include <catch2_test_memory_resources.h>

//! @file
//! This file contains utilities for device-scope API tests of both conventional two-phase APIs and single-phase
//! environment-based APIs (using the wrapper macros suffixed with `_ENV`). See the "Launch wrappers" section of
//! docs/cub/developer/test_overview.rst for usage.
//!
//! Device-scope APIs in CUB can be launched from the host, from device code, or through CUDA graph capture.
//! Utilities in this file facilitate testing in all three cases.
//!
//!
//! ```
//! // Add PARAM to make CMake generate a test for host, device, and graph launches:
//! // %PARAM% TEST_LAUNCH lid 0:1:2
//!
//! // Declare CDP wrapper for CUB API. The wrapper will accept the same
//! // arguments as the CUB API. The wrapper name is provided as the second argument.
//! DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, cub_reduce_sum);
//!
//! CUB_TEST("Reduce test", "[device][reduce]", CUB_SMALL)
//! {
//!   // ...
//!   // Invoke the wrapper from the test. It'll allocate temporary storage and
//!   // invoke the CUB API on the host or device side while checking return
//!   // codes and launch errors.
//!   cub_reduce_sum(d_in, d_out, n, should_be_invoked_on_device);
//!
//!   // Tests with stream-ordered setup can pass a stream-like object as the first argument.
//!   cub_reduce_sum(cuda::get_stream(stream), d_in, d_out, n, should_be_invoked_on_device);
//! }
//!
//! ```
//!
//! It's also possible to cover cuda graph capture. To do that, extend
//! launcher ids with `2` as follows:
//!
//! ```
//! // %PARAM% TEST_LAUNCH lid 0:1:2
//! ```
//!
//! Graph capture backend of launch helper will add extra parameter to each call,
//! so `cub_reduce_sum(d_in, d_out, n, should_be_invoked_on_device)` implicitly turns
//! into `cub_reduce_sum(d_in, d_out, n, should_be_invoked_on_device, stream)`.
//!
//! The stream-aware launch overload uses the caller-provided stream for host and
//! graph launches. Device-side launches cannot consume a host stream; in that mode,
//! the helper synchronizes the caller stream as a dependency boundary and invokes the
//! wrapped API with its default stream argument.
//!
//! If the wrapped API contains default parameters before stream, you'd want to explicitly
//! specify those at all invocations that use graph launch or the stream-aware overload.
//!
//! Consult with `test/catch2_test_launch_wrapper.cu` for more usage examples.

#if !defined(TEST_LAUNCH)
#  error Test file should contain %PARAM% TEST_LAUNCH lid 0:1:2
#endif

#define DECLARE_INVOCABLE(API, WRAPPED_API_NAME, TMPL_HEAD_OPT, TMPL_ARGS_OPT)                        \
  TMPL_HEAD_OPT                                                                                       \
  struct WRAPPED_API_NAME##_invocable_t                                                               \
  {                                                                                                   \
    template <class... Ts>                                                                            \
    CUB_RUNTIME_FUNCTION cudaError_t                                                                  \
    operator()(cuda::std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, Ts... args) const \
    {                                                                                                 \
      return API TMPL_ARGS_OPT(d_temp_storage, temp_storage_bytes, args...);                          \
    }                                                                                                 \
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

namespace launch_helper_detail
{
inline cuda::device_ref device_for_stream(cuda::stream_ref stream)
{
  if (stream == ::cudaStream_t{})
  {
    return ::cub_test::current_device();
  }

  return stream.device();
}

class scoped_current_device
{
public:
  explicit scoped_current_device(cuda::device_ref device)
  {
    REQUIRE(cudaSuccess == cudaGetDevice(&m_previous_device));

    if (m_previous_device != device.get())
    {
      REQUIRE(cudaSuccess == cudaSetDevice(device.get()));
      m_restore = true;
    }
  }

  scoped_current_device(const scoped_current_device&)            = delete;
  scoped_current_device& operator=(const scoped_current_device&) = delete;

  ~scoped_current_device() noexcept
  {
    if (m_restore)
    {
      (void) cudaSetDevice(m_previous_device);
    }
  }

private:
  int m_previous_device = 0;
  bool m_restore        = false;
};

inline void synchronize(cuda::stream_ref stream)
{
  REQUIRE(cudaSuccess == cudaStreamSynchronize(stream.get()));
}

template <typename T>
T read_single(cuda::stream_ref stream, const cuda::device_buffer<T>& buffer)
{
  REQUIRE(buffer.size() == 1);

  T result{};
  cuda::copy_bytes(stream, buffer, cuda::std::span<T>{&result, 1});
  stream.sync();
  return result;
}
} // namespace launch_helper_detail

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
void launch(cuda::stream_ref stream, ActionT action, Args... args)
{
  const auto device = launch_helper_detail::device_for_stream(stream);
  const launch_helper_detail::scoped_current_device device_scope{device};

  std::size_t temp_storage_bytes{};
  cudaError_t error = action(nullptr, temp_storage_bytes, args..., stream.get());
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == error);

  {
    // Keep temp_storage scoped so cuda::device_buffer deallocates on stream before the stream is destroyed.
    auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, temp_storage_bytes, cuda::no_init);

    cudaGraph_t graph{};
    REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal));
    error = action(temp_storage.data(), temp_storage_bytes, args..., stream.get());
    REQUIRE(cudaSuccess == cudaStreamEndCapture(stream.get(), &graph));
    REQUIRE(cudaSuccess == error);

    cudaGraphExec_t exec{};
    REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream.get()));
    launch_helper_detail::synchronize(stream);

    REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
    REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  }
}

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));
  launch(cuda::stream_ref{stream}, action, args...);
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

#elif TEST_LAUNCH == 1

template <class ActionT, class... Args>
__global__ void device_side_api_launch_kernel(
  cuda::std::uint8_t* d_temp_storage,
  std::size_t* temp_storage_bytes,
  cudaError_t* d_error,
  ActionT action,
  Args... args)
{
  // The clang-tidy job uses clang-20 but clang does not support CUDA dynamic parallelism until
  // clang-22. Since we are inside clang-tidy we don't actually care whether the kernel is
  // invoked so do what we must to silence any compiler errors (though if we ever do use
  // clang-22+ then invoke the kernel anyways to have clang-tidy check it).
#  ifdef _CCCL_CLANG_TIDY_INVOKED
#    if _CCCL_HAS_CDP()
  *d_error = action(d_temp_storage, *temp_storage_bytes, args...);
#    else // ^^^  _CCCL_HAS_CDP() ^^^ / vvv ! _CCCL_HAS_CDP() vvv
  static_cast<void>(d_temp_storage);
  static_cast<void>(temp_storage_bytes);
  static_cast<void>(action);
  (static_cast<void>(args), ...);
  *d_error = cudaSuccess;
#    endif // ! _CCCL_HAS_CDP()
#  else // ^^^ _CCCL_CLANG_TIDY_INVOKED ^^^ / vvv !_CCCL_CLANG_TIDY_INVOKED vvv
  *d_error = action(d_temp_storage, *temp_storage_bytes, args...);
#  endif // !_CCCL_CLANG_TIDY_INVOKED
}

// A host stream cannot be consumed by the device-side CUB call. The stream-aware
// overload only uses it to make pending setup visible before launching the CDP kernel.

template <class ActionT, class... Args>
void launch(cuda::stream_ref stream, ActionT action, Args... args)
{
  const auto device = launch_helper_detail::device_for_stream(stream);
  const launch_helper_detail::scoped_current_device device_scope{device};

  auto d_error              = c2h::make_device_buffer<cudaError_t>(stream, device, 1, cuda::no_init);
  auto d_temp_storage_bytes = c2h::make_device_buffer<cuda::std::size_t>(stream, device, 1, cuda::no_init);

  auto* const d_error_ptr              = d_error.data();
  auto* const d_temp_storage_bytes_ptr = d_temp_storage_bytes.data();

  launch_helper_detail::synchronize(stream);
  device_side_api_launch_kernel<<<1, 1>>>(nullptr, d_temp_storage_bytes_ptr, d_error_ptr, action, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == launch_helper_detail::read_single(stream, d_error));

  const auto temp_storage_bytes = launch_helper_detail::read_single(stream, d_temp_storage_bytes);
  auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, temp_storage_bytes, cuda::no_init);

  launch_helper_detail::synchronize(stream);
  device_side_api_launch_kernel<<<1, 1>>>(temp_storage.data(), d_temp_storage_bytes_ptr, d_error_ptr, action, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == launch_helper_detail::read_single(stream, d_error));
}

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  auto [device, stream] = ::cub_test::make_current_device_and_owning_stream();
  launch(cuda::stream_ref{stream}, action, args...);
}

#elif TEST_LAUNCH == 0

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  cuda::std::size_t temp_storage_bytes{};
  cudaError_t error = action(nullptr, temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);

  REQUIRE(temp_storage_bytes > 0); // required by API contract

  // randomly offset the temporary storage address by one byte
  const int offset      = GENERATE(take(1, random(0, 1)));
  auto [device, stream] = ::cub_test::make_current_device_and_owning_stream();
  auto temp_storage =
    c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, temp_storage_bytes + offset, cuda::no_init);

  error = action(temp_storage.data() + offset, temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);
}

template <class ActionT, class... Args>
void launch(cuda::stream_ref stream, ActionT action, Args... args)
{
  const auto device = launch_helper_detail::device_for_stream(stream);
  const launch_helper_detail::scoped_current_device device_scope{device};
  launch(action, args..., stream.get());
}
#else // TEST_LAUNCH == 2
#  error "Unsupported TEST_LAUNCH value. Supported values are 0, 1, or 2"
#endif // TEST_LAUNCH == 2

template <class ActionT, class StreamT, class... Args>
auto launch(ActionT action, StreamT&& stream, Args... args) -> ::cuda::std::void_t<decltype(::cuda::get_stream(stream))>
{
  launch(::cuda::get_stream(stream), action, args...);
}

struct get_expected_allocation_size_t
{};

[[nodiscard]] inline _CCCL_API cuda::std::execution::prop<get_expected_allocation_size_t, size_t>
expected_allocation_size(size_t expected)
{
  return cuda::std::execution::prop{get_expected_allocation_size_t{}, expected};
}

template <size_t... Is, class TplT, class EnvT>
[[nodiscard]] auto replace_back(cuda::std::integer_sequence<size_t, Is...>, TplT tpl, const EnvT& env)
{
  return cuda::std::make_tuple(cuda::std::get<Is>(tpl)..., env);
}

#define DECLARE_INVOCABLE_ENV(API, WRAPPED_API_NAME, TMPL_HEAD_OPT, TMPL_ARGS_OPT) \
  TMPL_HEAD_OPT                                                                    \
  struct WRAPPED_API_NAME##_invocable_t                                            \
  {                                                                                \
    template <class... Ts>                                                         \
    CUB_RUNTIME_FUNCTION cudaError_t operator()(Ts... args) const                  \
    {                                                                              \
      return API TMPL_ARGS_OPT(args...);                                           \
    }                                                                              \
  }

#define DECLARE_LAUNCH_WRAPPER_ENV(API, WRAPPED_API_NAME)       \
  DECLARE_INVOCABLE_ENV(API, WRAPPED_API_NAME, , );             \
  [[maybe_unused]] inline constexpr struct WRAPPED_API_NAME##_t \
  {                                                             \
    template <class... As>                                      \
    void operator()(As... args) const                           \
    {                                                           \
      launch_env(WRAPPED_API_NAME##_invocable_t{}, args...);    \
    }                                                           \
  } WRAPPED_API_NAME

// TODO(bgruber): make the following macro also produce a global instance of a functor, but to pass the template
// arguments, we need variable templates from C++14.
#define DECLARE_TMPL_LAUNCH_WRAPPER_ENV(API, WRAPPED_API_NAME, TMPL_PARAMS, TMPL_ARGS)                         \
  DECLARE_INVOCABLE_ENV(API, WRAPPED_API_NAME, ESCAPE_LIST(template <TMPL_PARAMS>), ESCAPE_LIST(<TMPL_ARGS>)); \
  template <TMPL_PARAMS, class... As>                                                                          \
  static void WRAPPED_API_NAME(As... args)                                                                     \
  {                                                                                                            \
    launch_env(WRAPPED_API_NAME##_invocable_t<TMPL_ARGS>{}, args...);                                          \
  }

#if TEST_LAUNCH == 2

template <class ActionT, class... Args>
void launch_env(ActionT action, Args... args)
{
  check_uses_stream_registry_factory<ActionT>();

  // Environment is always last
  constexpr size_t env_idx = sizeof...(Args) - 1;

  // Extract environment from the argument list
  using tpl_t = cuda::std::tuple<Args...>;
  using env_t = cuda::std::tuple_element_t<env_idx, tpl_t>;
  tpl_t tuple(args...);
  const env_t env = cuda::std::get<env_idx>(tuple);

  // Environment-based API should use default stream if not specified in the environment
  cudaStream_t stream{nullptr};

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
  REQUIRE(stream != cudaStream_t{nullptr});

  size_t bytes_allocated{};
  size_t bytes_deallocated{};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");
  auto mr         = device_memory_resource{stream, &bytes_allocated, &bytes_deallocated};
  auto mr_env     = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto fixed_env  = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  cuda::std::apply(
    [stream, action](auto... args) {
      // Make sure specified stream is used
      const stream_scope scope(stream);
      const cudaError_t error = action(args...);
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

  if constexpr (cuda::std::execution::__queryable_with<env_t, get_expected_allocation_size_t>)
  {
    const size_t expected_bytes_allocated = fixed_env.query(get_expected_allocation_size_t{});
    REQUIRE(expected_bytes_allocated == bytes_allocated);
  }
}

#elif TEST_LAUNCH == 1

template <class ActionT, class... Args>
__global__ void device_side_api_launch_kernel_env(cudaError_t* d_error, ActionT action, Args... args)
{
  // The clang-tidy job uses clang-20 but clang does not support CUDA dynamic parallelism until
  // clang-22. Since we are inside clang-tidy we don't actually care whether the kernel is
  // invoked so do what we must to silence any compiler errors (though if we ever do use
  // clang-22+ then invoke the kernel anyways to have clang-tidy check it).
#  ifdef _CCCL_CLANG_TIDY_INVOKED
#    if _CCCL_HAS_CDP()
  *d_error = action(args...);
#    else // ^^^  _CCCL_HAS_CDP() ^^^ / vvv ! _CCCL_HAS_CDP() vvv
  static_cast<void>(action);
  (static_cast<void>(args), ...);
  *d_error = cudaSuccess;
#    endif // ! _CCCL_HAS_CDP()
#  else // ^^^ _CCCL_CLANG_TIDY_INVOKED ^^^ / vvv !_CCCL_CLANG_TIDY_INVOKED vvv
  *d_error = action(args...);
#  endif // !_CCCL_CLANG_TIDY_INVOKED
}

template <class ActionT, class... Args>
void launch_env(ActionT action, Args... args)
{
  check_uses_stream_registry_factory<ActionT>();

  // Environment is always last
  constexpr size_t env_idx = sizeof...(Args) - 1;

  // Extract environment from the argument list
  using tpl_t = cuda::std::tuple<Args...>;
  using env_t = cuda::std::tuple_element_t<env_idx, tpl_t>;
  tpl_t tuple(args...);
  const env_t env = cuda::std::get<env_idx>(tuple);

  static_assert(cuda::std::execution::__queryable_with<env_t, get_expected_allocation_size_t>,
                "Unit tests using env launch wrappers (declared with DECLARE_LAUNCH_WRAPPER_ENV) must pass "
                "expected_allocation_size as property in their env");

  const size_t expected_bytes_allocated = env.query(get_expected_allocation_size_t{});

  auto [device, owning_stream] = ::cub_test::make_current_device_and_owning_stream();
  const auto stream            = ::cuda::stream_ref{owning_stream};
  auto d_error                 = c2h::make_device_buffer<cudaError_t>(stream, device, 1, cuda::no_init);
  auto d_temp_storage =
    c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, expected_bytes_allocated, cuda::no_init);

  auto d_allocated   = c2h::make_device_buffer<cuda::std::size_t>(stream, device, 1, std::size_t{0});
  auto d_deallocated = c2h::make_device_buffer<cuda::std::size_t>(stream, device, 1, std::size_t{0});

  auto* const d_error_ptr       = d_error.data();
  auto* const d_allocated_ptr   = d_allocated.data();
  auto* const d_deallocated_ptr = d_deallocated.data();
  stream.sync();

  // Host-side stream is unusable in device code, force it to be 0
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{cudaStream_t{}}};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");
  auto mr        = device_side_memory_resource{d_temp_storage.data(), d_allocated_ptr, d_deallocated_ptr};
  auto mr_env    = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
  auto fixed_env = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

  cuda::std::apply(
    [&](auto... args) {
      device_side_api_launch_kernel_env<<<1, 1>>>(d_error_ptr, action, args...);
      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());
      REQUIRE(cudaSuccess == launch_helper_detail::read_single(stream, d_error));
    },
    fixed_args);

  const auto allocated   = launch_helper_detail::read_single(stream, d_allocated);
  const auto deallocated = launch_helper_detail::read_single(stream, d_deallocated);
  REQUIRE(allocated == expected_bytes_allocated);
  REQUIRE(allocated == deallocated);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

#else // TEST_LAUNCH == 0

template <class ActionT, class... Args>
void launch_env(ActionT action, Args... args)
{
  check_uses_stream_registry_factory<ActionT>();

  // Environment is always last
  constexpr size_t env_idx = sizeof...(Args) - 1;

  // Extract environment from the argument list
  using tpl_t = cuda::std::tuple<Args...>;
  using env_t = cuda::std::tuple_element_t<env_idx, tpl_t>;
  tpl_t tuple(args...);
  const env_t env = cuda::std::get<env_idx>(tuple);

  // Environment-based API should use default stream if not specified in the environment
  cudaStream_t stream{nullptr};

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

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");

  {
    auto mr         = throwing_memory_resource{};
    auto mr_env     = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
    auto fixed_env  = cuda::std::execution::env{mr_env, env};
    auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

    cuda::std::apply(
      [action](auto... args) {
        REQUIRE(cudaErrorMemoryAllocation == action(args...));
      },
      fixed_args);
  }

  auto mr         = device_memory_resource{stream, &bytes_allocated, &bytes_deallocated};
  auto mr_env     = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto fixed_env  = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);
  auto kernels    = cuda::std::execution::__query_or(env, get_allowed_kernels_t{}, cuda::std::span<void*>{});

  cuda::std::apply(
    [stream, kernels, action](auto... args) {
      // Make sure specified stream and kernels are used
      const stream_scope allowed_stream(stream);
      const kernel_scope allowed_kernels(kernels);
      const cudaError_t error = action(args...);
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

  if constexpr (cuda::std::execution::__queryable_with<env_t, get_expected_allocation_size_t>)
  {
    const size_t expected_bytes_allocated = fixed_env.query(get_expected_allocation_size_t{});
    REQUIRE(expected_bytes_allocated == bytes_allocated);
  }
}

#endif // TEST_LAUNCH == 0
