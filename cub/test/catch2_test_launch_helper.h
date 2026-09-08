// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include "catch2_test_memory_resources.h"
#include "stream_registry_factory.h"
#include <c2h/catch2_test_helper.h>

//! @file
//! This file contains utilities for device-scope API tests, both for plain two phase APIs and for single-phase APIs
//! (suffixed with `_ENV`). See the "Launch wrappers" section of docs/cub/developer/test_overview.rst for usage.

#if !defined(TEST_LAUNCH)
#  error Test file should contain %PARAM% TEST_LAUNCH lid 0:1:2
#endif

#define DECLARE_INVOCABLE(API, WRAPPED_API_NAME, TMPL_HEAD_OPT, TMPL_ARGS_OPT)                  \
  TMPL_HEAD_OPT                                                                                 \
  struct WRAPPED_API_NAME##_invocable_t                                                         \
  {                                                                                             \
    template <class... Ts>                                                                      \
    CUB_RUNTIME_FUNCTION cudaError_t                                                            \
    operator()(std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, Ts... args) const \
    {                                                                                           \
      return API TMPL_ARGS_OPT(d_temp_storage, temp_storage_bytes, args...);                    \
    }                                                                                           \
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
  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  std::size_t temp_storage_bytes{};
  cudaError_t error = action(nullptr, temp_storage_bytes, args..., stream);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == error);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  error = action(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, args..., stream);
  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));
  REQUIRE(cudaSuccess == error);

  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

#elif TEST_LAUNCH == 1

template <class ActionT, class... Args>
__global__ void device_side_api_launch_kernel(
  std::uint8_t* d_temp_storage, std::size_t* temp_storage_bytes, cudaError_t* d_error, ActionT action, Args... args)
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

// We should assign 0 to stream argument when launching on device side, because host stream is not valid there.

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  c2h::device_vector<cudaError_t> d_error(1, cudaErrorInvalidValue);
  c2h::device_vector<std::size_t> d_temp_storage_bytes(1, thrust::no_init);
  device_side_api_launch_kernel<<<1, 1>>>(
    nullptr,
    thrust::raw_pointer_cast(d_temp_storage_bytes.data()),
    thrust::raw_pointer_cast(d_error.data()),
    action,
    args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == d_error[0]);

  c2h::device_vector<std::uint8_t> temp_storage(d_temp_storage_bytes[0], thrust::no_init);

  device_side_api_launch_kernel<<<1, 1>>>(
    thrust::raw_pointer_cast(temp_storage.data()),
    thrust::raw_pointer_cast(d_temp_storage_bytes.data()),
    thrust::raw_pointer_cast(d_error.data()),
    action,
    args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == d_error[0]);
}

#else // TEST_LAUNCH == 0

template <class ActionT, class... Args>
void launch(ActionT action, Args... args)
{
  std::size_t temp_storage_bytes{};
  cudaError_t error = action(nullptr, temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);

  REQUIRE(temp_storage_bytes > 0); // required by API contract

  // randomly offset the temporary storage address by one byte
  const int offset = GENERATE(take(1, random(0, 1)));
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes + offset, thrust::no_init);

  error = action(thrust::raw_pointer_cast(temp_storage.data()) + offset, temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);
}

#endif // TEST_LAUNCH == 0

struct get_expected_allocation_size_t
{};

[[nodiscard]] __host__ __device__ static cuda::std::execution::prop<get_expected_allocation_size_t, size_t>
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
  auto kernels    = cuda::std::execution::__query_or(env, get_allowed_kernels_t{}, cuda::std::span<void*>{});

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  cuda::std::apply(
    [stream, kernels, action](auto... args) {
      // Make sure specified stream and kernels are used
      const stream_scope scope(stream);
      const kernel_scope allowed_kernels(kernels);
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

  c2h::device_vector<cudaError_t> d_error(1, cudaErrorInvalidValue);
  c2h::device_vector<std::uint8_t> d_temp_storage(expected_bytes_allocated);
  c2h::device_vector<std::size_t> d_allocated(1, 0);
  c2h::device_vector<std::size_t> d_deallocated(1, 0);

  // Host-side stream is unusable in device code, force it to be 0
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{cudaStream_t{}}};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");
  auto mr = device_side_memory_resource{
    thrust::raw_pointer_cast(d_temp_storage.data()),
    thrust::raw_pointer_cast(d_allocated.data()),
    thrust::raw_pointer_cast(d_deallocated.data())};
  auto mr_env    = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
  auto fixed_env = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

  cuda::std::apply(
    [&](auto... args) {
      device_side_api_launch_kernel_env<<<1, 1>>>(thrust::raw_pointer_cast(d_error.data()), action, args...);
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
