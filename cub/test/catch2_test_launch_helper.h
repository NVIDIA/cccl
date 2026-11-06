// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/__functional/call_or.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/execution>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <stream_registry_factory.h>

#include <c2h/buffer_utils.cuh>
#include <c2h/catch2_test_macros.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/device_and_stream.h>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2_test_memory_resources.h>

//! @file
//! This file contains utilities for device-scope API tests of both conventional two-phase APIs and single-phase
//! environment-based APIs (using the wrapper macros suffixed with `_ENV`). See the "Launch wrappers" section of
//! docs/cub/developer/test_overview.rst for usage.

#if !defined(TEST_LAUNCH)
#  error Test file should contain %PARAM% TEST_LAUNCH lid 0:1:2
#endif

#define DECLARE_INVOCABLE(API, WRAPPED_API_NAME, TMPL_HEAD_OPT, TMPL_ARGS_OPT)                        \
  TMPL_HEAD_OPT                                                                                       \
  struct WRAPPED_API_NAME##_invocable_t                                                               \
  {                                                                                                   \
    template <class... Ts>                                                                            \
    CUB_RUNTIME_FUNCTION auto                                                                         \
    operator()(cuda::std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, Ts... args) const \
      -> decltype(API TMPL_ARGS_OPT(d_temp_storage, temp_storage_bytes, args...))                     \
    {                                                                                                 \
      return API TMPL_ARGS_OPT(d_temp_storage, temp_storage_bytes, args...);                          \
    }                                                                                                 \
  }

#define DECLARE_LAUNCH_WRAPPER(API, WRAPPED_API_NAME)                            \
  DECLARE_INVOCABLE(API, WRAPPED_API_NAME, , );                                  \
  [[maybe_unused]] inline constexpr struct WRAPPED_API_NAME##_t                  \
  {                                                                              \
    template <class... As>                                                       \
    void operator()(As&&... args) const                                          \
    {                                                                            \
      launch(WRAPPED_API_NAME##_invocable_t{}, cuda::std::forward<As>(args)...); \
    }                                                                            \
  } WRAPPED_API_NAME

#define ESCAPE_LIST(...) __VA_ARGS__

// TODO(bgruber): make the following macro also produce a global instance of a functor, but to pass the template
// arguments, we need variable templates from C++14.
#define DECLARE_TMPL_LAUNCH_WRAPPER(API, WRAPPED_API_NAME, TMPL_PARAMS, TMPL_ARGS)                         \
  DECLARE_INVOCABLE(API, WRAPPED_API_NAME, ESCAPE_LIST(template <TMPL_PARAMS>), ESCAPE_LIST(<TMPL_ARGS>)); \
  template <TMPL_PARAMS, class... As>                                                                      \
  static void WRAPPED_API_NAME(As&&... args)                                                               \
  {                                                                                                        \
    launch(WRAPPED_API_NAME##_invocable_t<TMPL_ARGS>{}, cuda::std::forward<As>(args)...);                  \
  }

namespace launch_helper_detail
{
// Own a fallback stream only when the environment does not supply a stream property.
template <class Env>
[[nodiscard]] cuda::stream make_fallback_stream()
{
  if constexpr (cuda::std::execution::__queryable_with<Env, cuda::get_stream_t>)
  {
    return cuda::stream{cuda::no_init};
  }
  else
  {
    return c2h::make_current_device_stream();
  }
}

template <class StreamOrEnv>
[[nodiscard]] auto override_stream(const StreamOrEnv& stream_or_env, cuda::stream_ref stream)
{
  if constexpr (cuda::std::is_same_v<StreamOrEnv, cudaStream_t>)
  {
    return stream.get();
  }
  else
  {
    return cuda::std::execution::env{stream, stream_or_env};
  }
}

#if TEST_LAUNCH == 2

// Executes the captured graph on stream, waits for completion, and destroys both graph handles.
inline void execute_and_destroy_graph(cudaGraph_t graph, cuda::stream_ref stream)
{
  cudaGraphExec_t exec{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  REQUIRE(cudaSuccess == cudaGraphLaunch(exec, stream.get()));
  stream.sync();
  REQUIRE(cudaSuccess == cudaGraphExecDestroy(exec));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
}

template <bool HasStream, class ActionT, class StreamOrEnv, class... Args>
void launch_impl(
  cuda::stream_ref stream, ActionT action, [[maybe_unused]] const StreamOrEnv& stream_or_env, Args... args)
{
  cuda::stream owning_capture_stream{cuda::no_init};
  auto capture_stream = stream;
  if (c2h::detail::is_default_stream(stream))
  {
    owning_capture_stream = c2h::make_current_device_stream();
    capture_stream        = owning_capture_stream;
  }

  const auto invoke = [&](cuda::std::uint8_t* temp_storage, std::size_t& temp_storage_bytes) {
    if constexpr (HasStream)
    {
      if (capture_stream == stream)
      {
        return action(temp_storage, temp_storage_bytes, args..., stream_or_env);
      }
      return action(temp_storage, temp_storage_bytes, args..., override_stream(stream_or_env, capture_stream));
    }
    else
    {
      return action(temp_storage, temp_storage_bytes, args..., capture_stream.get());
    }
  };

  std::size_t temp_storage_bytes{};
  cudaError_t error = invoke(nullptr, temp_storage_bytes);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == error);

  auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(capture_stream, temp_storage_bytes, cuda::no_init);

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(capture_stream.get(), cudaStreamCaptureModeGlobal));
  error = invoke(temp_storage.data(), temp_storage_bytes);
  REQUIRE(cudaSuccess == cudaStreamEndCapture(capture_stream.get(), &graph));
  REQUIRE(cudaSuccess == error);

  // Launch on the requested stream, preserving ordering with the caller's preparation.
  execute_and_destroy_graph(graph, stream);
}

#elif TEST_LAUNCH == 1

inline void check_device_launch_completion(cuda::stream_ref stream, const cuda::device_buffer<cudaError_t>& d_error)
{
  REQUIRE(cudaSuccess == c2h::read_single(stream, d_error));
}

template <class ActionT, class... Args>
__global__ void device_side_api_launch_kernel(
  cuda::std::uint8_t* d_temp_storage,
  std::size_t* temp_storage_bytes,
  cudaError_t* d_error,
  ActionT action,
  Args... args)
{
  *d_error = action(d_temp_storage, *temp_storage_bytes, args...);
}

template <bool HasStream, class ActionT, class StreamOrEnv, class... Args>
void launch_impl(
  cuda::stream_ref stream, ActionT action, [[maybe_unused]] const StreamOrEnv& stream_or_env, Args... args)
{
  const auto device         = c2h::current_device();
  auto d_error              = c2h::make_device_buffer<cudaError_t>(stream, device, 1, cudaErrorInvalidValue);
  auto d_temp_storage_bytes = c2h::make_device_buffer<cuda::std::size_t>(stream, device, 1, cuda::no_init);

  const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>());

  // Launch the parent on the requested stream to preserve preparation ordering.
  // The device-side API receives its own default stream.
  const auto invoke = [&](cuda::std::uint8_t* temp_storage) {
    if constexpr (HasStream)
    {
      const auto device_arg = override_stream(stream_or_env, cuda::stream_ref{cudaStream_t{}});
      cuda::launch(
        stream,
        config,
        device_side_api_launch_kernel<ActionT, Args..., cuda::std::remove_cv_t<decltype(device_arg)>>,
        temp_storage,
        d_temp_storage_bytes.data(),
        d_error.data(),
        action,
        args...,
        device_arg);
    }
    else
    {
      cuda::launch(
        stream,
        config,
        device_side_api_launch_kernel<ActionT, Args...>,
        temp_storage,
        d_temp_storage_bytes.data(),
        d_error.data(),
        action,
        args...);
    }
    check_device_launch_completion(stream, d_error);
  };

  invoke(nullptr);
  const auto temp_storage_bytes = c2h::read_single(stream, d_temp_storage_bytes);
  auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, temp_storage_bytes, cuda::no_init);
  invoke(temp_storage.data());
}

#elif TEST_LAUNCH == 0

template <bool HasStream, class ActionT, class StreamOrEnv, class... Args>
void launch_impl(
  cuda::stream_ref stream, ActionT action, [[maybe_unused]] const StreamOrEnv& stream_or_env, Args... args)
{
  const auto invoke = [&](cuda::std::uint8_t* temp_storage, std::size_t& temp_storage_bytes) {
    if constexpr (HasStream)
    {
      return action(temp_storage, temp_storage_bytes, args..., stream_or_env);
    }
    else
    {
      return action(temp_storage, temp_storage_bytes, args...);
    }
  };

  std::size_t temp_storage_bytes{};
  cudaError_t error = invoke(nullptr, temp_storage_bytes);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == error);
  REQUIRE(temp_storage_bytes > 0); // required by API contract

  // randomly offset the temporary storage address by one byte
  const int offset  = GENERATE(take(1, random(0, 1)));
  auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(
    stream, c2h::current_device(), temp_storage_bytes + offset, cuda::no_init);

  error = invoke(temp_storage.data() + offset, temp_storage_bytes);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  stream.sync();
  REQUIRE(cudaSuccess == error);
}

#else
#  error "Unsupported TEST_LAUNCH value. Supported values are 0, 1, or 2"
#endif

template <class ActionT, class TupleT, cuda::std::size_t... Is>
void launch_with_stream(ActionT action, TupleT& args, cuda::std::index_sequence<Is...>)
{
  const auto& stream_or_env = cuda::std::get<sizeof...(Is)>(args);
  const auto stream         = cuda::__call_or(cuda::get_stream, c2h::default_stream_ref(), stream_or_env);
  using last_arg_t          = cuda::std::remove_cvref_t<decltype(stream_or_env)>;
  constexpr bool is_environment =
    !cuda::std::is_invocable_v<cuda::get_stream_t, const last_arg_t&> || cuda::__has_query_get_stream<last_arg_t>;
  if constexpr (is_environment)
  {
    launch_impl<true>(stream, action, stream_or_env, cuda::std::get<Is>(args)...);
  }
  else
  {
    // make stream copyable to handle owning stream arguments
    launch_impl<true>(stream, action, cuda::stream_ref{stream}, cuda::std::get<Is>(args)...);
  }
}
} // namespace launch_helper_detail

template <class ActionT, class... Args>
void launch(ActionT action, Args&&... args)
{
  if constexpr (sizeof...(Args) != 0)
  {
    using last_arg_t = cuda::std::tuple_element_t<sizeof...(Args) - 1, cuda::std::tuple<Args...>>;
    // MSVC 19.29 can report C1004 when the trait using this local alias appears directly in if constexpr.
    constexpr bool has_stream = cuda::std::is_invocable_v<cuda::get_stream_t, last_arg_t&>;
#if TEST_LAUNCH == 2
    // If appending a stream is invalid, the supplied final argument already occupies
    // the API's environment parameter. Preserve it and inject the capture stream.
    constexpr bool has_stream_or_env =
      has_stream || !cuda::std::is_invocable_v<ActionT&, cuda::std::uint8_t*, std::size_t&, Args..., cudaStream_t>;
#else
    constexpr bool has_stream_or_env = has_stream;
#endif
    if constexpr (has_stream_or_env)
    {
      auto tuple = cuda::std::forward_as_tuple(args...);
      launch_helper_detail::launch_with_stream(action, tuple, cuda::std::make_index_sequence<sizeof...(Args) - 1>{});
    }
    else
    {
      const auto default_stream = c2h::default_stream_ref();
      const cudaStream_t unused_stream_argument{};
      launch_helper_detail::launch_impl<false>(default_stream, action, unused_stream_argument, args...);
    }
  }
  else
  {
    const auto default_stream = c2h::default_stream_ref();
    const cudaStream_t unused_stream_argument{};
    launch_helper_detail::launch_impl<false>(default_stream, action, unused_stream_argument);
  }
}

struct get_expected_allocation_size_t
{};

[[nodiscard]] inline _CCCL_API cuda::std::execution::prop<get_expected_allocation_size_t, size_t>
expected_allocation_size(size_t expected)
{
  return cuda::std::execution::prop{get_expected_allocation_size_t{}, expected};
}

template <size_t... Is, class TplT, class EnvT>
[[nodiscard]] auto replace_back(cuda::std::index_sequence<Is...>, TplT tpl, const EnvT& env)
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

  // Keep a capture stream alive when the environment does not provide one.
  [[maybe_unused]] const auto owning_stream = launch_helper_detail::make_fallback_stream<env_t>();
  cuda::stream_ref stream{cudaStream_t{}};

  if constexpr (cuda::std::execution::__queryable_with<env_t, cuda::get_stream_t>)
  {
    // Retrieve stream from the environment if present
    stream = cuda::get_stream(env);
  }
  else
  {
    stream = owning_stream;
  }

  // cuda graphs do not support default stream
  REQUIRE(!c2h::detail::is_default_stream(stream));

  size_t bytes_allocated{};
  size_t bytes_deallocated{};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");
  auto mr         = device_memory_resource{stream.get(), &bytes_allocated, &bytes_deallocated};
  auto mr_env     = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, stream};
  auto fixed_env  = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal));

  cuda::std::apply(
    [stream, action](auto... args) {
      // Make sure specified stream is used
      const stream_scope scope(stream.get());
      const cudaError_t error = action(args...);
      REQUIRE(cudaSuccess == error);
    },
    fixed_args);

  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream.get(), &graph));

  launch_helper_detail::execute_and_destroy_graph(graph, stream);

  // Make sure there are no memory leaks
  REQUIRE(bytes_deallocated == bytes_allocated);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

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
  *d_error = action(args...);
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

  const auto stream = cuda::std::execution::__query_or(env, cuda::get_stream_t{}, c2h::default_stream_ref());

  auto d_error        = c2h::make_device_buffer<cudaError_t>(stream, 1, cudaErrorInvalidValue);
  auto d_temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(stream, expected_bytes_allocated, cuda::no_init);

  auto d_allocated   = c2h::make_device_buffer<cuda::std::size_t>(stream, 1, std::size_t{0});
  auto d_deallocated = c2h::make_device_buffer<cuda::std::size_t>(stream, 1, std::size_t{0});

  auto* const d_error_ptr       = d_error.data();
  auto* const d_allocated_ptr   = d_allocated.data();
  auto* const d_deallocated_ptr = d_deallocated.data();

  // Host-side stream is unusable in device code, force it to be 0
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{cudaStream_t{}}};

  static_assert(!cuda::std::execution::__queryable_with<env_t, cuda::mr::get_memory_resource_t>,
                "Don't specify memory resource for launch tests.");
  auto mr        = device_side_memory_resource{d_temp_storage.data(), d_allocated_ptr, d_deallocated_ptr};
  auto mr_env    = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, mr};
  auto fixed_env = cuda::std::execution::env{mr_env, stream_env, env};

  auto fixed_args = replace_back(cuda::std::make_index_sequence<env_idx>{}, tuple, fixed_env);

  const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>());

  cuda::std::apply(
    [&](auto... args) {
      cuda::launch(
        stream, config, device_side_api_launch_kernel_env<ActionT, decltype(args)...>, d_error_ptr, action, args...);
      launch_helper_detail::check_device_launch_completion(stream, d_error);
    },
    fixed_args);

  const auto allocated   = c2h::read_single(stream, d_allocated);
  const auto deallocated = c2h::read_single(stream, d_deallocated);
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

  // Keep a fallback stream alive when the environment does not provide one.
  [[maybe_unused]] const auto owning_stream = launch_helper_detail::make_fallback_stream<env_t>();
  cudaStream_t stream{nullptr};

  if constexpr (cuda::std::execution::__queryable_with<env_t, cuda::get_stream_t>)
  {
    // Retrieve stream from the environment if present
    stream = cuda::get_stream(env).get();
  }
  else
  {
    stream = owning_stream.get();
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

  if constexpr (cuda::std::execution::__queryable_with<env_t, get_expected_allocation_size_t>)
  {
    const size_t expected_bytes_allocated = fixed_env.query(get_expected_allocation_size_t{});
    REQUIRE(expected_bytes_allocated == bytes_allocated);
  }
}

#endif // TEST_LAUNCH == 0
