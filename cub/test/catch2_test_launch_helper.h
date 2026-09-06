// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/__algorithm/copy.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include <c2h/catch2_test_macros.h>
#include <c2h/checked_memory_resource.cuh>
#include <catch2/generators/catch_generators_all.hpp>

//! @file
//! This file contains utilities for device-scope API tests
//!
//! Device-scope API in CUB can be launched from the host or device side.
//! Utilities in this file facilitate testing in both cases.
//!
//!
//! ```
//! // Add PARAM to make CMake generate a test for both host and device launch:
//! // %PARAM% TEST_LAUNCH lid 0:1
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
//!   // Tests with stream-ordered setup can pass a stream as the first argument.
//!   cub_reduce_sum(stream, d_in, d_out, n, should_be_invoked_on_device);
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
//! The stream-aware wrapper overload uses the caller-provided stream for host and
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

#define DECLARE_LAUNCH_WRAPPER(API, WRAPPED_API_NAME)                                                               \
  DECLARE_INVOCABLE(API, WRAPPED_API_NAME, , );                                                                     \
  [[maybe_unused]] inline constexpr struct WRAPPED_API_NAME##_t                                                     \
  {                                                                                                                 \
    template <class Stream, class... As>                                                                            \
    ::cuda::std::enable_if_t<launch_helper_detail::is_stream_argument<Stream>::value>                               \
    operator()(Stream&& stream, As... args) const                                                                   \
    {                                                                                                               \
      launch(::cuda::stream_ref{stream}, WRAPPED_API_NAME##_invocable_t{}, args...);                                \
    }                                                                                                               \
                                                                                                                    \
    template <class... As>                                                                                          \
    ::cuda::std::enable_if_t<!launch_helper_detail::first_arg_is_stream<As...>::value> operator()(As... args) const \
    {                                                                                                               \
      launch(WRAPPED_API_NAME##_invocable_t{}, args...);                                                            \
    }                                                                                                               \
  } WRAPPED_API_NAME

#define ESCAPE_LIST(...) __VA_ARGS__

namespace launch_helper_detail
{
template <class T>
using remove_cvref_t = ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<T>>;

template <class T>
struct is_stream_argument;

template <>
struct is_stream_argument<cudaStream_t> : ::cuda::std::true_type
{};

template <typename T>
struct is_stream_argument<T*> : ::cuda::std::false_type
{};

template <typename T>
struct is_stream_argument : ::cuda::std::is_convertible<remove_cvref_t<T>, ::cuda::stream_ref>
{};

template <class...>
struct first_arg_is_stream : ::cuda::std::false_type
{};

template <class First, class... Rest>
struct first_arg_is_stream<First, Rest...> : is_stream_argument<First>
{};

template <typename... As>
struct first_arg_is_stream<cudaStream_t, As...> : ::cuda::std::true_type
{};

template <typename T, typename... As>
struct first_arg_is_stream<T*, As...> : ::cuda::std::false_type
{};

template <typename... As>
struct first_arg_is_stream<::cuda::stream_ref, As...> : ::cuda::std::true_type
{};

inline cuda::device_ref current_device()
{
  int device{0};
  REQUIRE(cudaSuccess == cudaGetDevice(&device));
  return cuda::device_ref{device};
}

inline cuda::device_ref device_for_stream(cuda::stream_ref stream)
{
  if (stream == ::cudaStream_t{})
  {
    return current_device();
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
#define DECLARE_TMPL_LAUNCH_WRAPPER(API, WRAPPED_API_NAME, TMPL_PARAMS, TMPL_ARGS)                            \
  DECLARE_INVOCABLE(API, WRAPPED_API_NAME, ESCAPE_LIST(template <TMPL_PARAMS>), ESCAPE_LIST(<TMPL_ARGS>));    \
  template <TMPL_PARAMS, class Stream, class... As>                                                           \
  static ::cuda::std::enable_if_t<launch_helper_detail::is_stream_argument<Stream>::value> WRAPPED_API_NAME(  \
    Stream&& stream, As... args)                                                                              \
  {                                                                                                           \
    launch(::cuda::stream_ref{stream}, WRAPPED_API_NAME##_invocable_t<TMPL_ARGS>{}, args...);                 \
  }                                                                                                           \
  template <TMPL_PARAMS, class... As>                                                                         \
  static ::cuda::std::enable_if_t<!launch_helper_detail::first_arg_is_stream<As...>::value> WRAPPED_API_NAME( \
    As... args)                                                                                               \
  {                                                                                                           \
    launch(WRAPPED_API_NAME##_invocable_t<TMPL_ARGS>{}, args...);                                             \
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
  const auto device = launch_helper_detail::current_device();
  auto stream       = cuda::stream{device};
  launch(cuda::stream_ref{stream}, action, args...);
}

#elif TEST_LAUNCH == 0

template <class ActionT, class... Args>
void launch(cuda::stream_ref stream, ActionT action, Args... args)
{
  const auto device = launch_helper_detail::device_for_stream(stream);
  const launch_helper_detail::scoped_current_device device_scope{device};

  cuda::std::size_t temp_storage_bytes{};
  cudaError_t error = action(nullptr, temp_storage_bytes, args..., stream.get());
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  launch_helper_detail::synchronize(stream);
  REQUIRE(cudaSuccess == error);

  REQUIRE(temp_storage_bytes > 0); // required by API contract

  // randomly offset the temporary storage address by one byte
  const int offset = GENERATE(take(1, random(0, 1)));
  auto temp_storage =
    c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, temp_storage_bytes + offset, cuda::no_init);

  error = action(temp_storage.data() + offset, temp_storage_bytes, args..., stream.get());
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  launch_helper_detail::synchronize(stream);
  REQUIRE(cudaSuccess == error);
}

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
  const int offset  = GENERATE(take(1, random(0, 1)));
  const auto device = launch_helper_detail::current_device();
  auto stream       = cuda::stream{device};
  auto temp_storage =
    c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, temp_storage_bytes + offset, cuda::no_init);

  error = action(temp_storage.data() + offset, temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);
}
#else // TEST_LAUNCH == 2
#  error "Unsupported TEST_LAUNCH value. Supported values are 0, 1, or 2"
#endif // TEST_LAUNCH == 2
