// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <c2h/catch2_test_helper.h>

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
//! C2H_TEST("Reduce test", "[device][reduce]")
//! {
//!   // ...
//!   // Invoke the wrapper from the test. It'll allocate temporary storage and
//!   // invoke the CUB API on the host or device side while checking return
//!   // codes and launch errors.
//!   cub_reduce_sum(d_in, d_out, n, should_be_invoked_on_device);
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
//! If the wrapped API contains default parameters before stream, you'd want to explicitly
//! specify those at all invocations.
//!
//! Consult with `test/catch2_test_launch_wrapper.cu` for more usage examples.

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
  *d_error = action(d_temp_storage, *temp_storage_bytes, args...);
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

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);

  error = action(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);
}

#endif // TEST_LAUNCH == 0
