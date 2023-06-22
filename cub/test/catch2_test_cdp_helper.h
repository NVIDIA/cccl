/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <thrust/device_vector.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include "catch2_test_helper.h"

//! @file This file contains utilities for device-scope API tests
//!
//! Device-scope API in CUB can be launched from the host or device side.
//! Utilities in this file facilitate testing in both cases.
//!
//!
//! ```
//! // Add PARAM to make CMake generate a test for both host and device launch:
//! // %PARAM% TEST_CDP cdp 0:1
//!
//! // Declare CDP wrapper for CUB API. The wrapper will accept the same
//! // arguments as the CUB API. The wrapper name is provided as the second argument.
//! DECLARE_CDP_WRAPPER(cub::DeviceReduce::Sum, cub_reduce_sum);
//!
//! CUB_TEST("Reduce test", "[device][reduce]")
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
//! Consult with `test/catch2_test_cdp_wrapper.cu` for more usage examples.

#if !defined(TEST_CDP)
#error Test file should contain %PARAM% TEST_CDP cdp 0:1
#endif

#if TEST_CDP == 1
template <class ActionT, class... Args>
__global__ void device_side_api_launch_kernel(std::uint8_t *d_temp_storage,
                                              std::size_t *temp_storage_bytes,
                                              cudaError_t *d_error,
                                              ActionT action,
                                              Args... args)
{
  *d_error = action(d_temp_storage, *temp_storage_bytes, args...);
}

template <class ActionT, class... Args>
void device_side_api_launch(ActionT action, Args... args)
{
  std::uint8_t *d_temp_storage = nullptr;
  thrust::device_vector<cudaError_t> d_error(1, cudaErrorInvalidValue);
  thrust::device_vector<std::size_t> d_temp_storage_bytes(1, 0);
  device_side_api_launch_kernel<<<1, 1>>>(d_temp_storage,
                                          thrust::raw_pointer_cast(d_temp_storage_bytes.data()),
                                          thrust::raw_pointer_cast(d_error.data()),
                                          action,
                                          args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == d_error[0]);

  thrust::device_vector<std::uint8_t> temp_storage(d_temp_storage_bytes[0]);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  device_side_api_launch_kernel<<<1, 1>>>(d_temp_storage,
                                          thrust::raw_pointer_cast(d_temp_storage_bytes.data()),
                                          thrust::raw_pointer_cast(d_error.data()),
                                          action,
                                          args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == d_error[0]);
}

#define cdp_launch device_side_api_launch

#define DECLARE_CDP_INVOCABLE(API, WRAPPED_API_NAME)                                               \
  struct WRAPPED_API_NAME##_device_invocable_t                                                     \
  {                                                                                                \
    template <class... Ts>                                                                         \
    CUB_RUNTIME_FUNCTION cudaError_t operator()(std::uint8_t *d_temp_storage,                      \
                                                std::size_t &temp_storage_bytes,                   \
                                                Ts... args)                                        \
    {                                                                                              \
      return API(d_temp_storage, temp_storage_bytes, args...);                                     \
    }                                                                                              \
  };                                                                                               

#define DECLARE_CDP_WRAPPER(API, WRAPPED_API_NAME)                                                 \
  DECLARE_CDP_INVOCABLE(API, WRAPPED_API_NAME);                                                    \
  template <class... As>                                                                           \
  static void WRAPPED_API_NAME(As... args)                                                         \
  {                                                                                                \
    cdp_launch(WRAPPED_API_NAME##_device_invocable_t{}, args...);                                  \
  }                                                                                                

#else

template <class ActionT, class... Args>
void host_side_api_launch(ActionT action, Args... args)
{
  std::uint8_t *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes{};
  cudaError_t error = action(d_temp_storage, temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  error = action(d_temp_storage, temp_storage_bytes, args...);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(cudaSuccess == error);
}

#define cdp_launch host_side_api_launch

#define DECLARE_CDP_INVOCABLE(API, WRAPPED_API_NAME)                                               \
  struct WRAPPED_API_NAME##_host_invocable_t                                                       \
  {                                                                                                \
    template <class... Ts>                                                                         \
    CUB_RUNTIME_FUNCTION cudaError_t operator()(std::uint8_t *d_temp_storage,                      \
                                                std::size_t &temp_storage_bytes,                   \
                                                Ts... args)                                        \
    {                                                                                              \
      return API(d_temp_storage, temp_storage_bytes, args...);                                     \
    }                                                                                              \
  };                                                                                               

#define DECLARE_CDP_WRAPPER(API, WRAPPED_API_NAME)                                                 \
  DECLARE_CDP_INVOCABLE(API, WRAPPED_API_NAME);                                                    \
  template <class... As>                                                                           \
  static void WRAPPED_API_NAME(As... args)                                                         \
  {                                                                                                \
    cdp_launch(WRAPPED_API_NAME##_host_invocable_t{}, args...);                                    \
  }                                                                                                

#endif
