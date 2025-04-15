/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/detail/type_traits.cuh>

#include <cuda/std/__complex/is_complex.h>
#include <cuda/std/type_traits>

#include "c2h/catch2_test_helper.h"

/**
 * @brief Compares the results returned from system under test against the expected results.
 */
// TODO: Replace this function by REQUIRE_APPROX_EQ once it supports integral vector types like short2
template <typename T>
void verify_results(const c2h::host_vector<T>& expected_data, const c2h::host_vector<T>& test_results)
{
  using namespace cub::internal;
  int device_id                = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  CubDebugExit(cudaGetDevice(&device_id));
  CubDebugExit(cudaDeviceGetAttribute(&compute_capability_major, cudaDevAttrComputeCapabilityMajor, device_id));
  CubDebugExit(cudaDeviceGetAttribute(&compute_capability_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  int compute_capability = 10 * compute_capability_major + compute_capability_minor;
  if (compute_capability < 80 && is_any_bfloat16_v<T>)
  {
    return;
  }
  if (compute_capability < 53 && is_any_half_v<T>)
  {
    return;
  }
  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    REQUIRE_APPROX_EQ(expected_data, test_results);
  }
  else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>)
  {
    REQUIRE_APPROX_EQ_EPSILON(expected_data, test_results, 0.11f);
  }
  else if constexpr (cuda::std::is_same_v<T, __half>)
  {
    REQUIRE_APPROX_EQ_EPSILON(expected_data, test_results, 0.05f);
  }
  else if constexpr (cuda::std::__is_complex_v<T> || cuda::std::is_same_v<T, float2>)
  {
    for (size_t i = 0; i < test_results.size(); ++i)
    {
      if constexpr (cuda::std::is_floating_point_v<T>)
      {
        REQUIRE_APPROX_EQ(expected_data[i], test_results[i]);
      }
      else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>)
      {
        REQUIRE_APPROX_EQ_EPSILON(expected_data, test_results, 0.11f);
      }
      else if constexpr (cuda::std::is_same_v<T, __half>)
      {
        REQUIRE_APPROX_EQ_EPSILON(expected_data[i], test_results[i], 0.05f);
      }
    }
  }
  else
  {
    REQUIRE(expected_data == test_results);
  }
}

template <typename T>
void verify_results(const c2h::host_vector<T>& expected_data, const c2h::device_vector<T>& test_results)
{
  c2h::host_vector<T> test_results_host = test_results;
  verify_results(expected_data, test_results_host);
}
