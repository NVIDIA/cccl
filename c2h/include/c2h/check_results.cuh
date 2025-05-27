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

#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include <test_util.h>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

template <typename T>
void verify_results(const c2h::host_vector<T>& expected_data, const c2h::host_vector<T>& test_results)
{
  using namespace cub::detail;
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
  else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16> || cuda::std::is_same_v<T, __half>)
  {
    constexpr auto rel_err = cuda::std::is_same_v<T, __half> ? 0.08f : 0.14f;
    REQUIRE_APPROX_EQ_EPSILON(expected_data, test_results, rel_err);
  }
  else if constexpr (cuda::std::is_same_v<T, float2>)
  {
    for (size_t i = 0; i < test_results.size(); ++i)
    {
      REQUIRE_THAT(expected_data[i].x, Catch::Matchers::WithinRel(test_results[i].x));
      REQUIRE_THAT(expected_data[i].y, Catch::Matchers::WithinRel(test_results[i].y));
    }
  }
  else if constexpr (cuda::std::is_same_v<T, __nv_bfloat162> || cuda::std::is_same_v<T, __half2>)
  {
    constexpr auto rel_err = cuda::std::is_same_v<T, __half2> ? 0.08f : 0.14f;
    for (size_t i = 0; i < test_results.size(); ++i)
    {
      REQUIRE_THAT(expected_data[i].x, Catch::Matchers::WithinRel(test_results[i].x, rel_err));
      REQUIRE_THAT(expected_data[i].y, Catch::Matchers::WithinRel(test_results[i].y, rel_err));
    }
  }
  else if constexpr (cuda::std::is_same_v<T, cuda::std::complex<__nv_bfloat16>>
                     || cuda::std::is_same_v<T, cuda::std::complex<__half>>)
  {
    constexpr auto rel_err = cuda::std::is_same_v<T, cuda::std::complex<__half>> ? 0.08f : 0.14f;
    for (size_t i = 0; i < test_results.size(); ++i)
    {
      auto expected_real = static_cast<float>(expected_data[i].real());
      auto test_real     = test_results[i].real();
      auto expected_imag = static_cast<float>(expected_data[i].imag());
      auto test_imag     = test_results[i].imag();
      REQUIRE_THAT(expected_real, Catch::Matchers::WithinRel(test_real, rel_err));
      REQUIRE_THAT(expected_imag, Catch::Matchers::WithinRel(test_imag, rel_err));
    }
  }
  else if constexpr (cuda::std::__is_complex_v<T>)
  {
    for (size_t i = 0; i < test_results.size(); ++i)
    {
      auto expected_real = expected_data[i].real();
      auto test_real     = test_results[i].real();
      auto expected_imag = expected_data[i].imag();
      auto test_imag     = test_results[i].imag();
      REQUIRE_THAT(expected_real, Catch::Matchers::WithinRel(test_real));
      REQUIRE_THAT(expected_imag, Catch::Matchers::WithinRel(test_imag));
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

//----------------------------------------------------------------------------------------------------------------------
// Min/Max comparison requires bitwise identical results (excluding NaN). Vector Types require only the first element to
// match due to how it defined the operator<

template <typename T>
void verify_results_exact(const c2h::host_vector<T>& expected_data, const c2h::host_vector<T>& test_results)
{
  using namespace cub::detail;
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
  if constexpr (is_vector2_fp_type_v<T>)
  {
    for (size_t i = 0; i < test_results.size(); ++i)
    {
      auto expected    = static_cast<float>(expected_data[i].x);
      auto test_result = static_cast<float>(test_results[i].x);
      REQUIRE(expected == test_result);
    }
  }
  if constexpr (is_vector2_type_v<T>)
  {
    for (size_t i = 0; i < test_results.size(); ++i)
    {
      REQUIRE(expected_data[i].x == test_results[i].x);
    }
  }
  else
  {
    REQUIRE_BITWISE_EQ(expected_data, test_results);
  }
}

template <typename T>
void verify_results_exact(const c2h::host_vector<T>& expected_data, const c2h::device_vector<T>& test_results)
{
  c2h::host_vector<T> test_results_host = test_results;
  if constexpr (is_vector2_type_v<T> || cuda::is_floating_point_v<T>)
  {
    verify_results_exact(expected_data, test_results_host);
  }
  else
  {
    verify_results(expected_data, test_results_host);
  }
}
