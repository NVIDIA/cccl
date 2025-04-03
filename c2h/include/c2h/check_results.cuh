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

#include <cuda/std/__complex/is_complex.h>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cmath>

#include "c2h/catch2_test_helper.h"

template <typename T>
inline constexpr bool is_any_bfloat16_v = false;

template <>
inline constexpr bool is_any_bfloat16_v<__nv_bfloat16> = true;

template <>
inline constexpr bool is_any_bfloat16_v<__nv_bfloat162> = true;

template <>
inline constexpr bool is_any_bfloat16_v<cuda::std::complex<__nv_bfloat16>> = true;

template <typename T>
inline constexpr bool is_any_half_v = false;

template <>
inline constexpr bool is_any_half_v<__half> = true;

template <>
inline constexpr bool is_any_half_v<__half2> = true;

template <>
inline constexpr bool is_any_half_v<cuda::std::complex<__half>> = true;

/**
 * @brief Compares the results returned from system under test against the expected results.
 */
// TODO: Replace this function by REQUIRE_APPROX_EQ once it supports integral vector types like short2
template <typename T>
void verify_results(const c2h::host_vector<T>& expected_data, const c2h::host_vector<T>& test_results)
{
  int device_id          = 0;
  int compute_capability = 0;
  CubDebugExit(cudaGetDevice(&device_id));
  CubDebugExit(cudaDeviceGetAttribute(&compute_capability, cudaDevAttrComputeCapabilityMajor, device_id));
  CubDebugExit(cudaDeviceGetAttribute(&compute_capability, cudaDevAttrComputeCapabilityMinor, device_id));
  if (compute_capability < 90 && is_any_bfloat16_v<T>)
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
  else if constexpr (cuda::std::__is_extended_floating_point_v<T>)
  {
    REQUIRE_APPROX_EQ_EPSILON(expected_data, test_results, 0.05f);
  }
  else if constexpr (cuda::std::__is_complex_v<T>)
  {
    using value_type  = typename cuda::std::remove_cv_t<T>::value_type;
    using promotion_t = _CUDA_VSTD::_If<sizeof(value_type) <= sizeof(float), float, double>;
    c2h::host_vector<value_type> test_results_real(test_results.size());
    c2h::host_vector<value_type> expected_data_real(expected_data.size());
    c2h::host_vector<value_type> test_results_img(test_results.size());
    c2h::host_vector<value_type> expected_data_img(expected_data.size());
    // zip_iterator does not work with complex<__half>
    std::transform(test_results.begin(), test_results.end(), test_results_real.begin(), [](T x) {
      return x.real();
    });
    std::transform(test_results.begin(), test_results.end(), test_results_img.begin(), [](T x) {
      return x.imag();
    });
    std::transform(expected_data.begin(), expected_data.end(), expected_data_real.begin(), [](T x) {
      return x.real();
    });
    std::transform(expected_data.begin(), expected_data.end(), expected_data_img.begin(), [](T x) {
      return x.imag();
    });
    verify_results(test_results_real, expected_data_real);
    verify_results(test_results_img, expected_data_img);
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
