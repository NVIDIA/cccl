/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
 * disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 * following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
 * products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************************************/
#pragma once

#include <cub/thread/thread_operators.cuh>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include <c2h/custom_type.h>
#include <c2h/test_util_vec.h>

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

template <typename Operator, typename T>
inline constexpr T identity_v = cub::detail::identity_v<Operator, T>;

template <typename T>
inline const T identity_v<cuda::std::plus<>, T> = T{}; // e.g. short2, float2, complex<__half> etc.

template <>
inline constexpr short2 identity_v<cuda::maximum<>, short2> =
  short2{cuda::std::numeric_limits<int16_t>::lowest(), cuda::std::numeric_limits<int16_t>::lowest()};

template <>
inline constexpr ushort2 identity_v<cuda::maximum<>, ushort2> = ushort2{0, 0};

template <>
inline constexpr float2 identity_v<cuda::maximum<>, float2> =
  float2{cuda::std::numeric_limits<float>::lowest(), cuda::std::numeric_limits<float>::lowest()};

template <>
inline const __half2 identity_v<cuda::maximum<>, __half2> =
  __half2{cuda::std::numeric_limits<__half>::lowest(), cuda::std::numeric_limits<__half>::lowest()};

template <>
inline const __nv_bfloat162 identity_v<cuda::maximum<>, __nv_bfloat162> = __nv_bfloat162{
  cuda::std::numeric_limits<__nv_bfloat16>::lowest(), cuda::std::numeric_limits<__nv_bfloat16>::lowest()};

template <>
inline constexpr short2 identity_v<cuda::minimum<>, short2> =
  short2{cuda::std::numeric_limits<int16_t>::max(), cuda::std::numeric_limits<int16_t>::max()};

template <>
inline constexpr ushort2 identity_v<cuda::minimum<>, ushort2> =
  ushort2{cuda::std::numeric_limits<uint16_t>::max(), cuda::std::numeric_limits<uint16_t>::max()};

template <>
inline const __half2 identity_v<cuda::minimum<>, __half2> =
  __half2{cuda::std::numeric_limits<__half>::max(), cuda::std::numeric_limits<__half>::max()};

template <>
inline const __nv_bfloat162 identity_v<cuda::minimum<>, __nv_bfloat162> =
  __nv_bfloat162{cuda::std::numeric_limits<__nv_bfloat16>::max(), cuda::std::numeric_limits<__nv_bfloat16>::max()};

template <template <typename> class... Policies>
inline const c2h::custom_type_t<Policies...> identity_v<cuda::maximum<>, c2h::custom_type_t<Policies...>> =
  cuda::std::numeric_limits<c2h::custom_type_t<Policies...>>::lowest();

template <template <typename> class... Policies>
inline const c2h::custom_type_t<Policies...> identity_v<cuda::minimum<>, c2h::custom_type_t<Policies...>> =
  cuda::std::numeric_limits<c2h::custom_type_t<Policies...>>::max();

struct custom_plus : cuda::std::plus<>
{};
