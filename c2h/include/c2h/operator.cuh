// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
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
