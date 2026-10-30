// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/iterator>
#include <cuda/std/functional>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <vector_types.h>

#include <c2h/custom_type.h>
#include <c2h/extended_types.h>

// Comparing results computed on CPU and GPU for extended floating point types is impossible.
// For instance, when used with a constant iterator of two, the accumulator in sequential reference
// computation (CPU) bumps into the 4096 limits, which will never change (`4096 + 2 = 4096`).
// Meanwhile, per-thread aggregates (`2 * 16 = 32`) are accumulated within and among thread blocks,
// yielding `inf` as a result. No reasonable epsilon can be selected to compare `inf` with `4096`.
// To make `__half` and `__nv_bfloat16` arithmetic associative, the function object below raises
// extended floating points to the area of unsigned short integers. This allows us to test large
// inputs with few code-path differences in device algorithms.
struct ExtendedFloatSum
{
  template <class T>
  __host__ __device__ T operator()(T a, T b) const
  {
    T result{};
    result.__x = a.raw() + b.raw();
    return result;
  }

#if TEST_HALF_T()
  __host__ __device__ __half operator()(__half a, __half b) const
  {
    uint16_t result = this->operator()(half_t{a}, half_t(b)).raw();
    return reinterpret_cast<__half&>(result);
  }
#endif // TEST_HALF_T()

#if TEST_BF_T()
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const
  {
    uint16_t result = this->operator()(bfloat16_t{a}, bfloat16_t(b)).raw();
    return reinterpret_cast<__nv_bfloat16&>(result);
  }
#endif // TEST_BF_T()
};

template <class It>
It unwrap_it(It it)
{
  return it;
}

#if TEST_HALF_T()
inline __half* unwrap_it(half_t* it)
{
  return reinterpret_cast<__half*>(it);
}

template <class OffsetT>
cuda::constant_iterator<__half, OffsetT> unwrap_it(cuda::constant_iterator<half_t, OffsetT> it)
{
  half_t wrapped_val = *it;
  __half val         = wrapped_val.operator __half();
  return cuda::constant_iterator<__half, OffsetT>(val);
}
#endif // TEST_HALF_T()

#if TEST_BF_T()
inline __nv_bfloat16* unwrap_it(bfloat16_t* it)
{
  return reinterpret_cast<__nv_bfloat16*>(it);
}

template <class OffsetT>
cuda::constant_iterator<__nv_bfloat16, OffsetT> unwrap_it(cuda::constant_iterator<bfloat16_t, OffsetT> it)
{
  bfloat16_t wrapped_val = *it;
  __nv_bfloat16 val      = wrapped_val.operator __nv_bfloat16();
  return cuda::constant_iterator<__nv_bfloat16, OffsetT>(val);
}
#endif // TEST_BF_T()

template <typename T>
using unwrap_value_t = std::remove_reference_t<decltype(*unwrap_it(std::declval<T*>()))>;

template <class WrappedItT, //
          class ItT = decltype(unwrap_it(std::declval<WrappedItT>()))>
std::integral_constant<bool, !std::is_same_v<WrappedItT, ItT>> //
  inline reference_extended_fp(WrappedItT)
{
  return {};
}

inline constexpr ExtendedFloatSum unwrap_op(std::true_type /* extended float */, ::cuda::std::plus<>) //
{
  return {};
}

template <bool V, class OpT>
inline constexpr OpT unwrap_op(std::integral_constant<bool, V> /* base case */, OpT op)
{
  return op;
}

/**
 * @brief Initializes the given item type with a constant non-zero value.
 */
template <typename T>
inline void init_default_constant(T& val, int element_val = 2)
{
  val = T{static_cast<T>(element_val)};
}

template <template <typename> class... Policies>
inline void init_default_constant(c2h::custom_type_t<Policies...>& val, int element_val = 2)
{
  val.key = static_cast<size_t>(element_val);
  val.val = static_cast<size_t>(element_val);
}

inline void init_default_constant(uchar3& val, int element_val = 2)
{
  const auto element_init = static_cast<unsigned char>(element_val);
  val                     = uchar3{element_init, element_init, element_init};
}

_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
inline void init_default_constant(ulonglong4& val, int element_val = 2)
{
  const auto element_init = static_cast<unsigned long long>(element_val);
  val                     = ulonglong4{element_init, element_init, element_init, element_init};
}
_CCCL_SUPPRESS_DEPRECATED_POP

#if _CCCL_CTK_AT_LEAST(13, 0)
inline void init_default_constant(ulonglong4_16a& val, int element_val = 2)
{
  const auto element_init = static_cast<unsigned long long>(element_val);
  val                     = ulonglong4_16a{element_init, element_init, element_init, element_init};
}
#endif // _CCCL_CTK_AT_LEAST(13, 0)
