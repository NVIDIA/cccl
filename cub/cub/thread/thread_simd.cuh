// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file
 * Simple binary operator functor types
 */

/******************************************************************************
 * Simple functor operators
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_operators.cuh>

#include <cuda/functional> // cuda::maximum, cuda::minimum
#include <cuda/std/cstdint>
#include <cuda/std/type_traits> // cuda::std::common_type

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVBF16()

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/***********************************************************************************************************************
 * SIMD operators
 **********************************************************************************************************************/

namespace detail
{

template <typename T>
extern _CCCL_HOST_DEVICE T simd_operation_is_not_supported_before_sm80();

template <typename T>
extern _CCCL_HOST_DEVICE T simd_operation_is_not_supported_before_sm53();

template <typename T>
struct SimdMin
{
  static_assert(::cuda::std::__always_false_v<T>, "Unsupported specialization");
};

template <>
struct SimdMin<short2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE short2 operator()(short2 a, short2 b) const
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (auto a1 = ::cuda::std::bit_cast<uint32_t>(a); //
       auto b1 = ::cuda::std::bit_cast<uint32_t>(b);
       return ::cuda::std::bit_cast<short2>(::__vmins2(a1, b1));),
      (return short2{::cuda::minimum<>{}(a.x, b.x), ::cuda::minimum<>{}(a.y, b.y)};))
  }
};

template <>
struct SimdMin<ushort2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ushort2 operator()(ushort2 a, ushort2 b) const
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (auto a1 = ::cuda::std::bit_cast<uint32_t>(a); //
       auto b1 = ::cuda::std::bit_cast<uint32_t>(b);
       return ::cuda::std::bit_cast<ushort2>(::__vminu2(a1, b1));),
      (return ushort2{::cuda::minimum<>{}(a.x, b.x), ::cuda::minimum<>{}(a.y, b.y)};))
  }
};

#  if _CCCL_HAS_NVFP16()

template <>
struct SimdMin<__half2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    return ::__hmin2(a, b);
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct SimdMin<__nv_bfloat162>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                      (return ::__hmin2(a, b);),
                      (return simd_operation_is_not_supported_before_sm80<__nv_bfloat162>();))
  }
};

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdMax
{
  static_assert(::cuda::std::__always_false_v<T>, "Unsupported specialization");
};

template <>
struct SimdMax<short2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE short2 operator()(short2 a, short2 b) const
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (auto a1 = ::cuda::std::bit_cast<uint32_t>(a); //
       auto b1 = ::cuda::std::bit_cast<uint32_t>(b);
       return ::cuda::std::bit_cast<short2>(::__vmaxs2(a1, b1));),
      (return short2{::cuda::maximum<>{}(a.x, b.x), ::cuda::maximum<>{}(a.y, b.y)};))
  }
};

template <>
struct SimdMax<ushort2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ushort2 operator()(ushort2 a, ushort2 b) const
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (auto a1 = ::cuda::std::bit_cast<uint32_t>(a); //
       auto b1 = ::cuda::std::bit_cast<uint32_t>(b);
       return ::cuda::std::bit_cast<ushort2>(::__vmaxu2(a1, b1));),
      (return ushort2{::cuda::maximum<>{}(a.x, b.x), ::cuda::maximum<>{}(a.y, b.y)};))
  }
};

#  if _CCCL_HAS_NVFP16()

template <>
struct SimdMax<__half2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    return ::__hmax2(a, b);
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct SimdMax<__nv_bfloat162>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                      (return ::__hmax2(a, b);), //
                      (return simd_operation_is_not_supported_before_sm80<__nv_bfloat162>();))
  }
};

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdSum
{
  static_assert(::cuda::std::__always_false_v<T>, "Unsupported specialization");
};

#  if _CCCL_HAS_NVFP16()

template <>
struct SimdSum<__half2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
                      (return ::__hadd2(a, b);), //
                      (return simd_operation_is_not_supported_before_sm53<__half2>();))
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct SimdSum<__nv_bfloat162>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                      (return ::__hadd2(a, b);), //
                      (return simd_operation_is_not_supported_before_sm80<__nv_bfloat162>();))
  }
};

#  endif // _CCCL_HAS_NVBF16()

template <>
struct SimdSum<ushort2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ushort2 operator()(ushort2 a, ushort2 b) const
  {
    // clang-format off
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                     (uint32_t ret;
                      asm("add.u16x2 %0, %1, %2;"
                          : "=r"(ret)
                          : "r"(::cuda::std::bit_cast<uint32_t>(a)),
                            "r"(::cuda::std::bit_cast<uint32_t>(b)));
                      return ::cuda::std::bit_cast<ushort2>(ret);),
                     (return ushort2{static_cast<uint16_t>(a.x + b.x), static_cast<uint16_t>(a.y + b.y)};));
    // clang-format on
  }
};

template <>
struct SimdSum<short2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE short2 operator()(short2 a, short2 b) const
  {
    // clang-format off
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                     (uint32_t ret;
                      asm("add.s16x2 %0, %1, %2;"
                          : "=r"(ret)
                          : "r"(::cuda::std::bit_cast<uint32_t>(a)),
                            "r"(::cuda::std::bit_cast<uint32_t>(b)));
                      return ::cuda::std::bit_cast<short2>(ret);),
                     (return short2{static_cast<int16_t>(a.x + b.x), static_cast<int16_t>(a.y + b.y)};));
    // clang-format on
  }
};

template <>
struct SimdSum<float2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE float2 operator()(float2 a, float2 b) const
  {
    // clang-format off
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_100,
                     (uint64_t ret;
                      asm("add.f32x2 %0, %1, %2;"
                           : "=l"(ret)
                           : "l"(::cuda::std::bit_cast<uint64_t>(a)),
                             "l"(::cuda::std::bit_cast<uint64_t>(b)));
                      return ::cuda::std::bit_cast<float2>(ret);),
                     (return float2{a.x + b.x,  a.y + b.y};))
    // clang-format on
  }
};

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdMul
{
  static_assert(::cuda::std::__always_false_v<T>, "Unsupported specialization");
};

#  if _CCCL_HAS_NVFP16()

template <>
struct SimdMul<__half2>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53,
                      (return ::__hmul2(a, b);), //
                      (return simd_operation_is_not_supported_before_sm53<__half2>();))
  }
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct SimdMul<__nv_bfloat162>
{
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                      (return ::__hmul2(a, b);),
                      (return simd_operation_is_not_supported_before_sm80<__nv_bfloat162>();))
  }
};

#  endif // _CCCL_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------

template <typename ReductionOp>
inline constexpr bool is_simd_operator_v = false;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdSum<T>> = true;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdMul<T>> = true;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdMin<T>> = true;

template <typename T>
inline constexpr bool is_simd_operator_v<SimdMax<T>> = true;

//----------------------------------------------------------------------------------------------------------------------
// SIMD type

template <typename T>
struct VectorTypeX2
{
  static_assert(::cuda::std::__always_false_v<T>, "Unsupported specialization");
};

template <>
struct VectorTypeX2<int16_t>
{
  using type = short2;
};

template <>
struct VectorTypeX2<uint16_t>
{
  using type = ushort2;
};

#  if _CCCL_HAS_NVFP16()

template <>
struct VectorTypeX2<__half>
{
  using type = __half2;
};

#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()

template <>
struct VectorTypeX2<__nv_bfloat16>
{
  using type = __nv_bfloat162;
};

#  endif // _CCCL_HAS_NVBF16()

template <typename T>
using vector_type_x2_t = typename VectorTypeX2<T>::type;

//----------------------------------------------------------------------------------------------------------------------
// Predefined CUDA operators to SIMD

template <typename ReduceOp, typename T>
struct CudaOperatorToSimdX2
{
  static_assert(::cuda::std::__always_false_v<T>, "Unsupported specialization");
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::minimum<>, T>
{
  using type = SimdMin<vector_type_x2_t<T>>;
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::minimum<T>, T>
{
  using type = SimdMin<vector_type_x2_t<T>>;
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::maximum<>, T>
{
  using type = SimdMax<vector_type_x2_t<T>>;
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::maximum<T>, T>
{
  using type = SimdMax<vector_type_x2_t<T>>;
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::std::plus<>, T>
{
  using type = SimdSum<vector_type_x2_t<T>>;
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::std::plus<T>, T>
{
  using type = SimdSum<vector_type_x2_t<T>>;
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::std::multiplies<>, T>
{
  using type = SimdMul<vector_type_x2_t<T>>;
};

template <typename T>
struct CudaOperatorToSimdX2<::cuda::std::multiplies<T>, T>
{
  using type = SimdMul<vector_type_x2_t<T>>;
};

template <typename ReduceOp, typename T>
using cuda_operator_to_simd_x2_t = typename CudaOperatorToSimdX2<ReduceOp, T>::type;

//----------------------------------------------------------------------------------------------------------------------

template <typename T, typename Op>
_CCCL_DEVICE _CCCL_FORCEINLINE auto try_simd_operator(Op op)
{
  using ::cuda::std::is_same_v;
  constexpr bool is_supported_vector_type = is_any_short2_v<T> || is_bfloat162_v<T> || is_half2_v<T>;
  if constexpr (is_cuda_std_plus_v<Op> && (is_same_v<T, float2> || is_supported_vector_type))
  {
    return SimdSum<T>{};
  }
  else if constexpr (is_cuda_minimum_v<Op> && is_supported_vector_type)
  {
    return SimdMin<T>{};
  }
  else if constexpr (is_cuda_maximum_v<Op> && is_supported_vector_type)
  {
    return SimdMax<T>{};
  }
  else
  {
    return op;
  }
}

} // namespace detail

#endif // !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
