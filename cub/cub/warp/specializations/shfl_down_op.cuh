/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_reduce.cuh>

#include <cuda/functional>
#include <cuda/std/bit>
#include <cuda/std/cstdint>

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif // _CCCL_HAS_NVBF16()

CUB_NAMESPACE_BEGIN
namespace detail
{

//----------------------------------------------------------------------------------------------------------------------
// SHUFFLE DOWN + OP: __half, __nv_bfloat16

#define _CUB_SHFL_DOWN_OP_16BIT(OPERATOR, TYPE, PTX_OP, PTX_TYPE)                                                      \
                                                                                                                       \
  template <typename = void>                                                                                           \
  [[nodiscard]] _CCCL_DEVICE TYPE shfl_down_op(                                                                        \
    OPERATOR, TYPE value, unsigned source_offset, unsigned shfl_c, unsigned mask)                                      \
  {                                                                                                                    \
    [[maybe_unused]] int pred;                                                                                         \
    auto tmp = cub::detail::unsafe_bitcast<uint16_t>(value);                                                           \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t"  \
      ".reg .b16  dummy;                                                                                      \n\t\t"  \
      ".reg .b16  h1;                                                                                         \n\t\t"  \
      ".reg .b32  r0;                                                                                         \n\t\t"  \
      "mov.b32 r0, {%0, dummy};                                                                               \n\t\t"  \
      "shfl.sync.down.b32 r0|p, r0, %2, %3, %4;                                                               \n\t\t"  \
      "mov.b32 {h1, dummy}, r0;                                                                               \n\t\t"  \
      "@p " #PTX_OP "." #PTX_TYPE "                                                                           \n\t\t"  \
      " %0, h1, %0;                                                                                           \n\t\t"  \
      "selp.s32 %1, 1, 0, p;                                                                                    \n\t"  \
      "}"                                                                                                              \
      : "+h"(tmp), "=r"(pred)                                                                                          \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                     \
  }

//----------------------------------------------------------------------------------------------------------------------
// SHUFFLE DOWN + OP: __half2, __nv_bfloat162

#define _CUB_SHFL_DOWN_OP_16BIT_X2(OPERATOR, TYPE, PTX_OP, PTX_TYPE, PTX_REG_TYPE)                                     \
                                                                                                                       \
  template <typename = void>                                                                                           \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op(                                                      \
    OPERATOR, TYPE value, unsigned source_offset, unsigned shfl_c, unsigned mask)                                      \
  {                                                                                                                    \
    [[maybe_unused]] int pred;                                                                                         \
    auto tmp = cub::detail::unsafe_bitcast<unsigned>(value);                                                           \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t"  \
      ".reg .b32  r0;                                                                                         \n\t\t"  \
      "shfl.sync.down.b32 r0|p, %0, %2, %3, %4;                                                               \n\t\t"  \
      "@p " #PTX_OP "." #PTX_TYPE "                                                                           \n\t\t"  \
      " %0, r0, %0;                                                                                           \n\t\t"  \
      "selp.s32 %1, 1, 0, p;                                                                                    \n\t"  \
      "}"                                                                                                              \
      : "+" #PTX_REG_TYPE(tmp), "=r"(pred)                                                                             \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                     \
  }

//----------------------------------------------------------------------------------------------------------------------
// SHUFFLE DOWN + OP: float, int

#define _CUB_SHFL_DOWN_OP_32BIT(OPERATOR, TYPE, PTX_OP, PTX_TYPE, PTX_REG_TYPE)                                       \
                                                                                                                      \
  template <typename = void>                                                                                          \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op(                                                     \
    OPERATOR, TYPE value, unsigned source_offset, unsigned shfl_c, unsigned mask)                                     \
  {                                                                                                                   \
    [[maybe_unused]] int pred;                                                                                        \
    asm volatile(                                                                                                     \
      "{                                                                                                      \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t" \
      ".reg .b32  r0;                                                                                         \n\t\t" \
      "shfl.sync.down.b32 r0|p, %0, %2, %3, %4;                                                               \n\t\t" \
      "@p " #PTX_OP "." #PTX_TYPE "                                                                           \n\t\t" \
      " %0, r0, %0;                                                                                           \n\t\t" \
      "selp.s32 %1, 1, 0, p;                                                                                    \n\t" \
      "}"                                                                                                             \
      : "+" #PTX_REG_TYPE(value), "=r"(pred)                                                                          \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                  \
    return value;                                                                                                     \
  }

// add
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<float>, float, add, f32, f) // shfl_down_add(float)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<unsigned>, unsigned, add, u32, r) // shfl_down_add(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<int>, int, add, s32, r) // shfl_down_add(int)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, float, add, f32, f) // shfl_down_add(float)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, unsigned, add, u32, r) // shfl_down_add(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, int, add, s32, r) // shfl_down_add(int)

#if _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 530
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<>, __half, add, f16) // shfl_down_add(__half)
_CUB_SHFL_DOWN_OP_16BIT_X2(_CUDA_VSTD::plus<>, __half2, add, f16x2, r) // shfl_down_add(__half2)
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<__half>, __half, add, f16) // shfl_down_add(__half)
_CUB_SHFL_DOWN_OP_16BIT_X2(_CUDA_VSTD::plus<__half2>, __half2, add, f16x2, r) // shfl_down_add(__half2)
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<>, __nv_bfloat16, add, bf16) // shfl_down_add(__nv_bfloat16)
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<__nv_bfloat16>, __nv_bfloat16, add, bf16) // shfl_down_add(__nv_bfloat16)
_CUB_SHFL_DOWN_OP_16BIT_X2(_CUDA_VSTD::plus<>, __nv_bfloat162, add, bf16x2, r) // shfl_down_add(__nv_bfloat162)
_CUB_SHFL_DOWN_OP_16BIT_X2(
  _CUDA_VSTD::plus<__nv_bfloat162>, __nv_bfloat162, add, bf16x2, r) // shfl_down_add(__nv_bfloat162)
#endif // _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900

// min/max
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, int, max, s32, r) // shfl_down_max(int)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, unsigned, max, u32, r) // shfl_down_max(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, int, min, s32, r) // shfl_down_min(int)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, unsigned, min, u32, r) // shfl_down_min(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<int>, int, max, s32, r) // shfl_down_max(int)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<unsigned>, unsigned, max, u32, r) // shfl_down_max(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<int>, int, min, s32, r) // shfl_down_min(int)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<unsigned>, unsigned, min, u32, r) // shfl_down_min(unsigned)
// bitwise
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_and<>, unsigned, and, b32, r) // shfl_down_and(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_or<>, unsigned, or, b32, r) // shfl_down_or(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_xor<>, unsigned, xor, b32, r) // shfl_down_xor(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_and<unsigned>, unsigned, and, b32, r) // shfl_down_and(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_or<unsigned>, unsigned, or, b32, r) // shfl_down_or(unsigned)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_xor<unsigned>, unsigned, xor, b32, r) // shfl_down_xor(unsigned)
#undef _CUB_SHFL_DOWN_OP_32BIT

//----------------------------------------------------------------------------------------------------------------------
// SHUFFLE DOWN + OP: double

#define _CUB_SHFL_OP_64BIT(OPERATOR, TYPE, PTX_OP, PTX_TYPE)                                                           \
                                                                                                                       \
  template <typename = void>                                                                                           \
  [[nodiscard]] _CCCL_DEVICE TYPE shfl_down_op(                                                                        \
    OPERATOR, TYPE value, unsigned source_offset, unsigned shfl_c, unsigned mask)                                      \
  {                                                                                                                    \
    [[maybe_unused]] int pred;                                                                                         \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t"  \
      ".reg .u32 lo;                                                                                          \n\t\t"  \
      ".reg .u32 hi;                                                                                          \n\t\t"  \
      ".reg ." #PTX_TYPE " r1;                                                                                \n\t\t"  \
      "mov.b64 {lo, hi}, %0;                                                                                  \n\t\t"  \
      "shfl.sync.down.b32 lo,   lo, %2, %3, %4;                                                               \n\t\t"  \
      "shfl.sync.down.b32 hi|p, hi, %2, %3, %4;                                                               \n\t\t"  \
      "mov.b64 r1, {lo, hi};                                                                                  \n\t\t"  \
      "@p " #PTX_OP "." #PTX_TYPE                                                                                      \
      " %0, r1, %0;                                                                   \n\t\t"                          \
      "selp.s32 %1, 1, 0, p;                                                                                    \n\t"  \
      "}"                                                                                                              \
      : "+d"(value), "=r"(pred)                                                                                        \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
    return value;                                                                                                      \
  }

_CUB_SHFL_OP_64BIT(_CUDA_VSTD::plus<>, double, add, f64) // shfl_down_add (double)
_CUB_SHFL_OP_64BIT(_CUDA_VSTD::plus<double>, double, add, f64) // shfl_down_add (double)
#undef _CUB_SHFL_OP_64BIT

} // namespace detail
CUB_NAMESPACE_END
