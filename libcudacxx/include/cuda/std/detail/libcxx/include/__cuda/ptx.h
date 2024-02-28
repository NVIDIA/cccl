// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_PTX_H
#define  _LIBCUDACXX___CUDA_PTX_H

#ifndef __cuda_std__
#error "<__cuda/ptx.h> should only be included in from <cuda/std/barrier>"
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include "../__cuda/ptx/ptx_dot_variants.h"
#include "../__cuda/ptx/ptx_helper_functions.h"
#include "../__cuda/ptx/parallel_synchronization_and_communication_instructions_mbarrier.h"
#include "../cstdint" // uint32_t
/*
 * The cuda::ptx namespace intends to provide PTX wrappers for new hardware
 * features and new PTX instructions so that they can be experimented with
 * before higher-level C++ APIs are designed and developed.
 *
 * The wrappers have the following responsibilities:
 *
 * - They must prevent any PTX assembler errors, that is:
 *   - They are defined only for versions of the CUDA Toolkit in which nvcc/ptxas
 *     actually recognizes the instruction.
 *   - Sizes and types of parameters are correct.
 * - They must convert state spaces correctly.
 * - They adhere to the libcu++ coding standards of using:
 *   - Reserved identifiers for all parameters, variables. E.g. `__meow` or `_Woof`
 *   - _CUDA_VSTD:: namespace for types
 *
 * The wrappers should not do the following:
 *
 * - Use any non-native types. For example, an mbarrier instruction wrapper
 *   takes the barrier address as a uint64_t pointer.
 *
 * This header is intended for:
 *
 * - internal consumption by higher-level APIs such as cuda::barrier,
 * - outside developers who want to experiment with the latest features of the
 *   hardware.
 *
 * Stability:
 *
 * - These headers are intended to present a stable API (not ABI) within one
 *   major version of the CTK. This means that:
 *   - All functions are marked inline
 *   - The type of a function parameter can be changed to be more generic if
 *     that means that code that called the original version can still be
 *     compiled.
 *
 * - Good exposure of the PTX should be high priority. If, at a new major
 *   version, we face a difficult choice between breaking backward-compatibility
 *   and an improvement of the PTX exposure, we will tend to the latter option
 *   more easily than in other parts of libcu++.
 */

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

/*
 *  Instructions
 *
 *  The organization of the instructions below follows that of the PTX ISA documentation:
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instructions
 *
 *  To improve code organization, some sections are separated into their own
 *  header. For instance, the mbarrier instructions are found in:
 *  __cuda/ptx/parallel_synchronization_and_communication_instructions_mbarrier.h
 *
 */

/*
 *  9.7.1. Integer Arithmetic Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions
 *
 */

// 9.7.1.7. Integer Arithmetic Instructions: sad
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad

// 9.7.1.8. Integer Arithmetic Instructions: div
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div

// 9.7.1.9. Integer Arithmetic Instructions: rem
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem

// 9.7.1.10. Integer Arithmetic Instructions: abs
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs

// 9.7.1.11. Integer Arithmetic Instructions: neg
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg

// 9.7.1.12. Integer Arithmetic Instructions: min
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-min

// 9.7.1.13. Integer Arithmetic Instructions: max
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-max

// 9.7.1.14. Integer Arithmetic Instructions: popc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-popc

// 9.7.1.15. Integer Arithmetic Instructions: clz
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz

// 9.7.1.16. Integer Arithmetic Instructions: bfind
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind

// 9.7.1.17. Integer Arithmetic Instructions: fns
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns

// 9.7.1.18. Integer Arithmetic Instructions: brev
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev

// 9.7.1.19. Integer Arithmetic Instructions: bfe
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe

// 9.7.1.20. Integer Arithmetic Instructions: bfi
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi

// 9.7.1.21. Integer Arithmetic Instructions: szext
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext

// 9.7.1.22. Integer Arithmetic Instructions: bmsk
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bmsk

// 9.7.1.23. Integer Arithmetic Instructions: dp4a
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a

// 9.7.1.24. Integer Arithmetic Instructions: dp2a
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a


/*
 *  9.7.2. Extended-Precision Integer Arithmetic Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions
 *
 */

// 9.7.2.1. Extended-Precision Arithmetic Instructions: add.cc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-add-cc

// 9.7.2.2. Extended-Precision Arithmetic Instructions: addc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc

// 9.7.2.3. Extended-Precision Arithmetic Instructions: sub.cc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-sub-cc

// 9.7.2.4. Extended-Precision Arithmetic Instructions: subc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc

// 9.7.2.5. Extended-Precision Arithmetic Instructions: mad.cc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-mad-cc

// 9.7.2.6. Extended-Precision Arithmetic Instructions: madc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc


/*
 *  9.7.3. Floating-Point Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions
 *
 */

// 9.7.3.1. Floating Point Instructions: testp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-testp

// 9.7.3.2. Floating Point Instructions: copysign
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-copysign

// 9.7.3.3. Floating Point Instructions: add
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-add

// 9.7.3.4. Floating Point Instructions: sub
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sub

// 9.7.3.5. Floating Point Instructions: mul
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul

// 9.7.3.6. Floating Point Instructions: fma
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma

// 9.7.3.7. Floating Point Instructions: mad
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mad

// 9.7.3.8. Floating Point Instructions: div
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div

// 9.7.3.9. Floating Point Instructions: abs
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-abs

// 9.7.3.10. Floating Point Instructions: neg
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-neg

// 9.7.3.11. Floating Point Instructions: min
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-min

// 9.7.3.12. Floating Point Instructions: max
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max

// 9.7.3.13. Floating Point Instructions: rcp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp

// 9.7.3.14. Floating Point Instructions: rcp.approx.ftz.f64
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp-approx-ftz-f64

// 9.7.3.15. Floating Point Instructions: sqrt
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sqrt

// 9.7.3.16. Floating Point Instructions: rsqrt
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt

// 9.7.3.17. Floating Point Instructions: rsqrt.approx.ftz.f64
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt-approx-ftz-f64

// 9.7.3.18. Floating Point Instructions: sin
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin

// 9.7.3.19. Floating Point Instructions: cos
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-cos

// 9.7.3.20. Floating Point Instructions: lg2
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2

// 9.7.3.21. Floating Point Instructions: ex2
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2

// 9.7.3.22. Floating Point Instructions: tanh
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh


/*
 *  9.7.4. Half Precision Floating-Point Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions
 *
 */

// 9.7.4.1. Half Precision Floating Point Instructions: add
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-add

// 9.7.4.2. Half Precision Floating Point Instructions: sub
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-sub

// 9.7.4.3. Half Precision Floating Point Instructions: mul
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul

// 9.7.4.4. Half Precision Floating Point Instructions: fma
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma

// 9.7.4.5. Half Precision Floating Point Instructions: neg
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-neg

// 9.7.4.6. Half Precision Floating Point Instructions: abs
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-abs

// 9.7.4.7. Half Precision Floating Point Instructions: min
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-min

// 9.7.4.8. Half Precision Floating Point Instructions: max
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-max

// 9.7.4.9. Half Precision Floating Point Instructions: tanh
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-tanh

// 9.7.4.10. Half Precision Floating Point Instructions: ex2
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-ex2


/*
 *  9.7.5. Comparison and Selection Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions
 *
 */

// 9.7.5.1. Comparison and Selection Instructions: set
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-set

// 9.7.5.2. Comparison and Selection Instructions: setp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp

// 9.7.5.3. Comparison and Selection Instructions: selp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp

// 9.7.5.4. Comparison and Selection Instructions: slct
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-slct


/*
 *  9.7.6. Half Precision Comparison Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions
 *
 */

// 9.7.6.1. Half Precision Comparison Instructions: set
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-set

// 9.7.6.2. Half Precision Comparison Instructions: setp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-setp


/*
 *  9.7.7. Logic and Shift Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions
 *
 */

// 9.7.7.1. Logic and Shift Instructions: and
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-and

// 9.7.7.2. Logic and Shift Instructions: or
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-or

// 9.7.7.3. Logic and Shift Instructions: xor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-xor

// 9.7.7.4. Logic and Shift Instructions: not
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-not

// 9.7.7.5. Logic and Shift Instructions: cnot
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-cnot

// 9.7.7.6. Logic and Shift Instructions: lop3
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3

// 9.7.7.7. Logic and Shift Instructions: shf
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shf

// 9.7.7.8. Logic and Shift Instructions: shl
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shl

// 9.7.7.9. Logic and Shift Instructions: shr
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shr


/*
 *  9.7.8. Data Movement and Conversion Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions
 *
 */

// 9.7.8.3. Data Movement and Conversion Instructions: mov
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov

// 9.7.8.4. Data Movement and Conversion Instructions: mov
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov-2

// 9.7.8.5. Data Movement and Conversion Instructions: shfl (deprecated)
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-deprecated

// 9.7.8.6. Data Movement and Conversion Instructions: shfl.sync
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync

// 9.7.8.7. Data Movement and Conversion Instructions: prmt
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt

// 9.7.8.8. Data Movement and Conversion Instructions: ld
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld

// 9.7.8.9. Data Movement and Conversion Instructions: ld.global.nc
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc

// 9.7.8.10. Data Movement and Conversion Instructions: ldu
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ldu

// 9.7.8.11. Data Movement and Conversion Instructions: st
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st

// 9.7.8.12. Data Movement and Conversion Instructions: st.async
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async

/*
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _Type>
_CCCL_DEVICE static inline void st_async(
  _Type* __addr,
  const _Type& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  static_assert(sizeof(_Type) == 4 || sizeof(_Type) == 8, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 4) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];    // 1. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "r"(__as_b32(__value)),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 8) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];    // 1. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "l"(__as_b64(__value)),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_st_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type (&value)[2],
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _Type>
_CCCL_DEVICE static inline void st_async(
  _Type* __addr,
  const _Type (&__value)[2],
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  static_assert(sizeof(_Type) == 4 || sizeof(_Type) == 8, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 4) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3]; // 2. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "r"(__as_b32(__value[0])),
          "r"(__as_b32(__value[1])),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (sizeof(_Type) == 8) {
      asm (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 [%0], {%1, %2}, [%3]; // 2. "
        :
        : "r"(__as_ptr_remote_dsmem(__addr)),
          "l"(__as_b64(__value[0])),
          "l"(__as_b64(__value[1])),
          "r"(__as_ptr_remote_dsmem(__remote_bar))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_st_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.  PTX ISA 81, SM_90
template <typename B32>
__device__ static inline void st_async(
  B32* addr,
  const B32 (&value)[4],
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void st_async(
  _B32* __addr,
  const _B32 (&__value)[4],
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];    // 3. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr)),
        "r"(__as_b32(__value[0])),
        "r"(__as_b32(__value[1])),
        "r"(__as_b32(__value[2])),
        "r"(__as_b32(__value[3])),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_st_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

// 9.7.8.13. Data Movement and Conversion Instructions: multimem.ld_reduce, multimem.st, multimem.red
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red

// 9.7.8.14. Data Movement and Conversion Instructions: prefetch, prefetchu
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu

// 9.7.8.15. Data Movement and Conversion Instructions: applypriority
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-applypriority

// 9.7.8.16. Data Movement and Conversion Instructions: discard
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard

// 9.7.8.17. Data Movement and Conversion Instructions: createpolicy
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy

// 9.7.8.18. Data Movement and Conversion Instructions: isspacep
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep

// 9.7.8.19. Data Movement and Conversion Instructions: cvta
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta

// 9.7.8.20. Data Movement and Conversion Instructions: cvt
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt

// 9.7.8.21. Data Movement and Conversion Instructions: cvt.pack
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt-pack

// 9.7.8.22. Data Movement and Conversion Instructions: mapa
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa

// 9.7.8.23. Data Movement and Conversion Instructions: getctarank
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-getctarank
/*
// getctarank{.space}.u32 dest, addr; // PTX ISA 78, SM_90
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline uint32_t getctarank(
  cuda::ptx::space_cluster_t,
  const void* addr);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_getctarank_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t getctarank(
  space_cluster_t,
  const void* __addr)
{
  // __space == space_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __dest;
    asm (
      "getctarank.shared::cluster.u32 %0, %1;"
      : "=r"(__dest)
      : "r"(__as_ptr_smem(__addr))
      :
    );
    return __dest;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_getctarank_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780


/*
 *  9.7.8.24. Data Movement and Conversion Instructions: Asynchronous copy
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy
 *
 */

// 9.7.8.24.3. Data Movement and Conversion Instructions: cp.async
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

// 9.7.8.24.4. Data Movement and Conversion Instructions: cp.async.commit_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group

// 9.7.8.24.5. Data Movement and Conversion Instructions: cp.async.wait_group / cp.async.wait_all
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all

// 9.7.8.24.6. Data Movement and Conversion Instructions: cp.async.bulk
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [smem_bar]; // 1a. unicast PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const _CUDA_VSTD::uint32_t& __size,
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3]; // 1a. unicast"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__as_ptr_gmem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk{.dst}{.src}.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask; // 1.  PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __srcMem,
  const _CUDA_VSTD::uint32_t& __size,
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4; // 1. "
      :
      : "r"(__as_ptr_dsmem(__dstMem)),
        "l"(__as_ptr_gmem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [rdsmem_bar]; // 2.  PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk(
  space_cluster_t,
  space_shared_t,
  void* __dstMem,
  const void* __srcMem,
  const _CUDA_VSTD::uint32_t& __size,
  _CUDA_VSTD::uint64_t* __rdsmem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3]; // 2. "
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.dst.src.bulk_group [dstMem], [srcMem], size; // 3.  PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  void* dstMem,
  const void* srcMem,
  const uint32_t& size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk(
  space_global_t,
  space_shared_t,
  void* __dstMem,
  const void* __srcMem,
  const _CUDA_VSTD::uint32_t& __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2; // 3. "
      :
      : "l"(__as_ptr_gmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

// 9.7.8.24.7. Data Movement and Conversion Instructions: cp.reduce.async.bulk
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk

// 9.7.8.24.8. Data Movement and Conversion Instructions: cp.async.bulk.prefetch
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch

// 9.7.8.24.9. Data Movement and Conversion Instructions: cp.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1a. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];// 1a."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2a. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2}], [%3], %4; // 2a."
      :
      : "r"(__as_ptr_dsmem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.1d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3a. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%0, {%1}], [%2]; // 3a."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1b. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];// 1b."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2b. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5; // 2b."
      :
      : "r"(__as_ptr_dsmem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3b. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3]; // 3b."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1c. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];// 1c."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2c. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6; // 2c."
      :
      : "r"(__as_ptr_dsmem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3c. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 3c."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1d. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];// 1d."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2d. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4, %5}], [%6], %7; // 2d."
      :
      : "r"(__as_ptr_dsmem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3d. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 3d."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1e. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];// 1e."
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2e. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar,
  const uint16_t& ctaMask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  _CUDA_VSTD::uint64_t* __smem_bar,
  const _CUDA_VSTD::uint16_t& __ctaMask)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8; // 2e."
      :
      : "r"(__as_ptr_dsmem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar)),
        "h"(__ctaMask)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3e. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename=void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 3e."
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

// 9.7.8.24.10. Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor
/*
// cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec || __op == op_and_op || __op == op_or_op || __op == op_xor_op, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_add) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_min) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_max) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_inc) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_dec) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_and_op) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_or_op) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__op == op_xor_op) {
      asm (
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

// 9.7.8.24.11. Data Movement and Conversion Instructions: cp.async.bulk.prefetch.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor

// 9.7.8.24.12. Data Movement and Conversion Instructions: cp.async.bulk.commit_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
/*
// cp.async.bulk.commit_group; // PTX ISA 80, SM_90
template <typename=void>
__device__ static inline void cp_async_bulk_commit_group();
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_commit_group_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void cp_async_bulk_commit_group()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "cp.async.bulk.commit_group;"
      :
      :
      :
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_commit_group_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

// 9.7.8.24.13. Data Movement and Conversion Instructions: cp.async.bulk.wait_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
/*
// cp.async.bulk.wait_group N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group(
  cuda::ptx::n32_t<N32> N);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_wait_group_is_not_supported_before_SM_90__();
template <int _N32>
_CCCL_DEVICE static inline void cp_async_bulk_wait_group(
  n32_t<_N32> __N)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "cp.async.bulk.wait_group %0;"
      :
      : "n"(__N)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_wait_group_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// cp.async.bulk.wait_group.read N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group_read(
  cuda::ptx::n32_t<N32> N);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_wait_group_read_is_not_supported_before_SM_90__();
template <int _N32>
_CCCL_DEVICE static inline void cp_async_bulk_wait_group_read(
  n32_t<_N32> __N)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(__N)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_cp_async_bulk_wait_group_read_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

// 9.7.8.25. Data Movement and Conversion Instructions: tensormap.replace
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace
/*
// tensormap.replace.tile.global_address.space.b1024.b64    [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <typename B64>
__device__ static inline void tensormap_replace_global_address(
  cuda::ptx::space_global_t,
  void* tm_addr,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_global_address_is_not_supported_before_SM_90a__();
template <typename _B64>
_CCCL_DEVICE static inline void tensormap_replace_global_address(
  space_global_t,
  void* __tm_addr,
  _B64 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.global_address.global.b1024.b64    [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "l"(__as_b64(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_global_address_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_address.space.b1024.b64    [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <typename B64>
__device__ static inline void tensormap_replace_global_address(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_global_address_is_not_supported_before_SM_90a__();
template <typename _B64>
_CCCL_DEVICE static inline void tensormap_replace_global_address(
  space_shared_t,
  void* __tm_addr,
  _B64 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.global_address.shared::cta.b1024.b64    [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "l"(__as_b64(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_global_address_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.rank.space.b1024.b32              [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <typename B32>
__device__ static inline void tensormap_replace_rank(
  cuda::ptx::space_global_t,
  void* tm_addr,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_rank_is_not_supported_before_SM_90a__();
template <typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_rank(
  space_global_t,
  void* __tm_addr,
  _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.rank.global.b1024.b32              [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_rank_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.rank.space.b1024.b32              [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <typename B32>
__device__ static inline void tensormap_replace_rank(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_rank_is_not_supported_before_SM_90a__();
template <typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_rank(
  space_shared_t,
  void* __tm_addr,
  _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.rank.shared::cta.b1024.b32              [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_rank_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.box_dim.space.b1024.b32           [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_box_dim(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_box_dim_is_not_supported_before_SM_90a__();
template <int _N32, typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_box_dim(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.box_dim.global.b1024.b32           [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_box_dim_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.box_dim.space.b1024.b32           [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_box_dim(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_box_dim_is_not_supported_before_SM_90a__();
template <int _N32, typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_box_dim(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.box_dim.shared::cta.b1024.b32           [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_box_dim_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_dim.space.b1024.b32        [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_global_dim(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_global_dim_is_not_supported_before_SM_90a__();
template <int _N32, typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_global_dim(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.global_dim.global.b1024.b32        [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_global_dim_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_dim.space.b1024.b32        [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_global_dim(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_global_dim_is_not_supported_before_SM_90a__();
template <int _N32, typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_global_dim(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32        [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_global_dim_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_stride.space.b1024.b64     [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B64>
__device__ static inline void tensormap_replace_global_stride(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_global_stride_is_not_supported_before_SM_90a__();
template <int _N32, typename _B64>
_CCCL_DEVICE static inline void tensormap_replace_global_stride(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B64 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.global_stride.global.b1024.b64     [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord),
        "l"(__as_b64(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_global_stride_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_stride.space.b1024.b64     [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B64>
__device__ static inline void tensormap_replace_global_stride(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_global_stride_is_not_supported_before_SM_90a__();
template <int _N32, typename _B64>
_CCCL_DEVICE static inline void tensormap_replace_global_stride(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B64 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.global_stride.shared::cta.b1024.b64     [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord),
        "l"(__as_b64(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_global_stride_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.element_stride.space.b1024.b32    [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_element_size(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_element_size_is_not_supported_before_SM_90a__();
template <int _N32, typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_element_size(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.element_stride.global.b1024.b32    [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_element_size_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.element_stride.space.b1024.b32    [tm_addr], ord, new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32, typename B32>
__device__ static inline void tensormap_replace_element_size(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_element_size_is_not_supported_before_SM_90a__();
template <int _N32, typename _B32>
_CCCL_DEVICE static inline void tensormap_replace_element_size(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __ord,
  _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.element_stride.shared::cta.b1024.b32    [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord),
        "r"(__as_b32(__new_val))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_element_size_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.elemtype.space.b1024.b32          [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_elemtype(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_elemtype_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_elemtype(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.elemtype.global.b1024.b32          [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_elemtype_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.elemtype.space.b1024.b32          [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_elemtype(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_elemtype_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_elemtype(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.elemtype.shared::cta.b1024.b32          [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_elemtype_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_interleave_layout(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_interleave_layout_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_interleave_layout(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.interleave_layout.global.b1024.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_interleave_layout_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_interleave_layout(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_interleave_layout_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_interleave_layout(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_interleave_layout_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.swizzle_mode.space.b1024.b32      [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_mode(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_swizzle_mode_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_swizzle_mode(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.swizzle_mode.global.b1024.b32      [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_swizzle_mode_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.swizzle_mode.space.b1024.b32      [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_mode(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_swizzle_mode_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_swizzle_mode(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32      [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_swizzle_mode_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.fill_mode.space.b1024.b32         [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_fill_mode(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_fill_mode_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_fill_mode(
  space_global_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.fill_mode.global.b1024.b32         [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_fill_mode_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.fill_mode.space.b1024.b32         [tm_addr], new_val; // PTX ISA 83, SM_90a
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_fill_mode(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_replace_fill_mode_is_not_supported_before_SM_90a__();
template <int _N32>
_CCCL_DEVICE static inline void tensormap_replace_fill_mode(
  space_shared_t,
  void* __tm_addr,
  n32_t<_N32> __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_HAS_FEATURE_SM_90a,(
    asm (
      "tensormap.replace.tile.fill_mode.shared::cta.b1024.b32         [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__new_val)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_replace_fill_mode_is_not_supported_before_SM_90a__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
 *  9.7.9. Texture Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions
 *
 */

// 9.7.9.3. Texture Instructions: tex
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tex

// 9.7.9.4. Texture Instructions: tld4
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tld4

// 9.7.9.5. Texture Instructions: txq
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-txq

// 9.7.9.6. Texture Instructions: istypep
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-istypep


/*
 *  9.7.10. Surface Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions
 *
 */

// 9.7.10.1. Surface Instructions: suld
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suld

// 9.7.10.2. Surface Instructions: sust
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// 9.7.10.3. Surface Instructions: sured
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sured

// 9.7.10.4. Surface Instructions: suq
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suq


/*
 *  9.7.11. Control Flow Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions
 *
 */

// 9.7.11.1. Control Flow Instructions: {}
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-curly-braces

// 9.7.11.2. Control Flow Instructions: @
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-at

// 9.7.11.3. Control Flow Instructions: bra
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra

// 9.7.11.4. Control Flow Instructions: brx.idx
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-brx-idx

// 9.7.11.5. Control Flow Instructions: call
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-call

// 9.7.11.6. Control Flow Instructions: ret
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-ret

// 9.7.11.7. Control Flow Instructions: exit
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit


/*
 *  9.7.12. Parallel Synchronization and Communication Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions
 *
 */

// 9.7.12.1. Parallel Synchronization and Communication Instructions: bar, barrier
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier

// 9.7.12.2. Parallel Synchronization and Communication Instructions: bar.warp.sync
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync

// 9.7.12.3. Parallel Synchronization and Communication Instructions: barrier.cluster
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster
/*
// barrier.cluster.arrive; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_arrive();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_arrive()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.arrive;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.wait; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_wait();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_wait()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.wait;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .release }
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_release_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(
  sem_release_t)
{
  // __sem == sem_release (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.arrive.release;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// Marked volatile
template <typename=void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_relaxed_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(
  sem_relaxed_t)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.arrive.relaxed;"
      :
      :
      :
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.wait.sem; // PTX ISA 80, SM_90
// .sem       = { .acquire }
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_wait(
  cuda::ptx::sem_acquire_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void barrier_cluster_wait(
  sem_acquire_t)
{
  // __sem == sem_acquire (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "barrier.cluster.wait.acquire;"
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

// 9.7.12.4. Parallel Synchronization and Communication Instructions: membar/fence
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
/*
// fence{.sem}.scope; // 1. PTX ISA 60, SM_70
// .sem       = { .sc, .acq_rel }
// .scope     = { .cta, .gpu, .sys }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 600
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_70__();
template <dot_sem _Sem, dot_scope _Scope>
_CCCL_DEVICE static inline void fence(
  sem_t<_Sem> __sem,
  scope_t<_Scope> __scope)
{
  static_assert(__sem == sem_sc || __sem == sem_acq_rel, "");
  static_assert(__scope == scope_cta || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc && __scope == scope_cta) {
      asm volatile (
        "fence.sc.cta; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc && __scope == scope_gpu) {
      asm volatile (
        "fence.sc.gpu; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc && __scope == scope_sys) {
      asm volatile (
        "fence.sc.sys; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel && __scope == scope_cta) {
      asm volatile (
        "fence.acq_rel.cta; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel && __scope == scope_gpu) {
      asm volatile (
        "fence.acq_rel.gpu; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel && __scope == scope_sys) {
      asm volatile (
        "fence.acq_rel.sys; // 1."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_is_not_supported_before_SM_70__();
  ));
}
#endif // __cccl_ptx_isa >= 600

/*
// fence{.sem}.scope; // 2. PTX ISA 78, SM_90
// .sem       = { .sc, .acq_rel }
// .scope     = { .cluster }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_90__();
template <dot_sem _Sem>
_CCCL_DEVICE static inline void fence(
  sem_t<_Sem> __sem,
  scope_cluster_t)
{
  static_assert(__sem == sem_sc || __sem == sem_acq_rel, "");
  // __scope == scope_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc) {
      asm volatile (
        "fence.sc.cluster; // 2."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel) {
      asm volatile (
        "fence.acq_rel.cluster; // 2."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 780
/*
// fence.mbarrier_init.sem.scope; // 3. PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
template <typename=void>
__device__ static inline void fence_mbarrier_init(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_mbarrier_init_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void fence_mbarrier_init(
  sem_release_t,
  scope_cluster_t)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "fence.mbarrier_init.release.cluster; // 3."
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_mbarrier_init_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// fence.proxy.alias; // 4. PTX ISA 75, SM_70
template <typename=void>
__device__ static inline void fence_proxy_alias();
*/
#if __cccl_ptx_isa >= 750
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_alias_is_not_supported_before_SM_70__();
template <typename=void>
_CCCL_DEVICE static inline void fence_proxy_alias()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,(
    asm volatile (
      "fence.proxy.alias; // 4."
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_alias_is_not_supported_before_SM_70__();
  ));
}
#endif // __cccl_ptx_isa >= 750
/*
// fence.proxy.async; // 5. PTX ISA 80, SM_90
template <typename=void>
__device__ static inline void fence_proxy_async();
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void fence_proxy_async()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "fence.proxy.async; // 5."
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
// .space     = { .global, .shared::cluster, .shared::cta }
template <cuda::ptx::dot_space Space>
__device__ static inline void fence_proxy_async(
  cuda::ptx::space_t<Space> space);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <dot_space _Space>
_CCCL_DEVICE static inline void fence_proxy_async(
  space_t<_Space> __space)
{
  static_assert(__space == space_global || __space == space_cluster || __space == space_shared, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__space == space_global) {
      asm volatile (
        "fence.proxy.async.global; // 6."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__space == space_cluster) {
      asm volatile (
        "fence.proxy.async.shared::cluster; // 6."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__space == space_shared) {
      asm volatile (
        "fence.proxy.async.shared::cta; // 6."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline void fence_proxy_tensormap_generic(
  sem_release_t,
  scope_t<_Scope> __scope)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.cta; // 7."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.cluster; // 7."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_gpu) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.gpu; // 7."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_sys) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.sys; // 7."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  const void* addr,
  cuda::ptx::n32_t<N32> size);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
template <int _N32, dot_scope _Scope>
_CCCL_DEVICE static inline void fence_proxy_tensormap_generic(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  const void* __addr,
  n32_t<_N32> __size)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.cta [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.cluster [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_gpu) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.gpu [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_sys) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.sys [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 830

// 9.7.12.5. Parallel Synchronization and Communication Instructions: atom
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom

// 9.7.12.6. Parallel Synchronization and Communication Instructions: red
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red

// 9.7.12.7. Parallel Synchronization and Communication Instructions: red.async
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .inc }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_inc_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_inc_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_inc (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .dec }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_dec_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_dec_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_dec (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_min_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_max_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::uint32_t* __dest,
  const _CUDA_VSTD::uint32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u32 (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_min_t,
  _CUDA_VSTD::int32_t* __dest,
  const _CUDA_VSTD::int32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_s32 (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_max_t,
  _CUDA_VSTD::int32_t* __dest,
  const _CUDA_VSTD::int32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_s32 (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::int32_t* __dest,
  const _CUDA_VSTD::int32_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_s32 (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .and }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_and_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void red_async(
  op_and_op_t,
  _B32* __dest,
  const _B32& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__as_b32(__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .or }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_or_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void red_async(
  op_or_op_t,
  _B32* __dest,
  const _B32& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__as_b32(__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .xor }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_xor_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_CCCL_DEVICE static inline void red_async(
  op_xor_op_t,
  _B32* __dest,
  const _B32& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(__as_b32(__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u64 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint64_t* dest,
  const uint64_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::uint64_t* __dest,
  const _CUDA_VSTD::uint64_t& __value,
  _CUDA_VSTD::uint64_t* __remote_bar)
{
  // __type == type_u64 (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "l"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int64_t* dest,
  const int64_t& value,
  int64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void red_async(
  op_add_t,
  _CUDA_VSTD::int64_t* __dest,
  const _CUDA_VSTD::int64_t& __value,
  _CUDA_VSTD::int64_t* __remote_bar)
{
  // __op == op_add (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [%0], %1, [%2]; // .u64 intentional"
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "l"(__value),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_red_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 810

// 9.7.12.8. Parallel Synchronization and Communication Instructions: vote (deprecated)
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-deprecated

// 9.7.12.9. Parallel Synchronization and Communication Instructions: vote.sync
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync

// 9.7.12.10. Parallel Synchronization and Communication Instructions: match.sync
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync

// 9.7.12.11. Parallel Synchronization and Communication Instructions: activemask
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask

// 9.7.12.12. Parallel Synchronization and Communication Instructions: redux.sync
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync

// 9.7.12.13. Parallel Synchronization and Communication Instructions: griddepcontrol
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol

// 9.7.12.14. Parallel Synchronization and Communication Instructions: elect.sync
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync

/*
 *  9.7.12.15. Parallel Synchronization and Communication Instructions: mbarrier
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier
 *
 *  Contained in: __cuda/ptx/parallel_synchronization_and_communication_instructions_mbarrier.h
 */

// 9.7.12.15.18. Parallel Synchronization and Communication Instructions: tensormap.cp_fenceproxy
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy
/*
// tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.sem.scope.sync.aligned  [dst], [src], size; // PTX ISA 83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void tensormap_cp_fenceproxy(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  void* dst,
  const void* src,
  cuda::ptx::n32_t<N32> size);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_cp_fenceproxy_is_not_supported_before_SM_90__();
template <int _N32, dot_scope _Scope>
_CCCL_DEVICE static inline void tensormap_cp_fenceproxy(
  sem_release_t,
  scope_t<_Scope> __scope,
  void* __dst,
  const void* __src,
  n32_t<_N32> __size)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cta.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cluster.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_gpu) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_sys) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.sys.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_cp_fenceproxy_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 830


/*
 *  9.7.13. Warp Level Matrix Multiply-Accumulate Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions
 *
 */

// 9.7.13.3.3. Warp-level Matrix Load Instruction: wmma.load
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-wmma-load

// 9.7.13.3.4. Warp-level Matrix Store Instruction: wmma.store
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-wmma-store

// 9.7.13.3.5. Warp-level Matrix Multiply-and-Accumulate Instruction: wmma.mma
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-and-accumulate-instruction-wmma-mma

// 9.7.13.4.14. Multiply-and-Accumulate Instruction: mma
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma

// 9.7.13.4.15. Warp-level matrix load instruction: ldmatrix
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix

// 9.7.13.4.16. Warp-level matrix store instruction: stmatrix
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix

// 9.7.13.4.17. Warp-level matrix transpose instruction: movmatrix
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-transpose-instruction-movmatrix

// 9.7.13.5.3. Multiply-and-Accumulate Instruction: mma.sp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma-sp


/*
 *  9.7.14. Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions
 *
 */

// 9.7.14.5.2. Asynchronous Multiply-and-Accumulate Instruction: wgmma.mma_async
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async

// 9.7.14.6.4. Asynchronous Multiply-and-Accumulate Instruction: wgmma.mma_async.sp
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async-sp

// 9.7.14.7.1. Asynchronous Multiply-and-Accumulate Instruction: wgmma.fence
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-fence

// 9.7.14.7.2. Asynchronous Multiply-and-Accumulate Instruction: wgmma.commit_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-commit-group

// 9.7.14.7.3. Asynchronous Multiply-and-Accumulate Instruction: wgmma.wait_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group


/*
 *  9.7.15. Stack Manipulation Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions
 *
 */

// 9.7.15.1. Stack Manipulation Instructions: stacksave
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stacksave

// 9.7.15.2. Stack Manipulation Instructions: stackrestore
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stackrestore

// 9.7.15.3. Stack Manipulation Instructions: alloca
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-alloca


/*
 *  9.7.16. Video Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#video-instructions
 *
 */

// 9.7.16.1.1. Scalar Video Instructions: vadd, vsub, vabsdiff, vmin, vmax
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax

// 9.7.16.1.2. Scalar Video Instructions: vshl, vshr
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vshl-vshr

// 9.7.16.1.3. Scalar Video Instructions: vmad
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vmad

// 9.7.16.1.4. Scalar Video Instructions: vset
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vset


/*
 *  9.7.16.2. SIMD Video Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions
 *
 */

// 9.7.16.2.1. SIMD Video Instructions: vadd2, vsub2, vavrg2, vabsdiff2, vmin2, vmax2
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd2-vsub2-vavrg2-vabsdiff2-vmin2-vmax2

// 9.7.16.2.2. SIMD Video Instructions: vset2
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset2

// 9.7.16.2.3. SIMD Video Instructions: vadd4, vsub4, vavrg4, vabsdiff4, vmin4, vmax4
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4

// 9.7.16.2.4. SIMD Video Instructions: vset4
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset4


/*
 *  9.7.17. Miscellaneous Instructions
 *  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions
 *
 */

// 9.7.17.1. Miscellaneous Instructions: brkpt
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt

// 9.7.17.2. Miscellaneous Instructions: nanosleep
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep

// 9.7.17.3. Miscellaneous Instructions: pmevent
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent

// 9.7.17.4. Miscellaneous Instructions: trap
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-trap

// 9.7.17.5. Miscellaneous Instructions: setmaxnreg
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg

// 10. Special Registers
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers
/*
// mov.u32 sreg_value, %%tid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_tid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_tid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%tid.x;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%tid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_tid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_tid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%tid.y;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%tid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_tid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_tid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%tid.z;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ntid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ntid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ntid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile (
    "mov.u32 %0, %%ntid.x;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ntid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ntid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ntid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile (
    "mov.u32 %0, %%ntid.y;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ntid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ntid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ntid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile (
    "mov.u32 %0, %%ntid.z;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%laneid; // PTX ISA 13
template <typename=void>
__device__ static inline uint32_t get_sreg_laneid();
*/
#if __cccl_ptx_isa >= 130
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_laneid()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%laneid;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 130

/*
// mov.u32 sreg_value, %%warpid; // PTX ISA 13
template <typename=void>
__device__ static inline uint32_t get_sreg_warpid();
*/
#if __cccl_ptx_isa >= 130
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_warpid()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile (
    "mov.u32 %0, %%warpid;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 130

/*
// mov.u32 sreg_value, %%nwarpid; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_nwarpid();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nwarpid_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nwarpid()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm volatile (
      "mov.u32 %0, %%nwarpid;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_nwarpid_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ctaid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ctaid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ctaid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%ctaid.x;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ctaid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ctaid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ctaid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%ctaid.y;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%ctaid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_ctaid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_ctaid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%ctaid.z;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%nctaid.x; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_nctaid_x();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nctaid_x()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%nctaid.x;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%nctaid.y; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_nctaid_y();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nctaid_y()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%nctaid.y;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%nctaid.z; // PTX ISA 20
template <typename=void>
__device__ static inline uint32_t get_sreg_nctaid_z();
*/
#if __cccl_ptx_isa >= 200
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nctaid_z()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%nctaid.z;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%smid; // PTX ISA 13
template <typename=void>
__device__ static inline uint32_t get_sreg_smid();
*/
#if __cccl_ptx_isa >= 130
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_smid()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm (
    "mov.u32 %0, %%smid;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 130

/*
// mov.u32 sreg_value, %%nsmid; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_nsmid();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nsmid_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nsmid()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm volatile (
      "mov.u32 %0, %%nsmid;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_nsmid_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u64 sreg_value, %%gridid; // PTX ISA 30
template <typename=void>
__device__ static inline uint64_t get_sreg_gridid();
*/
#if __cccl_ptx_isa >= 300
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_gridid()
{
  _CUDA_VSTD::uint64_t __sreg_value;
  asm (
    "mov.u64 %0, %%gridid;"
    : "=l"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 300

/*
// mov.pred sreg_value, %%is_explicit_cluster; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool get_sreg_is_explicit_cluster();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_is_explicit_cluster_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline bool get_sreg_is_explicit_cluster()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "{\n\t .reg .pred P_OUT; \n\t"
      "mov.pred P_OUT, %%is_explicit_cluster;\n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__sreg_value)
      :
      :
    );
    return static_cast<bool>(__sreg_value);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_is_explicit_cluster_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%clusterid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_clusterid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clusterid_x_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clusterid_x()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%clusterid.x;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_clusterid_x_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%clusterid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_clusterid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clusterid_y_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clusterid_y()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%clusterid.y;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_clusterid_y_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%clusterid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_clusterid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clusterid_z_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clusterid_z()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%clusterid.z;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_clusterid_z_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%nclusterid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_nclusterid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nclusterid_x_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nclusterid_x()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%nclusterid.x;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_nclusterid_x_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%nclusterid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_nclusterid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nclusterid_y_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nclusterid_y()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%nclusterid.y;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_nclusterid_y_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%nclusterid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_nclusterid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_nclusterid_z_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_nclusterid_z()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%nclusterid.z;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_nclusterid_z_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctaid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctaid_x_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctaid_x()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_ctaid.x;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_ctaid_x_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctaid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctaid_y_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctaid_y()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_ctaid.y;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_ctaid_y_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctaid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctaid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctaid_z_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctaid_z()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_ctaid.z;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_ctaid_z_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctaid.x; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_x();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctaid_x_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctaid_x()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_nctaid.x;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_nctaid_x_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctaid.y; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_y();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctaid_y_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctaid_y()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_nctaid.y;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_nctaid_y_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctaid.z; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctaid_z();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctaid_z_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctaid_z()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_nctaid.z;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_nctaid_z_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_ctarank; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_ctarank();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_ctarank_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_ctarank()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_ctarank;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_ctarank_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%cluster_nctarank; // PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_cluster_nctarank();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_cluster_nctarank_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_cluster_nctarank()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%cluster_nctarank;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_cluster_nctarank_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mov.u32 sreg_value, %%lanemask_eq; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_eq();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_eq_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_eq()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%lanemask_eq;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_lanemask_eq_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_le; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_le();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_le_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_le()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%lanemask_le;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_lanemask_le_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_lt; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_lt();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_lt_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_lt()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%lanemask_lt;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_lanemask_lt_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_ge; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_ge();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_ge_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_ge()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%lanemask_ge;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_lanemask_ge_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%lanemask_gt; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_lanemask_gt();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_lanemask_gt_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_lanemask_gt()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%lanemask_gt;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_lanemask_gt_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u32 sreg_value, %%clock; // PTX ISA 10
template <typename=void>
__device__ static inline uint32_t get_sreg_clock();
*/
#if __cccl_ptx_isa >= 100
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clock()
{
  _CUDA_VSTD::uint32_t __sreg_value;
  asm volatile (
    "mov.u32 %0, %%clock;"
    : "=r"(__sreg_value)
    :
    :
  );
  return __sreg_value;
}
#endif // __cccl_ptx_isa >= 100

/*
// mov.u32 sreg_value, %%clock_hi; // PTX ISA 50, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_clock_hi();
*/
#if __cccl_ptx_isa >= 500
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clock_hi_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_clock_hi()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm volatile (
      "mov.u32 %0, %%clock_hi;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_clock_hi_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 500

/*
// mov.u64 sreg_value, %%clock64; // PTX ISA 20, SM_35
template <typename=void>
__device__ static inline uint64_t get_sreg_clock64();
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_clock64_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_clock64()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint64_t __sreg_value;
    asm volatile (
      "mov.u64 %0, %%clock64;"
      : "=l"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_clock64_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 200

/*
// mov.u64 sreg_value, %%globaltimer; // PTX ISA 31, SM_35
template <typename=void>
__device__ static inline uint64_t get_sreg_globaltimer();
*/
#if __cccl_ptx_isa >= 310
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_globaltimer_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_globaltimer()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint64_t __sreg_value;
    asm volatile (
      "mov.u64 %0, %%globaltimer;"
      : "=l"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_globaltimer_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 310

/*
// mov.u32 sreg_value, %%globaltimer_lo; // PTX ISA 31, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_globaltimer_lo();
*/
#if __cccl_ptx_isa >= 310
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_globaltimer_lo_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_globaltimer_lo()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm volatile (
      "mov.u32 %0, %%globaltimer_lo;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_globaltimer_lo_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 310

/*
// mov.u32 sreg_value, %%globaltimer_hi; // PTX ISA 31, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_globaltimer_hi();
*/
#if __cccl_ptx_isa >= 310
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_globaltimer_hi_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_globaltimer_hi()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm volatile (
      "mov.u32 %0, %%globaltimer_hi;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_globaltimer_hi_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 310

/*
// mov.u32 sreg_value, %%total_smem_size; // PTX ISA 41, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_total_smem_size();
*/
#if __cccl_ptx_isa >= 410
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_total_smem_size_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_total_smem_size()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%total_smem_size;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_total_smem_size_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 410

/*
// mov.u32 sreg_value, %%aggr_smem_size; // PTX ISA 81, SM_90
template <typename=void>
__device__ static inline uint32_t get_sreg_aggr_smem_size();
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_aggr_smem_size_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_aggr_smem_size()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%aggr_smem_size;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_aggr_smem_size_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 810

/*
// mov.u32 sreg_value, %%dynamic_smem_size; // PTX ISA 41, SM_35
template <typename=void>
__device__ static inline uint32_t get_sreg_dynamic_smem_size();
*/
#if __cccl_ptx_isa >= 410
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_dynamic_smem_size_is_not_supported_before_SM_35__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t get_sreg_dynamic_smem_size()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_35,(
    _CUDA_VSTD::uint32_t __sreg_value;
    asm (
      "mov.u32 %0, %%dynamic_smem_size;"
      : "=r"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_dynamic_smem_size_is_not_supported_before_SM_35__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 410

/*
// mov.u64 sreg_value, %%current_graph_exec; // PTX ISA 80, SM_50
template <typename=void>
__device__ static inline uint64_t get_sreg_current_graph_exec();
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_get_sreg_current_graph_exec_is_not_supported_before_SM_50__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t get_sreg_current_graph_exec()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_50,(
    _CUDA_VSTD::uint64_t __sreg_value;
    asm (
      "mov.u64 %0, %%current_graph_exec;"
      : "=l"(__sreg_value)
      :
      :
    );
    return __sreg_value;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_get_sreg_current_graph_exec_is_not_supported_before_SM_50__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 800

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _LIBCUDACXX___CUDA_PTX_H
