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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _Type>
_LIBCUDACXX_DEVICE static inline void st_async(
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
    return __void__cuda_ptx_st_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _Type>
_LIBCUDACXX_DEVICE static inline void st_async(
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
    return __void__cuda_ptx_st_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_st_async_is_not_supported_before_SM_90__();
template <typename _B32>
_LIBCUDACXX_DEVICE static inline void st_async(
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
    return __void__cuda_ptx_st_async_is_not_supported_before_SM_90__();
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

// 9.7.8.24.7. Data Movement and Conversion Instructions: cp.reduce.async.bulk
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk

// 9.7.8.24.8. Data Movement and Conversion Instructions: cp.async.bulk.prefetch
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch

// 9.7.8.24.9. Data Movement and Conversion Instructions: cp.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor

// 9.7.8.24.10. Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor

// 9.7.8.24.11. Data Movement and Conversion Instructions: cp.async.bulk.prefetch.tensor
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor

// 9.7.8.24.12. Data Movement and Conversion Instructions: cp.async.bulk.commit_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group

// 9.7.8.24.13. Data Movement and Conversion Instructions: cp.async.bulk.wait_group
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group

// 9.7.8.25. Data Movement and Conversion Instructions: tensormap.replace
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace


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

// 9.7.12.4. Parallel Synchronization and Communication Instructions: membar/fence
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence

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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void red_async(
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
    return __void__cuda_ptx_red_async_is_not_supported_before_SM_90__();
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

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _LIBCUDACXX___CUDA_PTX_H
