// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_TENSORMAP_REPLACE_H_
#define _CUDA_PTX_TENSORMAP_REPLACE_H_

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include "../ptx_dot_variants.h"
#include "../ptx_helper_functions.h"
#include "../../../cstdint"

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

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

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_TENSORMAP_REPLACE_H_
