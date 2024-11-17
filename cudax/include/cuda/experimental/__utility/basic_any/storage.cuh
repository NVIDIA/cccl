//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_STORAGE_H
#define __CUDAX_DETAIL_BASIC_ANY_STORAGE_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>

namespace cuda::experimental
{
_CCCL_NODISCARD _CUDAX_API inline constexpr size_t __buffer_size(size_t __size)
{
  /// round up to the nearest multiple of `__word`, which is the size of a
  /// void*.
  return ((__size ? (_CUDA_VSTD::max)(__size, sizeof(void*)) : __default_buffer_size) + __word - 1) / __word * __word;
}

_CCCL_NODISCARD _CUDAX_API inline constexpr size_t __buffer_align(size_t __align)
{
  /// need to be able to store a void* in the buffer.
  return __align ? (_CUDA_VSTD::max)(__align, alignof(void*)) : __default_buffer_align;
}

template <class _Tp>
_CCCL_NODISCARD _CUDAX_API inline constexpr bool __is_small(size_t __size, size_t __align) noexcept
{
  return (sizeof(_Tp) <= __size) && (__align % alignof(_Tp) == 0) && _CUDA_VSTD::is_nothrow_move_constructible_v<_Tp>;
}

_CUDAX_API inline void __swap_ptr_ptr(void* __lhs, void* __rhs) noexcept
{
  _CUDA_VSTD::swap(*static_cast<void**>(__lhs), *static_cast<void**>(__rhs));
}

template <class _Tp, class _Up, class _Vp = decltype(true ? __identity_t<_Tp*>() : __identity_t<_Up*>())>
_CCCL_NODISCARD _CUDAX_TRIVIAL_API bool __ptr_eq(_Tp* __lhs, _Up* __rhs) noexcept
{
  return static_cast<_Vp>(__lhs) == static_cast<_Vp>(__rhs);
}

_CCCL_NODISCARD _CUDAX_TRIVIAL_API constexpr bool __ptr_eq(detail::__ignore, detail::__ignore) noexcept
{
  return false;
}

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_STORAGE_H
