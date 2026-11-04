//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_CLEAR_PADDING_H
#define _CUDA_STD___BIT_CLEAR_PADDING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_CLEAR_PADDING)

template <class _Tp>
_CCCL_HOST_DEVICE_API void __clear_padding(_Tp& __value) noexcept;

template <class _Tp, size_t _Size>
_CCCL_HOST_DEVICE_API void __clear_padding_array(_Tp (&__value)[_Size]) noexcept
{
  // Clang 23 misses padding in an array type named by a template parameter, so we need to clear each element
  _CCCL_PRAGMA_UNROLL_FULL()
  for (auto& __element : __value)
  {
    ::cuda::std::__clear_padding(__element);
  }
}

template <class _Tp>
_CCCL_HOST_DEVICE_API void __clear_padding(_Tp& __value) noexcept
{
  if constexpr (::cuda::std::is_array_v<_Tp>)
  {
    ::cuda::std::__clear_padding_array(__value);
  }
  else if constexpr (::cuda::std::is_trivially_copyable_v<_Tp>) // NOLINT(cccl-prefer-cuda-traits)
  {
    _CCCL_BUILTIN_CLEAR_PADDING(::cuda::std::addressof(__value));
  }
  else // cuda::is_trivially_copyable_v<_Tp>
  {
    // Clang 23 and GCC 14 require the argument is address of a variable or parameter
    // We also need to prevent operator& overloading
    _Tp __copy[1]{__value};
    _CCCL_BUILTIN_CLEAR_PADDING(&__copy);
    ::cuda::std::memcpy(static_cast<void*>(::cuda::std::addressof(__value)), __copy, sizeof(_Tp));
  }
}

template <class _Tp>
struct __array_copy
{
  _Tp __value;
};

template <class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::conditional_t<::cuda::std::is_array_v<_Tp>, __array_copy<_Tp>, _Tp>
__create_copy(const _Tp& __value) noexcept
{
  if constexpr (::cuda::std::is_array_v<_Tp>)
  {
    return ::cuda::std::bit_cast<__array_copy<_Tp>>(__value);
  }
  else
  {
    return __value;
  }
}

template <class _Tp>
_CCCL_HOST_DEVICE_API void __clear_padding(__array_copy<_Tp>& __value) noexcept
{
  ::cuda::std::__clear_padding_array(__value.__value);
}

#endif // _CCCL_BUILTIN_CLEAR_PADDING

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_CLEAR_PADDING_H
