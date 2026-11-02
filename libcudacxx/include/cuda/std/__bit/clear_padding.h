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
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_CLEAR_PADDING)

template <class _Tp, size_t _Size>
_CCCL_HOST_DEVICE_API inline void __clear_padding_array(_Tp (&__value)[_Size]) noexcept
{
  // Clang 23 miss padding in an array type named by a template parameter
  _CCCL_PRAGMA_UNROLL_FULL()
  for (auto& __element : __value)
  {
    if constexpr (::cuda::std::is_array_v<_Tp>)
    {
      ::cuda::std::__clear_padding_array(__element);
    }
    else
    {
      _CCCL_BUILTIN_CLEAR_PADDING(::cuda::std::addressof(__element));
    }
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline array<unsigned char, sizeof(_Tp)> __clear_padding(const _Tp& __value) noexcept
{
  // Return an array of chars so copying the result cannot reintroduce object padding
  array<unsigned char, sizeof(_Tp)> __bytes;
  if constexpr (::cuda::std::is_array_v<_Tp>)
  {
    // Wrap arrays because bit_cast cannot return a raw array.
    struct __data_wrapper
    {
      _Tp __value;
    };
    auto __copy = ::cuda::std::bit_cast<__data_wrapper>(__value);
    ::cuda::std::__clear_padding_array(__copy.__value);
    ::cuda::std::memcpy(__bytes.data(), ::cuda::std::addressof(__copy.__value), sizeof(_Tp));
  }
  else
  {
    auto __copy = ::cuda::std::bit_cast<_Tp>(__value);
    _CCCL_BUILTIN_CLEAR_PADDING(::cuda::std::addressof(__copy));
    ::cuda::std::memcpy(__bytes.data(), ::cuda::std::addressof(__copy), sizeof(_Tp));
  }
  return __bytes;
}

#endif // _CCCL_BUILTIN_CLEAR_PADDING

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_CLEAR_PADDING_H
