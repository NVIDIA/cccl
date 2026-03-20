//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_ALIAS_TO_WORDS_H
#define _CUDA___MEMORY_ALIAS_TO_WORDS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <size_t _TypeSize,
          size_t _FullWords    = _TypeSize / sizeof(uint32_t),
          size_t _PartialBytes = _TypeSize % sizeof(uint32_t)>
struct __alias_storage
{
  uint32_t __full_words_[_FullWords];
  unsigned char __partial_bytes_[_PartialBytes];

  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr uint32_t& operator[](const size_t __index) noexcept
  {
    if (__index < _FullWords)
    {
      return __full_words_[__index];
    }
    else
    {
      uint32_t __result = static_cast<uint32_t>(__partial_bytes_[0]);
      if constexpr (_PartialBytes >= 2)
      {
        __result |= static_cast<uint32_t>(__partial_bytes_[1]) << 8;
      }
      if constexpr (_PartialBytes >= 3)
      {
        __result |= static_cast<uint32_t>(__partial_bytes_[2]) << 16;
      }
      return __result;
    }
  }

  template <class _Tp>
  _CCCL_API _CCCL_FORCEINLINE constexpr void __assign_to(_Tp& __value) noexcept
  {
    if constexpr (!::cuda::std::is_array_v<_Tp>
                  && (::cuda::std::is_trivially_copyable_v<_Tp> || ::cuda::std::__is_extended_arithmetic_v<_Tp>) )
    {
      __value = ::cuda::std::bit_cast<_Tp>(*this);
    }
    else
    {
      ::cuda::std::memcpy(
        static_cast<void*>(::cuda::std::addressof(__value)), static_cast<const void*>(this), sizeof(_Tp));
    }
  }

  template <class _Tp>
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr _Tp __convert_back() noexcept
  {
    _Tp __output;
    if constexpr (!::cuda::std::is_array_v<_Tp>
                  && (::cuda::std::is_trivially_copyable_v<_Tp> || ::cuda::std::__is_extended_arithmetic_v<_Tp>) )
    {
      __output = ::cuda::std::bit_cast<_Tp>(*this);
    }
    else
    {
      ::cuda::std::memcpy(
        static_cast<void*>(::cuda::std::addressof(__output)), static_cast<const void*>(this), sizeof(_Tp));
    }
    return __output;
  }
};

template <size_t _TypeSize, size_t _FullWords>
struct __alias_storage<_TypeSize, _FullWords, 0>
{
  uint32_t __full_words_[_FullWords];

  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr uint32_t& operator[](const size_t __index) noexcept
  {
    return __full_words_[__index];
  }

  template <class _Tp>
  _CCCL_API _CCCL_FORCEINLINE constexpr void __assign_to(_Tp& __value) noexcept
  {
    if constexpr (!::cuda::std::is_array_v<_Tp>
                  && (::cuda::std::is_trivially_copyable_v<_Tp> || ::cuda::std::__is_extended_arithmetic_v<_Tp>) )
    {
      __value = ::cuda::std::bit_cast<_Tp>(*this);
    }
    else
    {
      ::cuda::std::memcpy(
        static_cast<void*>(::cuda::std::addressof(__value)), static_cast<const void*>(this), sizeof(_Tp));
    }
  }

  template <class _Tp>
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr _Tp __convert_back() noexcept
  {
    _Tp __output;
    if constexpr (!::cuda::std::is_array_v<_Tp>
                  && (::cuda::std::is_trivially_copyable_v<_Tp> || ::cuda::std::__is_extended_arithmetic_v<_Tp>) )
    {
      __output = ::cuda::std::bit_cast<_Tp>(*this);
    }
    else
    {
      ::cuda::std::memcpy(
        static_cast<void*>(::cuda::std::addressof(__output)), static_cast<const void*>(this), sizeof(_Tp));
    }
    return __output;
  }
};

template <class _Tp>
_CCCL_API _CCCL_FORCEINLINE constexpr __alias_storage<sizeof(_Tp)> __alias_to_words(_Tp& __data) noexcept
{
  __alias_storage<sizeof(_Tp)> __words;
  if constexpr (!::cuda::std::is_array_v<_Tp>
                && (::cuda::std::is_trivially_copyable_v<_Tp> || ::cuda::std::__is_extended_arithmetic_v<_Tp>) )
  {
    __words = ::cuda::std::bit_cast<__alias_storage<sizeof(_Tp)>>(__data);
  }
  else
  {
    ::cuda::std::memcpy(static_cast<void*>(::cuda::std::addressof(__words)),
                        static_cast<const void*>(::cuda::std::addressof(__data)),
                        sizeof(_Tp));
  }
  return __words;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_ALIGN_DOWN_H
