//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024-26 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BIT_CAST_H
#define _CUDA_STD___BIT_BIT_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

// MSVC supports __builtin_bit_cast from 19.25 on
#if _CCCL_CHECK_BUILTIN(builtin_bit_cast) || _CCCL_COMPILER(MSVC, >, 19, 25)
#  define _CCCL_BUILTIN_BIT_CAST(...) __builtin_bit_cast(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bit_cast)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_BIT_CAST)
#  define _CCCL_CONSTEXPR_BIT_CAST       constexpr
#  define _CCCL_HAS_CONSTEXPR_BIT_CAST() 1
#else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ / vvv !_CCCL_BUILTIN_BIT_CAST vvv
#  define _CCCL_CONSTEXPR_BIT_CAST
#  define _CCCL_HAS_CONSTEXPR_BIT_CAST() 0
#endif // !_CCCL_BUILTIN_BIT_CAST

#if _CCCL_COMPILER(GCC, >=, 8)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wclass-memaccess")
#endif // _CCCL_COMPILER(GCC, >=, 8)

template <class _To, class _From>
[[nodiscard]] _CCCL_API inline _To __bit_cast_memcpy(const _From& __from) noexcept
{
#if !_CCCL_COMPILER(GCC, <=, 7)
  static_assert(::cuda::std::default_initializable<_To>,
                "bit_cast memcpy fallback requires the destination type to be default initializable");
#endif // !_CCCL_COMPILER(GCC, <=, 7)
  _To __temp;
  ::cuda::std::memcpy(&__temp, &__from, sizeof(_To));
  return __temp;
}

#if _CCCL_COMPILER(GCC, >=, 8)
_CCCL_DIAG_POP
#endif // _CCCL_COMPILER(GCC, >=, 8)

_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES((sizeof(_To) == sizeof(_From)) _CCCL_AND(::cuda::is_trivially_copyable_v<_To>)
                 _CCCL_AND(::cuda::is_trivially_copyable_v<_From>))
[[nodiscard]] _CCCL_API inline _CCCL_CONSTEXPR_BIT_CAST _To bit_cast(const _From& __from) noexcept
{
#if defined(_CCCL_BUILTIN_BIT_CAST)
  if constexpr (::cuda::std::is_trivially_copyable_v<_To> && ::cuda::std::is_trivially_copyable_v<_From>)
  {
    return _CCCL_BUILTIN_BIT_CAST(_To, __from);
  }
  else
#endif // _CCCL_BUILTIN_BIT_CAST
  {
    return ::cuda::std::__bit_cast_memcpy<_To>(__from);
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_CAST_H
