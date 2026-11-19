//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/is_trivially_constructible.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/remove_cvref.h>

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
[[nodiscard]] _CCCL_API _To __bit_cast_memcpy(const _From& __from) noexcept
{
  static_assert(::cuda::std::is_default_constructible_v<_To>,
                "bit_cast memcpy fallback requires the destination type to be default constructible");
  _To __temp;
  ::cuda::std::memcpy(::cuda::std::addressof(__temp), ::cuda::std::addressof(__from), sizeof(_To));
  return __temp;
}

#if _CCCL_COMPILER(GCC, >=, 8)
_CCCL_DIAG_POP
#endif // _CCCL_COMPILER(GCC, >=, 8)

#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_COMPILER(MSVC) && _CCCL_STD_VER == 2017

// NVCC + MSVC C++17 fails for guaranteed copy elision
template <class _To, class _From>
inline constexpr bool __can_be_directly_returned =
  ::cuda::std::is_scalar_v<_To>
  && ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_To>, ::cuda::std::remove_cvref_t<_From>>;

#else // !(_CCCL_CUDA_COMPILER(NVCC) && _CCCL_COMPILER(MSVC) && _CCCL_STD_VER == 2017)

// conjunction_v is used for the lazy evaluation of is_trivially_constructible
template <class _To, class _From>
inline constexpr bool __can_be_directly_returned =
  ::cuda::std::conjunction_v<::cuda::std::is_same<::cuda::std::remove_cvref_t<_To>, ::cuda::std::remove_cvref_t<_From>>,
                             ::cuda::std::is_trivially_constructible<_To, const _From&>>;

#endif // !(_CCCL_CUDA_COMPILER(NVCC) && _CCCL_COMPILER(MSVC) && _CCCL_STD_VER == 2017)

_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES((sizeof(_To) == sizeof(_From)) _CCCL_AND(::cuda::is_trivially_copyable_v<_To>)
                 _CCCL_AND(::cuda::is_trivially_copyable_v<_From>))
[[nodiscard]]
_CCCL_API _CCCL_CONSTEXPR_BIT_CAST _To bit_cast(const _From& __from) noexcept
{
  if constexpr (__can_be_directly_returned<_To, _From>)
  {
    return _To(__from); // this also allows explicit _To(const _To&) = default
  }
#if defined(_CCCL_BUILTIN_BIT_CAST)
  // Compilers unfortunately require the types to be trivially copyable per the letter of the
  // C++ law, and will emit errors if the builtin is called on types that aren't
  // NOLINTNEXTLINE(cccl-prefer-cuda-traits)
  else if constexpr (::cuda::std::is_trivially_copyable_v<_To> && ::cuda::std::is_trivially_copyable_v<_From>)
  {
    return _CCCL_BUILTIN_BIT_CAST(_To, __from);
  }
#endif // _CCCL_BUILTIN_BIT_CAST
  else
  {
    return ::cuda::std::__bit_cast_memcpy<_To>(__from);
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_CAST_H
