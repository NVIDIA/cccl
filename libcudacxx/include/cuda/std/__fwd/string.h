//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_STRING_H
#define _CUDA_STD___FWD_STRING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/allocator.h>
#include <cuda/std/__fwd/char_traits.h>
#include <cuda/std/__fwd/memory_resource.h>

#include <cuda/std/__cccl/prologue.h>

// std:: forward declarations

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

// libstdc++ puts basic_string to inline cxx11 namespace
#  if _GLIBCXX_USE_CXX11_ABI
inline _GLIBCXX_BEGIN_NAMESPACE_CXX11
#  endif // _GLIBCXX_USE_CXX11_ABI

  template <class _CharT, class _Traits, class _Alloc>
  class basic_string;

#  if _GLIBCXX_USE_CXX11_ABI
_GLIBCXX_END_NAMESPACE_CXX11
#  endif // _GLIBCXX_USE_CXX11_ABI

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

// cuda::std:: forward declarations

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if 0 // we don't support these features
template <class _CharT, class _Traits = char_traits<_CharT>, class _Allocator = allocator<_CharT>>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_string;

using string  = basic_string<char>;
using wstring = basic_string<wchar_t>;
#  if _CCCL_HAS_CHAR8_T()
using u8string = basic_string<char8_t>;
#  endif // _CCCL_HAS_CHAR8_T()
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;

namespace pmr
{

template <class _CharT, class _Traits = char_traits<_CharT>>
using basic_string = ::cuda::std::basic_string<_CharT, _Traits, polymorphic_allocator<_CharT>>;

using string  = basic_string<char>;
using wstring = basic_string<wchar_t>;
#  if _CCCL_HAS_CHAR8_T()
using u8string = basic_string<char8_t>;
#  endif // _CCCL_HAS_CHAR8_T()
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;

} // namespace pmr

// clang-format off
template <class _CharT, class _Traits, class _Allocator>
class _CCCL_PREFERRED_NAME(string)
      _CCCL_PREFERRED_NAME(wstring)
#if _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(u8string)
#endif // _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(u16string)
      _CCCL_PREFERRED_NAME(u32string)
      _CCCL_PREFERRED_NAME(pmr::string)
      _CCCL_PREFERRED_NAME(pmr::wstring)
#  if _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(pmr::u8string)
#  endif // _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(pmr::u16string)
      _CCCL_PREFERRED_NAME(pmr::u32string)
      basic_string;
// clang-format on
#endif // 0

template <class _Tp>
inline constexpr bool __is_std_basic_string_v = false;
#if _CCCL_HAS_HOST_STD_LIB()
template <class _CharT, class _Traits, class _Alloc>
inline constexpr bool __is_std_basic_string_v<::std::basic_string<_CharT, _Traits, _Alloc>> = true;
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_STRING_H
