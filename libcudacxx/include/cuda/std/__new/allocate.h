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

#ifndef _LIBCUDACXX___NEW_ALLOCATE_H
#define _LIBCUDACXX___NEW_ALLOCATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#if !defined(_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION) && !defined(_CCCL_COMPILER_NVRTC)
#  include <new> // for ::std::std::align_val_t
#endif // !_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION !_CCCL_COMPILER_NVRTC

#if !defined(__cpp_sized_deallocation) || __cpp_sized_deallocation < 201309L
#  define _LIBCUDACXX_HAS_NO_LANGUAGE_SIZED_DEALLOCATION
#endif

#if !defined(_LIBCUDACXX_BUILDING_LIBRARY) && _CCCL_STD_VER < 2014 \
  && defined(_LIBCUDACXX_HAS_NO_LANGUAGE_SIZED_DEALLOCATION)
#  define _LIBCUDACXX_HAS_NO_LIBRARY_SIZED_DEALLOCATION
#endif

#if defined(_LIBCUDACXX_HAS_NO_LIBRARY_SIZED_DEALLOCATION) || defined(_LIBCUDACXX_HAS_NO_LANGUAGE_SIZED_DEALLOCATION)
#  define _LIBCUDACXX_HAS_NO_SIZED_DEALLOCATION
#endif

#if !defined(_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION) && !defined(_CCCL_COMPILER_NVRTC)
#  include <new> // for ::std::align_val_t
#endif // !_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __is_overaligned_for_new(size_t __align) noexcept
{
#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
  return __align > __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else // ^^^ __STDCPP_DEFAULT_NEW_ALIGNMENT__ ^^^ / vvv !__STDCPP_DEFAULT_NEW_ALIGNMENT__ vvv
  return __align > _LIBCUDACXX_ALIGNOF(max_align_t);
#endif // !__STDCPP_DEFAULT_NEW_ALIGNMENT__
}

template <class... _Args>
_LIBCUDACXX_INLINE_VISIBILITY void* __libcpp_operator_new(_Args... __args)
{
  // Those builtins are not usable on device and the tests crash when using them
#if 0 && __has_builtin(__builtin_operator_new) && __has_builtin(__builtin_operator_delete)
  return __builtin_operator_new(__args...);
#else // ^^^ use builtin ^^^ / vvv no builtin
  return ::operator new(__args...);
#endif // !__builtin_operator_new || !__builtin_operator_delete
}

template <class... _Args>
_LIBCUDACXX_INLINE_VISIBILITY void __libcpp_operator_delete(_Args... __args)
{
  // Those builtins are not usable on device and the tests crash when using them
#if 0 && __has_builtin(__builtin_operator_new) && __has_builtin(__builtin_operator_delete)
  __builtin_operator_delete(__args...);
#else // ^^^ use builtin ^^^ / vvv no builtin
  ::operator delete(__args...);
#endif // !__builtin_operator_new || !__builtin_operator_delete
}

inline _LIBCUDACXX_INLINE_VISIBILITY void* __libcpp_allocate(size_t __size, size_t __align)
{
#ifndef _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align))
  {
    const ::std::align_val_t __align_val = static_cast<::std::align_val_t>(__align);
    return __libcpp_operator_new(__size, __align_val);
  }
#endif // !_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
  (void) __align;
  return __libcpp_operator_new(__size);
}

template <class... _Args>
_LIBCUDACXX_INLINE_VISIBILITY void __do_deallocate_handle_size(void* __ptr, size_t __size, _Args... __args)
{
#ifdef _LIBCUDACXX_HAS_NO_SIZED_DEALLOCATION
  (void) __size;
  return _CUDA_VSTD::__libcpp_operator_delete(__ptr, __args...);
#else // ^^^ _LIBCUDACXX_HAS_NO_SIZED_DEALLOCATION ^^^ / vvv !_LIBCUDACXX_HAS_NO_SIZED_DEALLOCATION vvv
  return _CUDA_VSTD::__libcpp_operator_delete(__ptr, __size, __args...);
#endif // !_LIBCUDACXX_HAS_NO_SIZED_DEALLOCATION
}

inline _LIBCUDACXX_INLINE_VISIBILITY void __libcpp_deallocate(void* __ptr, size_t __size, size_t __align)
{
#ifndef _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align))
  {
    const ::std::align_val_t __align_val = static_cast<::std::align_val_t>(__align);
    return __do_deallocate_handle_size(__ptr, __size, __align_val);
  }
#endif // !_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
  (void) __align;
  return __do_deallocate_handle_size(__ptr, __size);
}

inline _LIBCUDACXX_INLINE_VISIBILITY void __libcpp_deallocate_unsized(void* __ptr, size_t __align)
{
#ifndef _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
  if (__is_overaligned_for_new(__align))
  {
    const ::std::align_val_t __align_val = static_cast<::std::align_val_t>(__align);
    return __libcpp_operator_delete(__ptr, __align_val);
  }
#endif // !_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
  (void) __align;
  return __libcpp_operator_delete(__ptr);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_ALLOCATE_H
