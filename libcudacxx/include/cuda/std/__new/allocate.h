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

#ifndef _CUDA_STD___NEW_ALLOCATE_H
#define _CUDA_STD___NEW_ALLOCATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstdlib/aligned_alloc.h>
#include <cuda/std/__fwd/new.h>
#include <cuda/std/__new/bad_alloc.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/cstddef>

#if __cpp_sized_deallocation >= 201309L
#  define _CCCL_HAS_SIZED_DEALLOCATION() 1
#else
#  define _CCCL_HAS_SIZED_DEALLOCATION() 0
#endif

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Args>
[[nodiscard]] _CCCL_API void* __cccl_operator_new(_Args... __args)
{
  return ::operator new(__args...);
}

template <class... _Args>
[[nodiscard]] _CCCL_API void* __cccl_operator_new(size_t __size, align_val_t __align, [[maybe_unused]] _Args... __args)
{
#if _CCCL_CUDA_COMPILER(CLANG) && _CCCL_DEVICE_COMPILATION()
  void* __ret = ::cuda::std::aligned_alloc(__size, static_cast<size_t>(__align));
  if (__ret == nullptr)
  {
    ::cuda::std::__throw_bad_alloc(); // always terminates on device
  }
  return __ret;
#else // ^^^ clang-cuda in device mode ^^^ / vvv other vvv
  return ::operator new(__size, __align, __args...);
#endif // ^^^ other ^^^
}

template <class... _Args>
_CCCL_API void __cccl_operator_delete(_Args... __args)
{
  ::operator delete(__args...);
}

#if _CCCL_HAS_SIZED_DEALLOCATION()
template <class... _Args>
_CCCL_API void __cccl_operator_delete(void* __ptr, size_t __size, align_val_t __align, _Args... __args)
{
#  if _CCCL_CUDA_COMPILER(CLANG) && _CCCL_DEVICE_COMPILATION()
  ::cuda::std::free(__ptr);
#  else // ^^^ clang-cuda in device mode ^^^ / vvv other vvv
  return ::operator delete(__ptr, __size, __align, __args...);
#  endif // ^^^ other ^^^
}
#else // ^^^ _CCCL_HAS_SIZED_DEALLOCATION() ^^^ / vvv !_CCCL_HAS_SIZED_DEALLOCATION() vvv
template <class... _Args>
_CCCL_API void __cccl_operator_delete(void* __ptr, align_val_t __align, _Args... __args)
{
#  if _CCCL_CUDA_COMPILER(CLANG) && _CCCL_DEVICE_COMPILATION()
  ::cuda::std::free(__ptr);
#  else // ^^^ clang-cuda in device mode ^^^ / vvv other vvv
  return ::operator delete(__ptr, __align, __args...);
#  endif // ^^^ other ^^^
}
#endif // ^^^ !_CCCL_HAS_SIZED_DEALLOCATION() ^^^

[[nodiscard]] _CCCL_API inline void* __cccl_allocate(size_t __size, size_t __align)
{
  if (__align > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
  {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return ::cuda::std::__cccl_operator_new(__size, __align_val);
  }
  return ::cuda::std::__cccl_operator_new(__size);
}

template <class... _Args>
_CCCL_API inline void __do_deallocate_handle_size(void* __ptr, [[maybe_unused]] size_t __size, _Args... __args)
{
#if _CCCL_HAS_SIZED_DEALLOCATION()
  return ::cuda::std::__cccl_operator_delete(__ptr, __size, __args...);
#else // ^^^ _CCCL_HAS_SIZED_DEALLOCATION() ^^^ / vvv !_CCCL_HAS_SIZED_DEALLOCATION() vvv
  return ::cuda::std::__cccl_operator_delete(__ptr, __args...);
#endif // !_CCCL_HAS_SIZED_DEALLOCATION()
}

_CCCL_API inline void __cccl_deallocate(void* __ptr, size_t __size, size_t __align)
{
  if (__align > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
  {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return ::cuda::std::__do_deallocate_handle_size(__ptr, __size, __align_val);
  }
  return ::cuda::std::__do_deallocate_handle_size(__ptr, __size);
}

_CCCL_API inline void __cccl_deallocate_unsized(void* __ptr, size_t __align)
{
  if (__align > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
  {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return ::cuda::std::__cccl_operator_delete(__ptr, __align_val);
  }
  return ::cuda::std::__cccl_operator_delete(__ptr);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NEW_ALLOCATE_H
