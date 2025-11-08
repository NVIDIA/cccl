// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_ALLOCATION_GUARD_H
#define _CUDA_STD___MEMORY_ALLOCATION_GUARD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Helper class to allocate memory using an Allocator in an exception safe
// manner.
//
// The intended usage of this class is as follows:
//
// 0
// 1     __allocation_guard<SomeAllocator> guard(alloc, 10);
// 2     do_some_initialization_that_may_throw(guard.__get());
// 3     save_allocated_pointer_in_a_noexcept_operation(guard.__release_ptr());
// 4
//
// If line (2) throws an exception during initialization of the memory, the
// guard's destructor will be called, and the memory will be released using
// Allocator deallocation. Otherwise, we release the memory from the guard on
// line (3) in an operation that can't throw -- after that, the guard is not
// responsible for the memory anymore.
//
// This is similar to a unique_ptr, except it's easier to use with a
// custom allocator.

_CCCL_BEGIN_NV_DIAG_SUPPRESS(2659) // constexpr non-static member function will not be implicitly 'const' in C++14

template <class _Alloc>
struct __allocation_guard
{
  using _Pointer = typename allocator_traits<_Alloc>::pointer;
  using _Size    = typename allocator_traits<_Alloc>::size_type;

  template <class _AllocT> // we perform the allocator conversion inside the constructor
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 explicit __allocation_guard(_AllocT __alloc, _Size __n)
      : __alloc_(::cuda::std::move(__alloc))
      , __n_(__n)
      , __ptr_(allocator_traits<_Alloc>::allocate(__alloc_, __n_)) // initialization order is important
  {}

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__allocation_guard() noexcept
  {
    if (__ptr_ != nullptr)
    {
      allocator_traits<_Alloc>::deallocate(__alloc_, __ptr_, __n_);
    }
  }

  _CCCL_API constexpr _Pointer __release_ptr() noexcept
  { // not called __release() because it's a keyword in objective-c++
    _Pointer __tmp = __ptr_;
    __ptr_         = nullptr;
    return __tmp;
  }

  _CCCL_API constexpr _Pointer __get() const noexcept
  {
    return __ptr_;
  }

private:
  _Alloc __alloc_;
  _Size __n_;
  _Pointer __ptr_;
};

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_ALLOCATION_GUARD_H
