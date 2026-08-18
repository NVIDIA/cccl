//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_RAW_STORAGE_H
#define _CUDA___UTILITY_RAW_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/unique_ptr.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Frees storage that came from `new _OrigTp[]` (probably either std::byte or char), after
// destroying the `__count_` elements built in it.
template <class _Tp, class _OrigTp>
struct __raw_storage_array_deleter
{
  ::cuda::std::size_t __count_{};

  _CCCL_HOST_API void operator()(_Tp* __ptr) const noexcept
  {
    ::cuda::std::__reverse_destroy(__ptr, __ptr + __count_);
    delete[] reinterpret_cast<_OrigTp*>(__ptr);
  }
};

template <class _Tp>
using __raw_storage_array = ::cuda::std::unique_ptr<_Tp[], __raw_storage_array_deleter<_Tp, ::cuda::std::byte>>;

// Returns storage for `__count` elements, reporting zero of them constructed.
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API __raw_storage_array<_Tp> __make_raw_storage_array(::cuda::std::size_t __count)
{
  auto __bytes = ::cuda::std::make_unique<::cuda::std::byte[]>(sizeof(_Tp) * __count);

  static_assert(!::cuda::std::constructible_from<_Tp>,
                "Do not use this helper if your type is already default constructible. Just use a regular "
                "unique_ptr<T> in that case.");
  static_assert(alignof(_Tp) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
  return __raw_storage_array<_Tp>{reinterpret_cast<_Tp*>(__bytes.release()), {}};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_RAW_STORAGE_H
