//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___OPEN_ADDRESSING_FUNCTORS_CUH
#define _CUDAX___CUCO___OPEN_ADDRESSING_FUNCTORS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/tuple>

#include <cuda/experimental/__cuco/__detail/bitwise_compare.cuh>
#include <cuda/experimental/__cuco/traits.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Functor returning slot content for the given index.

template <bool _HasPayload, class _StorageRef>
struct __get_slot
{
  _StorageRef __storage;

  _CCCL_API explicit constexpr __get_slot(_StorageRef __storage_ref) noexcept
      : __storage{__storage_ref}
  {}

  _CCCL_DEVICE constexpr auto operator()(typename _StorageRef::__size_type __idx) const noexcept
  {
    if constexpr (_HasPayload)
    {
      const auto [__first, __second] = *(__storage.data() + __idx);
      return ::cuda::std::tuple{__first, __second};
    }
    else
    {
      return *(__storage.data() + __idx);
    }
  }
};

//! @brief Functor returning whether a slot is filled.

template <bool _HasPayload, class _Key>
struct __slot_is_filled
{
  _Key __empty_sentinel;
  _Key __erased_sentinel;

  _CCCL_API explicit constexpr __slot_is_filled(_Key __empty, _Key __erased) noexcept
      : __empty_sentinel{__empty}
      , __erased_sentinel{__erased}
  {}

  template <class _Slot>
  _CCCL_DEVICE constexpr bool operator()(_Slot __slot) const noexcept
  {
    const auto __key = [&]() {
      if constexpr (_HasPayload)
      {
        if constexpr (::cuda::experimental::cuco::__detail::__is_cuda_std_pair_like<_Slot>::value)
        {
          return ::cuda::std::get<0>(__slot);
        }
        else
        {
          return __slot.first;
        }
      }
      else
      {
        return __slot;
      }
    }();

    return !(__detail::__bitwise_compare(__key, __empty_sentinel)
             || __detail::__bitwise_compare(__key, __erased_sentinel));
  }
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_FUNCTORS_CUH
