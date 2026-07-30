//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_WARP_MATCH_ANY_H
#define _CUDA___WARP_WARP_MATCH_ANY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__type_traits/is_bitwise_comparable.h>
#  include <cuda/__type_traits/is_trivially_copyable.h>
#  include <cuda/__warp/lane_mask.h>
#  include <cuda/std/__cstring/memcpy.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

extern "C" _CCCL_DEVICE void __cuda__match_any_sync_is_not_supported_before_SM_70__();

//! @brief Returns the mask of lanes with the same bitwise value as the calling lane.
//!
//! @param[in] __data The data to compare across lanes.
//! @param[in] __lane_mask The mask of participating lanes.
//!
//! @return A lane mask containing lanes in `__lane_mask` whose `__data` matches the calling lane's data.
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API lane_mask
warp_match_any(const _Tp& __data, const lane_mask __lane_mask = lane_mask::all()) noexcept
{
  static_assert(is_trivially_copyable_v<_Tp>, "data must be trivially copyable");
  _CCCL_ASSERT(__lane_mask != lane_mask::none(), "lane_mask must be non-zero");

  if constexpr (::cuda::std::is_same_v<_Tp, bool>)
  {
    auto __mask = ::__ballot_sync(__lane_mask.value(), __data);
    if (!__data)
    {
      __mask = (~__mask) & __lane_mask.value();
    }
    return lane_mask{__mask};
  }
  else
  {
    constexpr int __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(::cuda::std::uint32_t));
    ::cuda::std::uint32_t __array[__ratio]{};

#  if defined(_CCCL_BUILTIN_CLEAR_PADDING)
    auto __data_copy = __data;
    _CCCL_BUILTIN_CLEAR_PADDING(&__data_copy);
    const auto __data_ptr = ::cuda::std::addressof(__data_copy);
#  else // ^^^ _CCCL_BUILTIN_CLEAR_PADDING ^^^ / vvv !_CCCL_BUILTIN_CLEAR_PADDING vvv
    static_assert(is_bitwise_comparable_v<_Tp>, "data must be bitwise comparable");
    const auto __data_ptr = ::cuda::std::addressof(__data);
#  endif // _CCCL_BUILTIN_CLEAR_PADDING
    ::cuda::std::memcpy(__array, __data_ptr, sizeof(_Tp));

    lane_mask __ret = __lane_mask;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      ::cuda::std::uint32_t __match_any_result = 0;
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,
                        (__match_any_result = ::__match_any_sync(__lane_mask.value(), __array[i]);),
                        (::cuda::device::__cuda__match_any_sync_is_not_supported_before_SM_70__();));
      __ret &= lane_mask{__match_any_result};
    }
    return __ret;
  }
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()
#endif // _CUDA___WARP_WARP_MATCH_ANY_H
