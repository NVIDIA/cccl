//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_SHUFFLE_DOWN_CUH
#define _CUDA_EXPERIMENTAL___COOP_SHUFFLE_DOWN_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/optional>

#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <bool _Dummy = false>
[[nodiscard]] _CCCL_DEVICE_API auto __shuffle_down_impl(...)
{
  static_assert(_Dummy, "cudax::coop::shuffle_down is not implemented for this group");
}

_CCCL_TEMPLATE(class _Group, class _Tp)
_CCCL_REQUIRES(is_group<_Group> _CCCL_AND ::cuda::std::is_same_v<typename _Group::unit_type, thread_level>
                 _CCCL_AND ::cuda::std::is_same_v<typename _Group::level_type, warp_level>)
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__shuffle_down_impl(const _Group& __group, const _Tp& __value, unsigned __offset) noexcept
{
  using _MappingResult         = typename _Group::__mapping_result_type;
  const auto& __mapping_result = __group.__mapping_result();

  const auto __lane_mask       = __mapping_result.lane_mask();
  const auto __offset_is_valid = (__offset < __mapping_result.count() - __mapping_result.rank());

  if constexpr (_MappingResult::is_always_contiguous())
  {
    const auto __real_offset = (__offset_is_valid) ? __offset : 0u;
    const auto __result =
      ::cuda::device::warp_shuffle_down(__value, static_cast<int>(__real_offset), __lane_mask.value());
    return (__offset_is_valid) ? ::cuda::std::optional{__result.data} : ::cuda::std::nullopt;
  }
  else
  {
    const auto __lane = ::cuda::ptx::get_sreg_laneid();
    const auto __src_lane =
      (__offset_is_valid) ? ::__fns(__lane_mask.value(), __lane, static_cast<int>(__offset + 1)) : __lane;
    const auto __result = ::cuda::device::warp_shuffle_idx(__value, static_cast<int>(__src_lane), __lane_mask.value());
    return (__offset_is_valid) ? ::cuda::std::optional{__result.data} : ::cuda::std::nullopt;
  }
}

//! @brief Gets the values from a unit with a greater rank by the specified offset.
//! @param[in] __group The group.
//! @param[in] __value This thread's value.
//! @param[in] __offset The offset of the source rank from this unit's rank.
//! @return The source's value or empty optional if no such rank exists.
template <class _Group, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
shuffle_down(const _Group& __group, const _Tp& __value, unsigned __offset) noexcept
{
  return ::cuda::experimental::coop::__shuffle_down_impl(__group, __value, __offset);
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_SHUFFLE_DOWN_CUH
