//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_SHUFFLE_CUH
#define _CUDA_EXPERIMENTAL___COOP_SHUFFLE_CUH

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

#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <bool _Dummy = false>
[[nodiscard]] _CCCL_DEVICE_API auto __shuffle_impl(...)
{
  static_assert(_Dummy, "cudax::coop::shuffle is not implemented for this group");
}

_CCCL_TEMPLATE(class _Group, class _Tp)
_CCCL_REQUIRES(is_group<_Group> _CCCL_AND ::cuda::std::is_same_v<typename _Group::unit_type, thread_level>
                 _CCCL_AND ::cuda::std::is_same_v<typename _Group::level_type, warp_level>)
[[nodiscard]] _CCCL_DEVICE_API _Tp __shuffle_impl(const _Group& __group, _Tp __value, unsigned __src_rank) noexcept
{
  using _MappingResult         = typename _Group::__mapping_result_type;
  const auto& __mapping_result = __group.__mapping_result();

  _CCCL_ASSERT(__src_rank < __mapping_result.count(),
               "invalid __src_rank - must be less than the number of units within the group");

  const auto __lane_mask   = __mapping_result.lane_mask();
  const auto __lane_offset = static_cast<int>(__src_rank) - static_cast<int>(__mapping_result.rank());

  unsigned __src_lane{};
  if constexpr (_MappingResult::is_always_contiguous())
  {
    const auto __lane = ::cuda::ptx::get_sreg_laneid();
    __src_lane        = static_cast<unsigned>(__lane + __lane_offset);
  }
  else
  {
    __src_lane = ::__fns(__lane_mask.value(), 0, static_cast<int>(__src_rank) + 1);
  }
  return ::cuda::device::warp_shuffle_idx(__value, static_cast<int>(__src_lane), __lane_mask.value());
}

//! @brief Shuffles values among units within a group.
//! @param[in] __group The group.
//! @param[in] __value This thread's value to be shuffled.
//! @param[in] __src_rank The rank of the unit whose value should be taken by this unit.
//! @return The value passed to the function by the equivalent thread from the source rank unit.
template <class _Group, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp shuffle(const _Group& __group, _Tp __value, unsigned __src_rank) noexcept
{
  return ::cuda::experimental::coop::__shuffle_impl(__group, __value, __src_rank);
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_SHUFFLE_CUH
