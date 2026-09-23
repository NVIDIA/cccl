//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_BY_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_BY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/queries.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental::coop
{
// todo(dabayer): do we want to add stride parameter?
template <::cuda::std::size_t _StaticUnitCount, bool _IsAlwaysExhaustive>
class group_by
{
  static_assert(_StaticUnitCount != 0, "_StaticUnitCount must not be zero");
  static_assert(::cuda::std::in_range<::cuda::std::uint32_t>(_StaticUnitCount),
                "_StaticUnitCount must be within uint32_t range");

public:
  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES(_IsAlwaysExhaustive)
  _CCCL_DEVICE_API constexpr group_by(unsigned __unit_count) noexcept
  {
    _CCCL_ASSERT(__unit_count == _StaticUnitCount, "__unit_count must match the static _StaticUnitCount");
  }

  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES((!_IsAlwaysExhaustive))
  _CCCL_DEVICE_API constexpr group_by(const non_exhaustive_t&, unsigned __unit_count) noexcept
  {
    _CCCL_ASSERT(__unit_count == _StaticUnitCount, "__unit_count must match the static _StaticUnitCount");
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr auto __unit_count = static_cast<::cuda::std::uint32_t>(_StaticUnitCount);

    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_prev_nunits  = _PrevMappingResult::static_unit_count();
    constexpr auto __static_curr_ngroups =
      (__static_prev_nunits != ::cuda::std::dynamic_extent)
        ? __static_prev_nunits / _StaticUnitCount
        : ::cuda::std::dynamic_extent;
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent && __static_curr_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * __static_curr_ngroups)
        : ::cuda::std::dynamic_extent;

    using _MappingResult =
      __mapping_result<__static_ngroups,
                       _StaticUnitCount,
                       _PrevMappingResult::is_always_exhaustive() && _IsAlwaysExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }

    const auto __prev_nunits     = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank  = __prev_mapping_result.unit_rank();
    const auto __curr_ngroups    = __prev_nunits / __unit_count;
    const auto __curr_group_rank = __prev_unit_rank / __unit_count;
    const auto __ngroups         = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise return invalid mapping for the remainder.
    if constexpr (_IsAlwaysExhaustive)
    {
      if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_prev_nunits % _StaticUnitCount == 0,
                      "group_by mapping _IsAlwaysExhaustive precondition violation");
      }
      else
      {
        _CCCL_ASSERT(__prev_nunits % __unit_count == 0, "group_by mapping _IsAlwaysExhaustive precondition violation");
      }
    }
    else if (__prev_nunits % __unit_count != 0)
    {
      if (__curr_group_rank >= __curr_ngroups)
      {
        return _MappingResult::__invalid();
      }
    }

    const auto __group_rank = __prev_mapping_result.group_rank() * __curr_ngroups + __curr_group_rank;
    const auto __n          = __unit_count;
    const auto __rank       = __prev_unit_rank % __n;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::coop::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __n, __rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__ngroups, __group_rank, __n, __rank, __lane_mask};
  }
};

template <bool _IsAlwaysExhaustive>
class group_by<::cuda::std::dynamic_extent, _IsAlwaysExhaustive>
{
  ::cuda::std::uint32_t __unit_count_;

public:
  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES(_IsAlwaysExhaustive2)
  _CCCL_DEVICE_API explicit constexpr group_by(::cuda::std::uint32_t __unit_count) noexcept
      : __unit_count_{__unit_count}
  {
    _CCCL_ASSERT(__unit_count > 0, "__unit_count must be greater than 0");
  }

  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES((!_IsAlwaysExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_by(const non_exhaustive_t&, ::cuda::std::uint32_t __unit_count) noexcept
      : __unit_count_{__unit_count}
  {
    _CCCL_ASSERT(__unit_count > 0, "__unit_count must be greater than 0");
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    using _MappingResult =
      __mapping_result<::cuda::std::dynamic_extent,
                       ::cuda::std::dynamic_extent,
                       _PrevMappingResult::is_always_exhaustive() && _IsAlwaysExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }

    const auto __prev_nunits     = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank  = __prev_mapping_result.unit_rank();
    const auto __curr_ngroups    = __prev_nunits / __unit_count_;
    const auto __curr_group_rank = __prev_unit_rank / __unit_count_;
    const auto __ngroups         = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(__prev_nunits % __unit_count_ == 0, "group_by mapping _IsAlwaysExhaustive precondition violation");
    }
    else if (__prev_nunits % __unit_count_ != 0)
    {
      if (__curr_group_rank >= __curr_ngroups)
      {
        return _MappingResult::__invalid();
      }
    }

    const auto __group_rank = __prev_mapping_result.group_rank() * __curr_ngroups + __curr_group_rank;
    const auto __n          = __unit_count_;
    const auto __rank       = __prev_unit_rank % __unit_count_;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::coop::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __n, __rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__ngroups, __group_rank, __n, __rank, __lane_mask};
  }
};

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_by(_Tp) -> group_by<::cuda::std::__maybe_static_ext<_Tp>, true>;

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_by(const non_exhaustive_t&, _Tp)
  -> group_by<::cuda::std::__maybe_static_ext<_Tp>, false>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_BY_CUH
