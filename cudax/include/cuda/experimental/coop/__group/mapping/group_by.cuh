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

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/hierarchy.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/integral_constant_like.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
template <::cuda::std::size_t _StaticUnitCount, class _PrevMappingResult>
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::size_t __group_by_static_group_count()
{
  using ::cuda::std::dynamic_extent;
  if constexpr (_StaticUnitCount == dynamic_extent || _PrevMappingResult::static_group_count() == dynamic_extent)
  {
    return dynamic_extent;
  }
  else if constexpr (!_PrevMappingResult::has_uniform_static_unit_count())
  {
    ::cuda::std::size_t __count = 0;
    for (::cuda::std::size_t __i = 0; __i < _PrevMappingResult::static_group_count(); ++__i)
    {
      __count += _PrevMappingResult::static_unit_count(__i) / _StaticUnitCount;
    }
    return __count;
  }
  else if constexpr (_PrevMappingResult::static_unit_count() != dynamic_extent)
  {
    return _PrevMappingResult::static_group_count() * (_PrevMappingResult::static_unit_count() / _StaticUnitCount);
  }
  else
  {
    return dynamic_extent;
  }
}

template <::cuda::std::size_t _StaticUnitCount, class _PrevMappingResult>
[[nodiscard]] _CCCL_DEVICE_API constexpr bool __group_by_is_always_divisible()
{
  using ::cuda::std::dynamic_extent;
  if constexpr (_StaticUnitCount == dynamic_extent)
  {
    return false;
  }
  else if constexpr (_PrevMappingResult::has_uniform_static_unit_count())
  {
    constexpr auto __count = _PrevMappingResult::static_unit_count();
    return __count != dynamic_extent && __count % _StaticUnitCount == 0;
  }
  else
  {
    for (::cuda::std::size_t __i = 0; __i < _PrevMappingResult::static_group_count(); ++__i)
    {
      if (_PrevMappingResult::static_unit_count(__i) % _StaticUnitCount != 0)
      {
        return false;
      }
    }
    return true;
  }
}

template <::cuda::std::size_t _StaticUnitCount, bool _Exhaustive, class _Unit, class _Level, class _PrevMappingResult>
[[nodiscard]] _CCCL_DEVICE_API auto
__group_by_partition(::cuda::std::uint32_t __unit_count, const _PrevMappingResult& __prev)
{
  using ::cuda::std::dynamic_extent;
  using ::cuda::std::uint32_t;

  constexpr bool __divisible = __group_by_is_always_divisible<_StaticUnitCount, _PrevMappingResult>();
  using _MappingResult       = __mapping_result<
    __uniform_static_unit_count<__group_by_static_group_count<_StaticUnitCount, _PrevMappingResult>(), _StaticUnitCount>,
    _Exhaustive || (_PrevMappingResult::is_always_exhaustive() && __divisible),
    _PrevMappingResult::is_always_contiguous()>;

  if constexpr (!_PrevMappingResult::is_always_exhaustive())
  {
    if (!__prev.is_valid())
    {
      return _MappingResult::__invalid();
    }
  }

  const auto __prev_unit_count   = __prev.unit_count();
  const auto __prev_unit_rank    = __prev.unit_rank();
  const auto __groups_per_parent = __prev_unit_count / __unit_count;
  const auto __local_group_rank  = __prev_unit_rank / __unit_count;

  const auto __prev_group_rank = __prev.group_rank();
  uint32_t __group_count       = 0;
  uint32_t __group_offset      = 0;
  if constexpr (_PrevMappingResult::has_uniform_static_unit_count())
  {
    __group_count  = __prev.group_count() * __groups_per_parent;
    __group_offset = __prev_group_rank * __groups_per_parent;
  }
  else if constexpr (_StaticUnitCount != dynamic_extent)
  {
    // NVCC can fold each prefix into an immediate. Indexing a local offset table would require a stack frame.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < _PrevMappingResult::static_group_count(); ++__i)
    {
      if (__i == __prev_group_rank)
      {
        __group_offset = __group_count;
      }
      __group_count += static_cast<uint32_t>(_PrevMappingResult::static_unit_count(__i) / _StaticUnitCount);
    }
  }
  else
  {
    for (::cuda::std::size_t __i = 0; __i < _PrevMappingResult::static_group_count(); ++__i)
    {
      const auto __count = static_cast<uint32_t>(_PrevMappingResult::static_unit_count(__i)) / __unit_count;
      __group_count += __count;
      if (__i < __prev_group_rank)
      {
        __group_offset += __count;
      }
    }
  }

  if constexpr (_Exhaustive)
  {
    if constexpr (_StaticUnitCount != dynamic_extent
                  && (!_PrevMappingResult::has_uniform_static_unit_count()
                      || _PrevMappingResult::static_unit_count() != dynamic_extent))
    {
      static_assert(__divisible, "group_by mapping would produce units without a group during exhaustive mapping");
    }
    else
    {
      _CCCL_ASSERT(__prev_unit_count % __unit_count == 0,
                   "group_by mapping would produce units without a group during exhaustive mapping");
    }
  }
  else if constexpr (!__divisible)
  {
    // Divisible parents have no tail. Checking divisibility first keeps register usage down for runtime widths.
    if (__prev_unit_count % __unit_count != 0 && __local_group_rank >= __groups_per_parent)
    {
      return _MappingResult::__invalid();
    }
  }

  const auto __unit_rank = __prev_unit_rank % __unit_count;
  const auto __lane_mask =
    ::cuda::experimental::coop::__make_lane_mask_for_n<_Unit, _Level, _PrevMappingResult::is_always_contiguous()>(
      __prev.lane_mask(), __unit_count, __unit_rank);
  return _MappingResult{__group_count, __group_offset + __local_group_rank, __unit_count, __unit_rank, __lane_mask};
}

template <::cuda::std::size_t _StaticUnitCount, bool _Exhaustive, class _Unit, class _ParentGroup, class _PrevMappingResult>
[[nodiscard]] _CCCL_DEVICE_API auto __group_by_map(
  ::cuda::std::uint32_t __unit_count, const _Unit&, const _ParentGroup& __parent, const _PrevMappingResult& __prev)
{
  using _Level = typename _ParentGroup::level_type;
  using ::cuda::std::uint32_t;

  auto __result = ::cuda::experimental::coop::__group_by_partition<_StaticUnitCount, _Exhaustive, _Unit, _Level>(
    __unit_count, __prev);

  if constexpr (_PrevMappingResult::has_uniform_static_unit_count()
                && _PrevMappingResult::static_unit_count() == ::cuda::std::dynamic_extent
                && _PrevMappingResult::static_group_count() != 1)
  {
    static_assert(::cuda::std::is_same_v<_Unit, thread_level> && ::cuda::std::is_same_v<_Level, warp_level>,
                  "group_by with multiple runtime-sized groups requires threads within a warp");

    // Excluded threads must also participate in the parent-wide ballot.
    const auto __leaders = ::__ballot_sync(
      __parent.__mapping_result().lane_mask().value(), __result.is_valid() && __result.unit_rank() == 0);
    if (__result.is_valid())
    {
      // The trailing zeros in the group's mask are the lanes below its first member.
      const auto __mask               = __result.lane_mask().value();
      const auto __lanes_before_group = ~__mask & (__mask - 1u);
      __result.__group_count_         = static_cast<uint32_t>(::cuda::std::popcount(__leaders));
      __result.__group_rank_          = static_cast<uint32_t>(::cuda::std::popcount(__leaders & __lanes_before_group));
    }
  }
  return __result;
}

template <::cuda::std::size_t _StaticUnitCount>
class group_by
{
  static_assert(_StaticUnitCount != 0, "_StaticUnitCount must not be zero");
  static_assert(::cuda::std::in_range<::cuda::std::uint32_t>(_StaticUnitCount),
                "_StaticUnitCount must be within uint32_t range");

public:
  _CCCL_DEVICE_API explicit constexpr group_by(unsigned __unit_count)
  {
    _CCCL_ASSERT(__unit_count == _StaticUnitCount, "__unit_count must match the static _StaticUnitCount");
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev) const
  {
    return ::cuda::experimental::coop::__group_by_map<_StaticUnitCount, true>(
      _StaticUnitCount, __unit, __parent, __prev);
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto map(
    const non_exhaustive_t&, const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev) const
  {
    return ::cuda::experimental::coop::__group_by_map<_StaticUnitCount, false>(
      _StaticUnitCount, __unit, __parent, __prev);
  }
};

template <>
class group_by<::cuda::std::dynamic_extent>
{
  ::cuda::std::uint32_t __unit_count_;

public:
  _CCCL_DEVICE_API explicit constexpr group_by(::cuda::std::uint32_t __unit_count)
      : __unit_count_{__unit_count}
  {
    _CCCL_ASSERT(__unit_count > 0, "__unit_count must be greater than 0");
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev) const
  {
    return ::cuda::experimental::coop::__group_by_map<::cuda::std::dynamic_extent, true>(
      __unit_count_, __unit, __parent, __prev);
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto map(
    const non_exhaustive_t&, const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev) const
  {
    return ::cuda::experimental::coop::__group_by_map<::cuda::std::dynamic_extent, false>(
      __unit_count_, __unit, __parent, __prev);
  }
};

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_by(_Tp) -> group_by<::cuda::std::__maybe_static_ext<_Tp>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_BY_CUH
