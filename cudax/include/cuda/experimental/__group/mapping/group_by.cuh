//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_BY_CUH
#define _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_BY_CUH

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

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/__group/queries.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental
{
struct non_exhaustive_t
{
  _CCCL_HIDE_FROM_ABI explicit non_exhaustive_t() = default;
};

_CCCL_DEVICE constexpr non_exhaustive_t non_exhaustive;

// Requirements on mappings:
// - must be copyable
// - must implement `map(_Unit, _Level, _Hierarchy)` method that returns an object that satisfies the
//   `__group_mapping_result` concept

// todo(dabayer): do we want to add stride parameter?
template <::cuda::std::size_t _Count, bool _IsExhaustive>
class group_by
{
  static_assert(_Count != 0, "_Count must not be zero");
  static_assert(::cuda::std::in_range<unsigned>(_Count), "_Count must be within uint32_t range");

public:
  _CCCL_HIDE_FROM_ABI explicit group_by() = default;

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive))
  _CCCL_DEVICE_API constexpr group_by(const non_exhaustive_t&) noexcept {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
  {
    return _Count;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_API constexpr unsigned count() const noexcept
  {
    return static_cast<unsigned>(_Count);
  }

  template <class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_prev_nunits  = _PrevMappingResult::static_count();
    constexpr auto __static_curr_ngroups =
      (__static_prev_nunits != ::cuda::std::dynamic_extent)
        ? __static_prev_nunits / _Count
        : ::cuda::std::dynamic_extent;
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent && __static_curr_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * __static_curr_ngroups)
        : ::cuda::std::dynamic_extent;

    using _MappingResult =
      __mapping_result<__static_ngroups,
                       _Count,
                       _PrevMappingResult::is_always_exhaustive() && _IsExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::invalid();
    }

    const auto __prev_nunits     = __prev_mapping_result.count();
    const auto __prev_unit_rank  = __prev_mapping_result.rank();
    const auto __curr_ngroups    = __prev_nunits / count();
    const auto __curr_group_rank = __prev_unit_rank / count();
    const auto __ngroups         = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise return invalid mapping for the remainder.
    if constexpr (_IsExhaustive)
    {
      if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_prev_nunits % _Count == 0, "group_by mapping _IsExhaustive precondition violation");
      }
      else
      {
        _CCCL_ASSERT(__prev_nunits % count() == 0, "group_by mapping _IsExhaustive precondition violation");
      }
    }
    else if (__prev_nunits % count() != 0)
    {
      if (__curr_group_rank >= __curr_ngroups)
      {
        return _MappingResult::invalid_with_group_count(__ngroups);
      }
    }
    return _MappingResult{
      __ngroups,
      __prev_mapping_result.group_rank() * __curr_ngroups + __curr_group_rank,
      count(),
      __prev_unit_rank % count()};
  }
};

template <bool _IsExhaustive>
class group_by<::cuda::std::dynamic_extent, _IsExhaustive>
{
  unsigned __count_;

public:
  _CCCL_DEVICE_API explicit constexpr group_by(unsigned __count) noexcept
      : __count_{__count}
  {
    _CCCL_ASSERT(__count > 0, "__count cannot be 0");
  }

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_by(unsigned __count, const non_exhaustive_t&) noexcept
      : __count_{__count}
  {
    _CCCL_ASSERT(__count > 0, "__count cannot be 0");
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
  {
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr unsigned count() const noexcept
  {
    return __count_;
  }

  template <class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    using _MappingResult =
      __mapping_result<::cuda::std::dynamic_extent,
                       ::cuda::std::dynamic_extent,
                       _PrevMappingResult::is_always_exhaustive() && _IsExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::invalid();
    }

    const auto __prev_nunits     = __prev_mapping_result.count();
    const auto __prev_unit_rank  = __prev_mapping_result.rank();
    const auto __curr_ngroups    = __prev_nunits / __count_;
    const auto __curr_group_rank = __prev_unit_rank / __count_;
    const auto __ngroups         = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_IsExhaustive)
    {
      _CCCL_ASSERT(__prev_nunits % __count_ == 0, "group_by mapping _IsExhaustive precondition violation");
    }
    else if (__prev_nunits % __count_ != 0)
    {
      if (__curr_group_rank >= __curr_ngroups)
      {
        return _MappingResult::invalid_with_group_count(__ngroups);
      }
    }
    return _MappingResult{
      __ngroups,
      __prev_mapping_result.group_rank() * __curr_ngroups + __curr_group_rank,
      __count_,
      __prev_unit_rank % __count_};
  }
};

_CCCL_DEVICE group_by(unsigned) -> group_by<::cuda::std::dynamic_extent>;

_CCCL_DEVICE group_by(unsigned, const non_exhaustive_t&) -> group_by<::cuda::std::dynamic_extent, false>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_BY_CUH
