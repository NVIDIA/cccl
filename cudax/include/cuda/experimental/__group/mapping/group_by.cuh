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
  template <::cuda::std::size_t _NGroups, bool _ParentIsAlwaysExhaustive, bool _ParentIsAlwaysContiguous>
  struct __mapping_result
  {
    unsigned __group_count_;
    unsigned __group_rank_;
    unsigned __rank_;

    [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result
    invalid(unsigned __group_count = __invalid_count_or_rank) noexcept
    {
      return {__group_count, __invalid_count_or_rank, __invalid_count_or_rank};
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
    {
      return _NGroups;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
    {
      _CCCL_ASSERT(__group_count_ != __invalid_count_or_rank,
                   "querying group_count() by a thread that was not part of the parent group is not allowed");
      return __group_count_;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
    {
      if constexpr (!is_always_exhaustive())
      {
        _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
      }
      return __group_rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
    {
      return _Count;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned count() const noexcept
    {
      if constexpr (!is_always_exhaustive())
      {
        _CCCL_ASSERT(is_valid(), "getting count of thread that is not part of the group is UB");
      }
      return static_cast<unsigned>(_Count);
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned rank() const noexcept
    {
      if constexpr (!is_always_exhaustive())
      {
        _CCCL_ASSERT(is_valid(), "getting rank of thread that is not part of the group is UB");
      }
      return __rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const noexcept
    {
      if constexpr (is_always_exhaustive())
      {
        return true;
      }
      else
      {
        return __rank_ != __invalid_count_or_rank;
      }
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
    {
      return _ParentIsAlwaysExhaustive && _IsExhaustive;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous() noexcept
    {
      return _ParentIsAlwaysContiguous;
    }
  };

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

  template <class _Unit, class _ParentGroup>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit& __unit, const _ParentGroup& __parent) const noexcept
  {
    constexpr auto __static_nunits = ::cuda::experimental::__static_count_query_group<_Unit, _ParentGroup>();
    constexpr auto __static_ngroups =
      (__static_nunits != ::cuda::std::dynamic_extent) ? __static_nunits / _Count : ::cuda::std::dynamic_extent;

    using _ParentMappingResult = typename _ParentGroup::__mapping_result_type;
    using _MappingResult =
      __mapping_result<__static_ngroups,
                       _ParentMappingResult::is_always_exhaustive(),
                       _ParentMappingResult::is_always_contiguous()>;

    const auto __nunits    = _Unit::template count_as<unsigned>(__parent);
    const auto __unit_rank = _Unit::template rank_as<unsigned>(__parent);

    _MappingResult __ret{};
    __ret.__group_count_ = __nunits / count();
    __ret.__group_rank_  = __unit_rank / count();
    __ret.__rank_        = __unit_rank % count();

    // If the mapping is exhaustive, check the preconditions, otherwise return invalid mapping for the remainder.
    if constexpr (_IsExhaustive)
    {
      if constexpr (__static_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_nunits % _Count == 0, "group_by mapping _IsExhaustive precondition violation");
      }
      else
      {
        _CCCL_ASSERT(__nunits % count() == 0, "group_by mapping _IsExhaustive precondition violation");
      }
    }
    else if (__nunits % count() != 0)
    {
      if (__ret.__group_rank_ >= __ret.__group_count_)
      {
        return _MappingResult::invalid(__ret.__group_count_);
      }
    }
    return __ret;
  }
};

template <bool _IsExhaustive>
class group_by<::cuda::std::dynamic_extent, _IsExhaustive>
{
  unsigned __count_;

public:
  template <bool _ParentIsAlwaysExhaustive, bool _ParentIsAlwaysContiguous>
  struct __mapping_result
  {
    unsigned __group_count_;
    unsigned __group_rank_;
    unsigned __count_;
    unsigned __rank_;

    [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result
    invalid(unsigned __group_count = __invalid_count_or_rank) noexcept
    {
      return {__group_count, __invalid_count_or_rank, __invalid_count_or_rank, __invalid_count_or_rank};
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
    {
      return ::cuda::std::dynamic_extent;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
    {
      _CCCL_ASSERT(__group_count_ != __invalid_count_or_rank,
                   "querying group_count() by a thread that was not part of the parent group is not allowed");
      return __group_count_;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
    {
      if constexpr (!is_always_exhaustive())
      {
        _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
      }
      return __group_rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
    {
      return ::cuda::std::dynamic_extent;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned count() const noexcept
    {
      if constexpr (!is_always_exhaustive())
      {
        _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
      }
      return __count_;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned rank() const noexcept
    {
      if constexpr (!is_always_exhaustive())
      {
        _CCCL_ASSERT(is_valid(), "getting rank of thread that is not part of the group is UB");
      }
      return __rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const noexcept
    {
      if constexpr (is_always_exhaustive())
      {
        return true;
      }
      else
      {
        return __rank_ != __invalid_count_or_rank;
      }
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
    {
      return _ParentIsAlwaysExhaustive && _IsExhaustive;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous() noexcept
    {
      return _ParentIsAlwaysContiguous;
    }
  };

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

  template <class _Unit, class _ParentGroup>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit& __unit, const _ParentGroup& __parent) const noexcept
  {
    using _ParentMappingResult = typename _ParentGroup::__mapping_result_type;
    using _MappingResult =
      __mapping_result<_ParentMappingResult::is_always_exhaustive(), _ParentMappingResult::is_always_contiguous()>;

    const auto __nunits    = __unit.template count_as<unsigned>(__parent);
    const auto __unit_rank = __unit.template rank_as<unsigned>(__parent);

    _MappingResult __ret{};
    __ret.__group_count_ = __nunits / __count_;
    __ret.__group_rank_  = __unit_rank / __count_;
    __ret.__count_       = __count_;
    __ret.__rank_        = __unit_rank % __count_;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_IsExhaustive)
    {
      _CCCL_ASSERT(__nunits % __count_ == 0, "group_by mapping _IsExhaustive precondition violation");
    }
    else if (__nunits % __count_ != 0)
    {
      if (__ret.__group_rank_ >= __ret.__group_count_)
      {
        return _MappingResult::invalid(__ret.__group_count_);
      }
    }
    return __ret;
  }
};

_CCCL_DEVICE group_by(unsigned) -> group_by<::cuda::std::dynamic_extent>;

_CCCL_DEVICE group_by(unsigned, const non_exhaustive_t&) -> group_by<::cuda::std::dynamic_extent, false>;

template <class _MappingResult>
_CCCL_DEVICE_API void __check_mapping_result(const _MappingResult& __mapping_result) noexcept
{
  // Don't check the mapping result if it's not valid. We can skip this check if mapping result is always exhaustive.
  if constexpr (!_MappingResult::is_always_exhaustive())
  {
    if (!__mapping_result.is_valid())
    {
      return;
    }
  }
  _CCCL_ASSERT(__mapping_result.group_rank() < __mapping_result.group_count(), "invalid group rank");
  _CCCL_ASSERT(__mapping_result.rank() < __mapping_result.count(), "invalid rank");
}
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_BY_CUH
