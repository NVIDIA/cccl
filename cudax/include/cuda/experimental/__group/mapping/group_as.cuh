//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_AS_CUH
#define _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_AS_CUH

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
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/span>

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/queries.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental
{
template <::cuda::std::size_t... _Counts>
struct __group_as_static_tag;

template <::cuda::std::size_t... _Counts, bool _IsExhaustive>
class group_as<__group_as_static_tag<_Counts...>, _IsExhaustive>
{
  static_assert(((_Counts != 0) && ...), "all _Counts must not be zero");
  static_assert((::cuda::std::in_range<unsigned>(_Counts) && ...), "all _Counts must be within uint32_t range");

  static constexpr auto __counts_sum = (0 + ... + _Counts);

public:
  template <bool _ParentIsAlwaysExhaustive, bool _ParentIsAlwaysContiguous>
  struct __mapping_result
  {
    unsigned __group_rank_;
    unsigned __count_;
    unsigned __rank_;

    [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result invalid() noexcept
    {
      return {__invalid_count_or_rank, __invalid_count_or_rank, __invalid_count_or_rank};
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
    {
      return sizeof...(_Counts);
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
    {
      return static_cast<unsigned>(sizeof...(_Counts));
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
        _CCCL_ASSERT(is_valid(), "getting count of thread that is not part of the group is UB");
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

  _CCCL_HIDE_FROM_ABI explicit group_as() = default;

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES(_IsExhaustive2)
  _CCCL_DEVICE_API explicit constexpr group_as(
    const ::cuda::std::integer_sequence<::cuda::std::size_t, _Counts...>&) noexcept
  {}

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_as(const ::cuda::std::integer_sequence<::cuda::std::size_t, _Counts...>&,
                                               const non_exhaustive_t&) noexcept
  {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return sizeof...(_Counts);
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count(::cuda::std::size_t __i) noexcept
  {
    if (__i >= sizeof...(_Counts))
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    constexpr ::cuda::std::size_t __counts[]{_Counts...};
    return __counts[__i];
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr unsigned count(::cuda::std::size_t __i) const noexcept
  {
    return static_cast<unsigned>(static_count(__i));
  }

  template <class _Unit, class _ParentGroup>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit& __unit, const _ParentGroup& __parent) const noexcept
  {
    using _ParentMappingResult = typename _ParentGroup::__mapping_result_type;
    using _MappingResult =
      __mapping_result<_ParentMappingResult::is_always_exhaustive(), _ParentMappingResult::is_always_contiguous()>;

    constexpr auto __static_nunits = ::cuda::experimental::__static_count_query_group<_Unit, _ParentGroup>();
    const auto __nunits            = _Unit::template count_as<unsigned>(__parent);
    const auto __unit_rank         = _Unit::template rank_as<unsigned>(__parent);
    const auto __ngroups           = static_cast<unsigned>(sizeof...(_Counts));

    if constexpr (_IsExhaustive)
    {
      if constexpr (__static_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_nunits == __counts_sum, "group_as mapping _IsExhaustive precondition violation");
      }
      else
      {
        _CCCL_ASSERT(__nunits == static_cast<unsigned>(__counts_sum),
                     "group_as mapping _IsExhaustive precondition violation");
      }
    }
    else
    {
      if constexpr (__static_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_nunits >= __counts_sum, "group_as mapping requires more units than are available");
      }
      else
      {
        _CCCL_ASSERT(__nunits >= static_cast<unsigned>(__counts_sum),
                     "group_as mapping requires more units than are available");
      }

      if (__unit_rank >= static_cast<unsigned>(__counts_sum))
      {
        return _MappingResult::invalid();
      }
    }

    unsigned __sum = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned __i = 0; __i < __ngroups; ++__i)
    {
      const auto __i_count = count(__i);
      if (__unit_rank < __sum + __i_count)
      {
        return _MappingResult{__i, __i_count, __unit_rank - __sum};
      }
      __sum += __i_count;
    }
    _CCCL_UNREACHABLE();
  }
};

template <::cuda::std::size_t _GroupCount>
struct __group_as_dynamic_tag;

template <::cuda::std::size_t _GroupCount, bool _IsExhaustive>
class group_as<__group_as_dynamic_tag<_GroupCount>, _IsExhaustive>
{
  static_assert(_GroupCount != ::cuda::std::dynamic_extent, "group_as requires static number of groups");

  unsigned __counts_[_GroupCount];

public:
  template <bool _ParentIsAlwaysExhaustive, bool _ParentIsAlwaysContiguous>
  struct __mapping_result
  {
    unsigned __group_rank_;
    unsigned __count_;
    unsigned __rank_;

    [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result invalid() noexcept
    {
      return {__invalid_count_or_rank, __invalid_count_or_rank, __invalid_count_or_rank};
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
    {
      return _GroupCount;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
    {
      return static_cast<unsigned>(_GroupCount);
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

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES(_IsExhaustive2)
  _CCCL_DEVICE_API explicit constexpr group_as(::cuda::std::span<const unsigned, _GroupCount> __counts) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < _GroupCount; ++__i)
    {
      _CCCL_ASSERT(__counts[__i] > 0, "none of the __counts can be 0");
      __counts_[__i] = __counts[__i];
    }
  }

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_as(::cuda::std::span<const unsigned, _GroupCount> __counts,
                                               const non_exhaustive_t&) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < _GroupCount; ++__i)
    {
      _CCCL_ASSERT(__counts[__i] > 0, "none of the __counts can be 0");
      __counts_[__i] = __counts[__i];
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return _GroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count(::cuda::std::size_t __i) noexcept
  {
    if (__i >= _GroupCount)
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr unsigned count(::cuda::std::size_t __i) const noexcept
  {
    if (__i >= _GroupCount)
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    return __counts_[__i];
  }

  template <class _Unit, class _ParentGroup>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit& __unit, const _ParentGroup& __parent) const noexcept
  {
    using _ParentMappingResult = typename _ParentGroup::__mapping_result_type;
    using _MappingResult =
      __mapping_result<_ParentMappingResult::is_always_exhaustive(), _ParentMappingResult::is_always_contiguous()>;

    const auto __nunits    = __unit.template count_as<unsigned>(__parent);
    const auto __unit_rank = __unit.template rank_as<unsigned>(__parent);

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_IsExhaustive)
    {
      _CCCL_ASSERT(::cuda::std::accumulate(__counts_, __counts_ + _GroupCount, 0u) == __nunits,
                   "group_as mapping _IsExhaustive precondition violation");
    }
    else if (__unit_rank >= ::cuda::std::accumulate(__counts_, __counts_ + _GroupCount, 0u))
    {
      return _MappingResult::invalid();
    }

    unsigned __sum = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned __i = 0; __i < _GroupCount; ++__i)
    {
      const auto __i_count = count(__i);
      if (__unit_rank < __sum + __i_count)
      {
        return _MappingResult{__i, __i_count, __unit_rank - __sum};
      }
      __sum += __i_count;
    }
    _CCCL_UNREACHABLE();
  }
};

template <::cuda::std::size_t... _Counts>
_CCCL_DEVICE group_as(const ::cuda::std::integer_sequence<::cuda::std::size_t, _Counts...>&)
  -> group_as<__group_as_static_tag<_Counts...>, true>;

template <::cuda::std::size_t... _Counts>
_CCCL_DEVICE group_as(const ::cuda::std::integer_sequence<::cuda::std::size_t, _Counts...>&, const non_exhaustive_t&)
  -> group_as<__group_as_static_tag<_Counts...>, false>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_spannable<_Tp> _CCCL_AND ::cuda::std::
                 is_same_v<unsigned, _SpanValueType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>>)
_CCCL_DEVICE group_as(_Tp& __v) -> group_as<__group_as_dynamic_tag<decltype(::cuda::std::span(__v))::extent>, true>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_spannable<_Tp> _CCCL_AND ::cuda::std::
                 is_same_v<unsigned, _SpanValueType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>>)
_CCCL_DEVICE group_as(_Tp& __v, const non_exhaustive_t&)
  -> group_as<__group_as_dynamic_tag<decltype(::cuda::std::span(__v))::extent>, false>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_AS_CUH
