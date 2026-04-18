//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_MAPPINGS_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_MAPPINGS_CUH

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

#include <cuda/experimental/__hierarchy/fwd.cuh>

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
template <::cuda::std::size_t _Np, bool _IsExhaustive>
class group_by
{
  static_assert(_Np != 0, "_Np must not be zero");
  static_assert(::cuda::std::in_range<unsigned>(_Np), "_Np must be within uint32_t range");

public:
  template <::cuda::std::size_t _NGroups>
  struct __mapping_result
  {
    unsigned __group_count_;
    unsigned __group_rank_;
    unsigned __rank_;

    [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result __invalid(unsigned __group_count) noexcept
    {
      return {__group_count, 0xffff'ffffu, 0xffff'ffffu};
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
    {
      return _NGroups;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
    {
      return __group_count_;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
    {
      if constexpr (!_IsExhaustive)
      {
        _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
      }
      return __group_rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
    {
      return _Np;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned count() const noexcept
    {
      if constexpr (!_IsExhaustive)
      {
        _CCCL_ASSERT(is_valid(), "getting count of thread that is not part of the group is UB");
      }
      return static_cast<unsigned>(_Np);
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned rank() const noexcept
    {
      if constexpr (!_IsExhaustive)
      {
        _CCCL_ASSERT(is_valid(), "getting rank of thread that is not part of the group is UB");
      }
      return __rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const noexcept
    {
      if constexpr (_IsExhaustive)
      {
        return true;
      }
      else
      {
        return __rank_ != 0xffff'ffffu;
      }
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
    {
      return _IsExhaustive;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous() noexcept
    {
      return true;
    }
  };

  _CCCL_HIDE_FROM_ABI explicit group_by() = default;

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive))
  _CCCL_DEVICE_API constexpr group_by(non_exhaustive_t) noexcept {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
  {
    return _Np;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_API constexpr unsigned count() const noexcept
  {
    return static_cast<unsigned>(_Np);
  }

  template <class _Unit, class _Level, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit& __unit, const _Level&, _Hierarchy __hier) const noexcept
  {
    constexpr auto __static_nunits = _Unit::static_count(_Level{}, __hier);
    constexpr auto __static_ngroups =
      (__static_nunits != ::cuda::std::dynamic_extent) ? __static_nunits / _Np : ::cuda::std::dynamic_extent;

    const auto __nunits    = _Unit::template count_as<unsigned>(_Level{}, __hier);
    const auto __unit_rank = _Unit::template rank_as<unsigned>(_Level{}, __hier);

    __mapping_result<__static_ngroups> __ret{};
    __ret.__group_count_ = __nunits / count();
    __ret.__group_rank_  = __unit_rank / count();
    __ret.__rank_        = __unit_rank % count();

    // If the mapping is exhaustive, check the preconditions, otherwise return invalid mapping for the remainder.
    if constexpr (_IsExhaustive)
    {
      if constexpr (__static_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_nunits % _Np == 0, "group_by mapping _IsExhaustive precondition violation");
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
        return __mapping_result<__static_ngroups>::__invalid(__ret.__group_count_);
      }
    }
    return __ret;
  }
};

template <bool _IsExhaustive>
class group_by<::cuda::std::dynamic_extent, _IsExhaustive>
{
  unsigned __n_;

public:
  struct __mapping_result
  {
    unsigned __group_count_;
    unsigned __group_rank_;
    unsigned __count_;
    unsigned __rank_;

    [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result __invalid(unsigned __group_count) noexcept
    {
      return {__group_count, 0xffff'ffffu, 0xffff'ffffu, 0xffff'ffffu};
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
    {
      return ::cuda::std::dynamic_extent;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
    {
      return __group_count_;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
    {
      if constexpr (!_IsExhaustive)
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
      if constexpr (!_IsExhaustive)
      {
        _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
      }
      return __count_;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned rank() const noexcept
    {
      if constexpr (!_IsExhaustive)
      {
        _CCCL_ASSERT(is_valid(), "getting rank of thread that is not part of the group is UB");
      }
      return __rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const noexcept
    {
      if constexpr (_IsExhaustive)
      {
        return true;
      }
      else
      {
        return __rank_ != 0xffff'ffffu;
      }
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
    {
      return _IsExhaustive;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous() noexcept
    {
      return true;
    }
  };

  _CCCL_DEVICE_API explicit constexpr group_by(unsigned __n) noexcept
      : __n_{__n}
  {
    _CCCL_ASSERT(__n_ > 0, "__n cannot be 0");
  }

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_by(unsigned __n, non_exhaustive_t) noexcept
      : __n_{__n}
  {
    _CCCL_ASSERT(__n_ > 0, "__n cannot be 0");
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
    return __n_;
  }

  template <class _Unit, class _Level, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _Level& __level, const _Hierarchy& __hier) const noexcept
  {
    const auto __nunits    = __unit.template count_as<unsigned>(__level, __hier);
    const auto __unit_rank = __unit.template rank_as<unsigned>(__level, __hier);

    __mapping_result __ret{};
    __ret.__group_count_ = __nunits / __n_;
    __ret.__group_rank_  = __unit_rank / __n_;
    __ret.__count_       = __n_;
    __ret.__rank_        = __unit_rank % __n_;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_IsExhaustive)
    {
      _CCCL_ASSERT(__nunits % __n_ == 0, "group_by mapping _IsExhaustive precondition violation");
    }
    else if (__nunits % __n_ != 0)
    {
      if (__ret.__group_rank_ >= __ret.__group_count_)
      {
        return __mapping_result::__invalid(__ret.__group_count_);
      }
    }
    return __ret;
  }
};

_CCCL_DEVICE group_by(unsigned) -> group_by<::cuda::std::dynamic_extent>;

_CCCL_DEVICE group_by(unsigned, non_exhaustive_t) -> group_by<::cuda::std::dynamic_extent, false>;

template <class _Mapping, class _Unit, class _Level, class _Hierarchy>
using __group_mapping_result_t =
  decltype(::cuda::std::declval<_Mapping>().map(_Unit{}, _Level{}, ::cuda::std::declval<_Hierarchy>()));

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

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_MAPPINGS_CUH
