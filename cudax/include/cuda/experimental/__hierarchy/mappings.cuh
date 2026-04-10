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
// Requirements on mappings:
// - must be copyable
// - must implement `map(_Unit, _Level, _Hierarchy)` method that returns an object that satisfies the
//   `__group_mapping_result` concept

// todo(dabayer): do we want to add stride parameter?
template <::cuda::std::size_t _Np = ::cuda::std::dynamic_extent>
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
      return __group_rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
    {
      return _Np;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned count() const noexcept
    {
      return static_cast<unsigned>(_Np);
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned rank() const noexcept
    {
      return __rank_;
    }

    // todo(dabayer): add method that determines whether the unit is part of the group or not.
  };

  _CCCL_HIDE_FROM_ABI explicit group_by() = default;

  [[nodiscard]] _CCCL_API constexpr unsigned n() const noexcept
  {
    return static_cast<unsigned>(_Np);
  }

  template <class _Unit, class _Level, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit& __unit, const _Level&, _Hierarchy __hier) noexcept
  {
    constexpr auto __nunits_in_level = _Unit::static_count(_Level{}, __hier);
    constexpr auto __static_ngroups =
      (__nunits_in_level != ::cuda::std::dynamic_extent) ? __nunits_in_level / _Np : ::cuda::std::dynamic_extent;

    __mapping_result<__static_ngroups> __ret{};
    __ret.__group_count_ = _Unit::template count_as<unsigned>(_Level{}, __hier) / n();
    __ret.__group_rank_  = _Unit::template rank_as<unsigned>(_Level{}, __hier) / n();
    __ret.__rank_        = _Unit::template rank_as<unsigned>(_Level{}, __hier) % n();
    return __ret;
  }
};

template <>
class group_by<::cuda::std::dynamic_extent>
{
  unsigned __n_;

public:
  template <::cuda::std::size_t _NGroups>
  struct __mapping_result
  {
    unsigned __group_count_;
    unsigned __group_rank_;
    unsigned __count_;
    unsigned __rank_;

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
      return __group_rank_;
    }

    [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
    {
      return ::cuda::std::dynamic_extent;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned count() const noexcept
    {
      return __count_;
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned rank() const noexcept
    {
      return __rank_;
    }
  };

  _CCCL_DEVICE_API explicit constexpr group_by(unsigned __n) noexcept
      : __n_{__n}
  {
    _CCCL_ASSERT(__n_ > 0, "__n cannot be 0");
  }

  [[nodiscard]] _CCCL_API constexpr unsigned n() const noexcept
  {
    return __n_;
  }

  template <class _Unit, class _Level, class _Hierarchy>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit& __unit, const _Level& __level, const _Hierarchy& __hier) noexcept
  {
    __mapping_result<::cuda::std::dynamic_extent> __ret{};
    __ret.__group_count_ = __unit.template count_as<unsigned>(__level, __hier) / n();
    __ret.__group_rank_  = __unit.template rank_as<unsigned>(__level, __hier) / n();
    __ret.__count_       = n();
    __ret.__rank_        = __unit.template rank_as<unsigned>(__level, __hier) % n();
    return __ret;
  }
};

_CCCL_HOST_DEVICE group_by(unsigned) -> group_by<::cuda::std::dynamic_extent>;

template <class _Mapping, class _Unit, class _Level, class _Hierarchy>
using __group_mapping_result_t =
  decltype(::cuda::std::declval<_Mapping>().map(_Unit{}, _Level{}, ::cuda::std::declval<_Hierarchy>()));
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_MAPPINGS_CUH
