//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_META_LEVEL_DIMENSIONS_H
#define _CUDA___HIERARCHY_META_LEVEL_DIMENSIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/hierarchy_levels.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/is_integer.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Wrapper requesting that a meta dimension be rounded up when needed.
template <class _Cnt>
struct at_least
{
  static_assert(::cuda::std::__cccl_is_integer_v<_Cnt>);

  _Cnt __value_;

  _CCCL_API constexpr explicit at_least(_Cnt __value_) noexcept
      : __value_(__value_)
  {}
};

template <class _Unit>
struct __target_count
{
  static_assert(__is_hierarchy_level_v<_Unit>);

  ::cuda::std::size_t __count_{};
  bool __ceil_div_{};

  _CCCL_API constexpr __target_count() noexcept {} // NOLINT(modernize-use-equals-default)

  template <class _Cnt>
  _CCCL_API constexpr explicit __target_count(_Cnt __count_) noexcept
      : __count_(static_cast<::cuda::std::size_t>(__count_))
  {
    static_assert(::cuda::std::__cccl_is_integer_v<_Cnt>);
  }

  template <class _Cnt>
  _CCCL_API constexpr explicit __target_count(at_least<_Cnt> __count_) noexcept
      : __count_(static_cast<::cuda::std::size_t>(__count_.__value_))
      , __ceil_div_(true)
  {}

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const __target_count& __lhs, const __target_count& __rhs) noexcept
  {
    return __lhs.__count_ == __rhs.__count_ && __lhs.__ceil_div_ == __rhs.__ceil_div_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const __target_count& __lhs, const __target_count& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
};

struct __max_occupancy
{};

[[nodiscard]] _CCCL_API constexpr bool operator==(__max_occupancy, __max_occupancy) noexcept
{
  return true;
}

[[nodiscard]] _CCCL_API constexpr bool operator!=(__max_occupancy, __max_occupancy) noexcept
{
  return false;
}

struct __device_fill
{
  float __fill_coeff_{};

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const __device_fill& __lhs, const __device_fill& __rhs) noexcept
  {
    return __lhs.__fill_coeff_ == __rhs.__fill_coeff_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const __device_fill& __lhs, const __device_fill& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
};

/**
 * @brief Type representing dimensions that must be finalized before launch.
 *
 * Meta dimensions communicate intent, such as "enough grid blocks for this
 * many threads", rather than a concrete extent. Finalization maps them to
 * ordinary `hierarchy_level_desc` objects.
 */
template <class _Level, class _Meta>
class _CCCL_DECLSPEC_EMPTY_BASES hierarchy_level_desc_meta : __hierarchy_level_desc_base
{
  static_assert(__is_hierarchy_level_v<_Level>);

  _Meta __meta_{};

public:
  using level_type = _Level;
  using meta_type  = _Meta;

  _CCCL_HIDE_FROM_ABI constexpr hierarchy_level_desc_meta() noexcept = default;

  _CCCL_API constexpr explicit hierarchy_level_desc_meta(_Meta __meta) noexcept
      : __meta_(__meta)
  {}

  [[nodiscard]] _CCCL_API constexpr const _Meta& meta() const noexcept
  {
    return __meta_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const hierarchy_level_desc_meta& __lhs, const hierarchy_level_desc_meta& __rhs) noexcept
  {
    return __lhs.__meta_ == __rhs.__meta_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const hierarchy_level_desc_meta& __lhs, const hierarchy_level_desc_meta& __rhs) noexcept
  {
    return __lhs.__meta_ != __rhs.__meta_;
  }
};

template <class _Cnt, class _Unit>
[[nodiscard]] _CCCL_API constexpr auto grid_dims(_Cnt __count, _Unit) noexcept
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  return hierarchy_level_desc_meta<grid_level, __target_count<_Unit>>{__target_count<_Unit>{__count}};
}

template <class _Cnt, class _Unit>
[[nodiscard]] _CCCL_API constexpr auto cluster_dims(_Cnt __count, _Unit) noexcept
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  return hierarchy_level_desc_meta<cluster_level, __target_count<_Unit>>{__target_count<_Unit>{__count}};
}

template <class _Cnt, class _Unit>
[[nodiscard]] _CCCL_API constexpr auto block_dims(_Cnt __count, _Unit) noexcept
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  return hierarchy_level_desc_meta<block_level, __target_count<_Unit>>{__target_count<_Unit>{__count}};
}

[[nodiscard]] _CCCL_API constexpr auto auto_block_dims() noexcept
{
  return hierarchy_level_desc_meta<block_level, __max_occupancy>{};
}

[[nodiscard]] _CCCL_API constexpr auto fill_device(float __fill_coeff = 1.0f) noexcept
{
  return hierarchy_level_desc_meta<grid_level, __device_fill>{__device_fill{__fill_coeff}};
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_META_LEVEL_DIMENSIONS_H
