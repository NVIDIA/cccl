//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_NATIVE_HIERARCHY_LEVEL_BASE_H
#define _CUDA___HIERARCHY_NATIVE_HIERARCHY_LEVEL_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/hierarchy.h>
#include <cuda/__hierarchy/hierarchy_level_base.h>
#include <cuda/__hierarchy/hierarchy_query_result.h>
#include <cuda/__hierarchy/traits.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Op>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL ::cuda::std::size_t
__merge_static_extents(::cuda::std::size_t __lhs, ::cuda::std::size_t __rhs) noexcept
{
  if (__lhs == ::cuda::std::dynamic_extent || __rhs == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::dynamic_extent;
  }
  else
  {
    return _Op{}(__lhs, __rhs);
  }
}

template <class _ResultIndex, class _Op, class _LhsExts, class _RhsExts, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_API constexpr auto __extents_op_impl(::cuda::std::index_sequence<_Is...>) noexcept
{
  if constexpr (_LhsExts::rank() == _RhsExts::rank())
  {
    return ::cuda::std::extents<
      _ResultIndex,
      ::cuda::__merge_static_extents<_Op>(_LhsExts::static_extent(_Is), _RhsExts::static_extent(_Is))...>{};
  }
  else
  {
    // todo: make this work when ranks don't match
    static_assert(::cuda::std::__always_false_v<_Op>);
  }
}

template <class _Op, class _LhsExts, class _RhsExts>
[[nodiscard]] _CCCL_API constexpr auto __extents_op(const _LhsExts& __lhs, const _RhsExts& __rhs) noexcept
{
  using _ResultIndex = ::cuda::std::common_type_t<typename _LhsExts::index_type, typename _RhsExts::index_type>;
  constexpr ::cuda::std::size_t __result_rank = ::cuda::std::max(_LhsExts::rank(), _RhsExts::rank());
  using _Result = decltype(::cuda::__extents_op_impl<_ResultIndex, _Op, _LhsExts, _RhsExts>(
    ::cuda::std::make_index_sequence<__result_rank>{}));

  // todo: make this work when ranks don't match
  static_assert(_LhsExts::rank() == _RhsExts::rank());
  ::cuda::std::array<_ResultIndex, __result_rank> __ret{};
  _Op __op{};
  for (::cuda::std::size_t __i = 0; __i < __result_rank; ++__i)
  {
    if (_Result::static_extent(__i) == ::cuda::std::dynamic_extent)
    {
      __ret[__i] = __op(__lhs.extent(__i), __rhs.extent(__i));
    }
    else
    {
      __ret[__i] = _Result::static_extent(__i);
    }
  }
  return _Result{__ret};
}

template <class _LhsIndex, ::cuda::std::size_t... _LhsExts, class _RhsIndex, ::cuda::std::size_t... _RhsExts>
[[nodiscard]] _CCCL_API constexpr auto __extents_add(const ::cuda::std::extents<_LhsIndex, _LhsExts...>& __lhs,
                                                     const ::cuda::std::extents<_RhsIndex, _RhsExts...>& __rhs) noexcept
{
  return ::cuda::__extents_op<::cuda::std::plus<>>(__lhs, __rhs);
}

template <class _LhsIndex, ::cuda::std::size_t... _LhsExts, class _RhsIndex, ::cuda::std::size_t... _RhsExts>
[[nodiscard]] _CCCL_API constexpr auto __extents_mul(const ::cuda::std::extents<_LhsIndex, _LhsExts...>& __lhs,
                                                     const ::cuda::std::extents<_RhsIndex, _RhsExts...>& __rhs) noexcept
{
  return ::cuda::__extents_op<::cuda::std::multiplies<>>(__lhs, __rhs);
}

template <class _Level>
struct __native_hierarchy_level_base : hierarchy_level_base<_Level>
{
  // todo: use this once cuda::hierarchy is implemented
  //
  // using __base_type = hierarchy_level_base<_Level>;
  // using __base_type::count;
  // using __base_type::dims;
  // using __base_type::extents;
  // using __base_type::index;
  // using __base_type::rank;
  // using __base_type::static_dims;

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto dims(const _InLevel& __level) noexcept
  {
    return __dims_impl(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto static_dims(const _InLevel& __level) noexcept
  {
    return __static_dims_impl(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto extents(const _InLevel& __level) noexcept
  {
    static_assert(__is_natively_reachable_hierarchy_level_v<_Level, _InLevel>,
                  "_InLevel must be reachable from _Level");

    using _NextLevel = typename _Level::__next_level;
    auto __next_exts = _NextLevel::extents(__level);
    auto __curr_exts = _Level::extents(_NextLevel{});
    return ::cuda::__extents_mul(__curr_exts, __next_exts);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::size_t count(const _InLevel& __level) noexcept
  {
    return __count_impl(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto index(const _InLevel& __level) noexcept
  {
    static_assert(__is_natively_reachable_hierarchy_level_v<_Level, _InLevel>,
                  "_InLevel must be reachable from _Level");

    using _NextLevel       = typename _Level::__next_level;
    const auto __curr_exts = _Level::extents(_NextLevel{});
    const auto __next_idx  = _NextLevel::index(__level);
    const auto __curr_idx  = _Level::index(_NextLevel{});

    using _CurrExts = decltype(__curr_exts);
    using _NextIdx  = decltype(__next_idx);
    using _CurrIdx  = decltype(__curr_idx);

    constexpr auto __curr_rank = _CurrExts::rank();
    constexpr auto __next_rank = _CurrIdx::__rank;

    using _Val = ::cuda::std::
      common_type_t<typename _CurrExts::index_type, typename _NextIdx::value_type, typename _CurrIdx::value_type>;
    constexpr ::cuda::std::size_t __rank = cuda::std::max(__curr_rank, __next_rank);

    hierarchy_query_result<_Val, __rank> __ret{};
    for (::cuda::std::size_t __i = 0; __i < __rank; ++__i)
    {
      if constexpr (__curr_rank == __next_rank)
      {
        __ret[__i] = __curr_idx[__i] + __curr_exts.extent(__i) * __next_idx[__i];
      }
      else
      {
        // todo: make this work when ranks don't match
        static_assert(__curr_rank == __next_rank);
      }
    }
    return __ret;
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::size_t rank(const _InLevel& __level) noexcept
  {
    return __rank_impl(__level);
  }
};

template <>
struct __native_hierarchy_level_base<grid_level> : hierarchy_level_base<grid_level>
{};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___HIERARCHY_NATIVE_HIERARCHY_LEVEL_BASE_H
