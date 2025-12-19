//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_HIERARCHY_LEVEL_BASE_H
#define _CUDA___HIERARCHY_HIERARCHY_LEVEL_BASE_H

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
#  include <cuda/__hierarchy/hierarchy_query_result.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/is_integer.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/array>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL ::cuda::std::size_t
__hierarchy_static_extents_mul_helper(::cuda::std::size_t __lhs, ::cuda::std::size_t __rhs) noexcept
{
  if (__lhs == ::cuda::std::dynamic_extent || __rhs == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::dynamic_extent;
  }
  else
  {
    return __lhs * __rhs;
  }
}

template <class _ResultIndex, class _LhsExts, class _RhsExts, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_API constexpr auto __hierarchy_static_extents_mul(::cuda::std::index_sequence<_Is...>) noexcept
{
  return ::cuda::std::extents<
    _ResultIndex,
    ::cuda::__hierarchy_static_extents_mul_helper((_Is < _LhsExts::rank()) ? _LhsExts::static_extent(_Is) : 1,
                                                  (_Is < _RhsExts::rank()) ? _RhsExts::static_extent(_Is) : 1)...>{};
}

//! @brief Multiplies 2 extents in column major order together, returning a new extents type. If the ranks don't match,
//!        the extent with lower rank is padded with 1s on the right to match the rank of the other.
//!
//! @param __lhs The left hand side extents to multiply.
//! @param __rhs The right hand side extents to multiply.
//!
//! @return The result of multiplying the extents together.
template <class _Index, ::cuda::std::size_t... _LhsExts, ::cuda::std::size_t... _RhsExts>
[[nodiscard]] _CCCL_API constexpr auto
__hierarchy_extents_mul(const ::cuda::std::extents<_Index, _LhsExts...>& __lhs,
                        const ::cuda::std::extents<_Index, _RhsExts...>& __rhs) noexcept
{
  using _Lhs = ::cuda::std::extents<_Index, _LhsExts...>;
  using _Rhs = ::cuda::std::extents<_Index, _RhsExts...>;

  constexpr auto __rank = ::cuda::std::max(_Lhs::rank(), _Rhs::rank());
  using _Ret =
    decltype(::cuda::__hierarchy_static_extents_mul<_Index, _Lhs, _Rhs>(::cuda::std::make_index_sequence<__rank>{}));

  ::cuda::std::array<_Index, __rank> __ret{};
  for (::cuda::std::size_t __i = 0; __i < __rank; ++__i)
  {
    if (_Ret::static_extent(__i) == ::cuda::std::dynamic_extent)
    {
      __ret[__i] = static_cast<_Index>((__i < _Lhs::rank()) ? __lhs.extent(__i) : 1)
                 * static_cast<_Index>((__i < _Rhs::rank()) ? __rhs.extent(__i) : 1);
    }
    else
    {
      __ret[__i] = _Ret::static_extent(__i);
    }
  }
  return _Ret{__ret};
}

template <class _Index, class _OrgIndex, ::cuda::std::size_t... _StaticExts>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::extents<_Index, _StaticExts...>
__hierarchy_extents_cast(::cuda::std::extents<_OrgIndex, _StaticExts...> __org_exts) noexcept
{
  using _OrgExts = ::cuda::std::extents<_OrgIndex, _StaticExts...>;
  ::cuda::std::array<_Index, _OrgExts::rank()> __ret{};
  for (::cuda::std::size_t __i = 0; __i < _OrgExts::rank(); ++__i)
  {
    if (_OrgExts::static_extent(__i) == ::cuda::std::dynamic_extent)
    {
      __ret[__i] = static_cast<_Index>(__org_exts.extent(__i));
    }
    else
    {
      __ret[__i] = _OrgExts::static_extent(__i);
    }
  }
  return ::cuda::std::extents<_Index, _StaticExts...>{__ret};
}

template <class _Level>
struct hierarchy_level_base
{
  using level_type = _Level;

  template <class _InLevel>
  using __default_md_query_type = unsigned;
  template <class _InLevel>
  using __default_1d_query_type = ::cuda::std::size_t;

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto dims(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template dims_as<__default_md_query_type<_InLevel>>(__level, __hier);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto static_dims(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return __static_dims_impl(__level, __hier);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto extents(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template extents_as<__default_md_query_type<_InLevel>>(__level, __hier);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t
  count(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template count_as<__default_1d_query_type<_InLevel>>(__level, __hier);
  }

#  if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto index(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template index_as<__default_md_query_type<_InLevel>>(__level, __hier);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t
  rank(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template rank_as<__default_1d_query_type<_InLevel>>(__level, __hier);
  }
#  endif // _CCCL_CUDA_COMPILATION()

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto dims_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return __dims_as_impl<_Tp>(__level, __hier);
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto extents_as(const _InLevel& __in_level, const _Hierarchy& __hier) noexcept
  {
    static_assert(has_unit_or_level_v<_Level, _Hierarchy>, "_Hierarchy doesn't contain _Level");
    static_assert(has_level_v<_InLevel, _Hierarchy>, "_Hierarchy doesn't contain _InLevel");

    using _NextLevel = __next_hierarchy_level_t<_Level, _Hierarchy>;
    using _CurrExts  = decltype(::cuda::__hierarchy_extents_cast<_Tp>(__hier.level(_NextLevel{}).dims));

    // Remove dependency on runtime storage. This makes the queries work for hierarchy levels with all static extents
    // in constant evaluated context.
    _CurrExts __curr_exts{};
    if constexpr (_CurrExts::rank_dynamic() > 0)
    {
      __curr_exts = ::cuda::__hierarchy_extents_cast<_Tp>(__hier.level(_NextLevel{}).dims);
    }

    if constexpr (!::cuda::std::is_same_v<_NextLevel, _InLevel>)
    {
      const auto __next_exts = _NextLevel::template extents_as<_Tp>(__in_level, __hier);
      return ::cuda::__hierarchy_extents_mul(__curr_exts, __next_exts);
    }
    else
    {
      return __curr_exts;
    }
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto count_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return __count_as_impl<_Tp>(__level, __hier);
  }

#  if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  index_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    static_assert(has_unit_or_level_v<_Level, _Hierarchy>, "_Hierarchy doesn't contain _Level");
    static_assert(has_level_v<_InLevel, _Hierarchy>, "_Hierarchy doesn't contain _InLevel");

    using _NextLevel = __next_hierarchy_level_t<_Level, _Hierarchy>;
    if constexpr (::cuda::std::is_same_v<_InLevel, _NextLevel>)
    {
      using _CurrExts = decltype(_Level::template extents_as<_Tp>(_NextLevel{}, __hier));
      auto __curr_idx = _Level::template index_as<_Tp>(_NextLevel{});
      for (::cuda::std::size_t __i = _CurrExts::rank(); __i < 3; ++__i)
      {
        __curr_idx[__i] = 0;
      }
      return __curr_idx;
    }
    else
    {
      const auto __curr_exts = _Level::template extents_as<_Tp>(_NextLevel{}, __hier);
      const auto __next_idx  = _NextLevel::template index_as<_Tp>(__level, __hier);
      const auto __curr_idx  = _Level::template index_as<_Tp>(_NextLevel{}, __hier);

      hierarchy_query_result<_Tp> __ret{};
      for (::cuda::std::size_t __i = 0; __i < 3; ++__i)
      {
        __ret[__i] = __curr_idx[__i] + ((__i < __curr_exts.rank()) ? __curr_exts.extent(__i) : 1) * __next_idx[__i];
      }
      return __ret;
    }
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_hierarchy_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  rank_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    static_assert(has_unit_or_level_v<_Level, _Hierarchy>, "_Hierarchy doesn't contain _Level");
    static_assert(has_level_v<_InLevel, _Hierarchy>, "_Hierarchy doesn't contain _InLevel");

    using _NextLevel = __next_hierarchy_level_t<_Level, _Hierarchy>;

    const auto __curr_exts = _Level::template extents_as<_Tp>(_NextLevel{}, __hier);
    const auto __curr_idx  = _Level::template index_as<_Tp>(_NextLevel{}, __hier);

    _Tp __ret = 0;
    if constexpr (!::cuda::std::is_same_v<_InLevel, _NextLevel>)
    {
      __ret = _NextLevel::template rank_as<_Tp>(__level, __hier) * _Level::template count_as<_Tp>(_NextLevel{}, __hier);
    }

    for (::cuda::std::size_t __i = __curr_exts.rank(); __i > 0; --__i)
    {
      _Tp __inc = __curr_idx[__i - 1];
      for (::cuda::std::size_t __j = __i - 1; __j > 0; --__j)
      {
        __inc *= __curr_exts.extent(__j - 1);
      }
      __ret += __inc;
    }
    return __ret;
  }
#  endif // _CCCL_CUDA_COMPILATION()

private:
  template <class>
  friend struct __native_hierarchy_level_base;

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class... _Args>
  [[nodiscard]] _CCCL_API static constexpr auto __dims_as_impl(const _Args&... __args) noexcept
  {
    auto __exts = _Level::template extents_as<_Tp>(__args...);
    using _Exts = decltype(__exts);

    hierarchy_query_result<_Tp> __ret{1, 1, 1};
    for (::cuda::std::size_t __i = 0; __i < _Exts::rank(); ++__i)
    {
      __ret[__i] = __exts.extent(__i);
    }
    return __ret;
  }

  template <class... _Args>
  [[nodiscard]] _CCCL_API static constexpr auto __static_dims_impl(const _Args&... __args) noexcept
  {
    using _Exts = decltype(_Level::extents(__args...));

    hierarchy_query_result<::cuda::std::size_t> __ret{1, 1, 1};
    for (::cuda::std::size_t __i = 0; __i < _Exts::rank(); ++__i)
    {
      __ret[__i] = _Exts::static_extent(__i);
    }
    return __ret;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class... _Args>
  [[nodiscard]] _CCCL_API static constexpr _Tp __count_as_impl(const _Args&... __args) noexcept
  {
    const auto __exts = _Level::template extents_as<_Tp>(__args...);

    _Tp __ret = 1;
    for (::cuda::std::size_t __i = 0; __i < __exts.rank(); ++__i)
    {
      __ret *= __exts.extent(__i);
    }
    return __ret;
  }
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_HIERARCHY_LEVEL_BASE_H
