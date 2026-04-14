//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
#  include <cuda/__hierarchy/queries/extents.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/is_integer.h>

#  if defined(_CUDAX_HIERARCHY)
#    include <cuda/experimental/__hierarchy/fwd.cuh>
#  endif // _CUDAX_HIERARCHY

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Used to either pass-through the hierarchy argument or unpack it from launch configuration
_CCCL_TEMPLATE(class _Type)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Type>)
[[nodiscard]] _CCCL_API constexpr auto& __unpack_hierarchy_if_needed(const _Type& __instance) noexcept
{
  if constexpr (__is_hierarchy_v<_Type>)
  {
    return __instance;
  }
  else
  {
    return __instance.hierarchy();
  }
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
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto dims(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template dims_as<__default_md_query_type<_InLevel>>(
      __level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto static_dims(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return __static_dims_impl(__level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto extents(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template extents_as<__default_md_query_type<_InLevel>>(
      __level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto static_count(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return __static_count_impl(__level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t
  count(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template count_as<__default_1d_query_type<_InLevel>>(
      __level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

#  if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto index(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template index_as<__default_md_query_type<_InLevel>>(
      __level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t
  rank(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return _Level::template rank_as<__default_1d_query_type<_InLevel>>(
      __level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }
#  endif // _CCCL_CUDA_COMPILATION()

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto dims_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return __dims_as_impl<_Tp>(__level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto extents_as(const _InLevel&, const _Hierarchy& __hier) noexcept
  {
    return __extents_query<_Level, _InLevel>::template __call<_Tp>(::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_API static constexpr auto count_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    return __count_as_impl<_Tp>(__level, ::cuda::__unpack_hierarchy_if_needed(__hier));
  }

#  if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  index_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    auto& __hier_unpacked    = ::cuda::__unpack_hierarchy_if_needed(__hier);
    using _HierarchyUnpacked = ::cuda::std::remove_cvref_t<decltype(__hier_unpacked)>;
    static_assert(__has_bottom_unit_or_level_v<_Level, _HierarchyUnpacked>, "_Hierarchy doesn't contain _Level");
    static_assert(_HierarchyUnpacked::template has_level<_InLevel>(), "_Hierarchy doesn't contain _InLevel");

    using _NextLevel = __next_hierarchy_level_t<_Level, _HierarchyUnpacked>;
    if constexpr (::cuda::std::is_same_v<_InLevel, _NextLevel>)
    {
      using _CurrExts = decltype(_Level::template extents_as<_Tp>(_NextLevel{}, __hier_unpacked));
      auto __curr_idx = _Level::template index_as<_Tp>(_NextLevel{});
      for (::cuda::std::size_t __i = 0; __i < 3; ++__i)
      {
        if (__i >= _CurrExts::rank() || _CurrExts::static_extent(__i) == 1)
        {
          __curr_idx[__i] = 0;
        }
      }
      return __curr_idx;
    }
    else
    {
      const auto __curr_exts = _Level::template extents_as<_Tp>(_NextLevel{}, __hier_unpacked);
      const auto __next_idx  = _NextLevel::template index_as<_Tp>(__level, __hier_unpacked);
      const auto __curr_idx  = _Level::template index_as<_Tp>(_NextLevel{}, __hier_unpacked);

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
                   __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  rank_as(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  {
    auto& __hier_unpacked    = ::cuda::__unpack_hierarchy_if_needed(__hier);
    using _HierarchyUnpacked = ::cuda::std::remove_cvref_t<decltype(__hier_unpacked)>;
    static_assert(__has_bottom_unit_or_level_v<_Level, _HierarchyUnpacked>, "_Hierarchy doesn't contain _Level");
    static_assert(_HierarchyUnpacked::template has_level<_InLevel>(), "_Hierarchy doesn't contain _InLevel");

    using _NextLevel = __next_hierarchy_level_t<_Level, _HierarchyUnpacked>;

    const auto __curr_exts = _Level::template extents_as<_Tp>(_NextLevel{}, __hier_unpacked);
    const auto __curr_idx  = _Level::template index_as<_Tp>(_NextLevel{}, __hier_unpacked);

    _Tp __ret = 0;
    if constexpr (!::cuda::std::is_same_v<_InLevel, _NextLevel>)
    {
      __ret = _NextLevel::template rank_as<_Tp>(__level, __hier_unpacked)
            * _Level::template count_as<_Tp>(_NextLevel{}, __hier_unpacked);
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

#  if defined(_CUDAX_HIERARCHY)
  _CCCL_TEMPLATE(class _Tp, class _Group)
  _CCCL_REQUIRES(
    ::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND ::cuda::experimental::__is_this_hierarchy_group_v<_Group>)
  [[nodiscard]] _CCCL_API static constexpr _Tp count_as(const _Group& __group) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_Level, typename _Group::level_type>)
    {
      return _Tp{1};
    }
    else
    {
      return _Level::template count_as<_Tp>(typename _Group::level_type{}, __group.hierarchy());
    }
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::__is_this_hierarchy_group_v<_Group>)
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t count(const _Group& __group) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_Level, typename _Group::level_type>)
    {
      return 1;
    }
    else
    {
      return _Level::count(typename _Group::level_type{}, __group.hierarchy());
    }
  }

#    if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _Tp, class _Group)
  _CCCL_REQUIRES(
    ::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND ::cuda::experimental::__is_this_hierarchy_group_v<_Group>)
  [[nodiscard]] _CCCL_API static constexpr _Tp rank_as(const _Group& __group) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_Level, typename _Group::level_type>)
    {
      return _Tp{0};
    }
    else
    {
      // todo: Pass __group.hierarchy() to the query.
      return _Level::template rank_as<_Tp>(typename _Group::level_type{} /*, __group.hierarchy()*/);
    }
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::__is_this_hierarchy_group_v<_Group>)
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t rank(const _Group& __group) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_Level, typename _Group::level_type>)
    {
      return 0;
    }
    else
    {
      // todo: Pass __group.hierarchy() to the query.
      return _Level::rank(typename _Group::level_type{} /*, __group.hierarchy()*/);
    }
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::__is_this_hierarchy_group_v<_Group>)
  [[nodiscard]] _CCCL_API static constexpr bool is_root_rank(const _Group& __group) noexcept
  {
    return _Level::rank(__group) == 0;
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::__is_this_hierarchy_group_v<_Group>)
  [[nodiscard]] _CCCL_API static constexpr bool is_part_of(const _Group& __group) noexcept
  {
    return true;
  }
#    endif // _CCCL_CUDA_COMPILATION()
#  endif // _CUDAX_HIERARCHY

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

  template <class... _Args>
  [[nodiscard]] _CCCL_API static constexpr auto __static_count_impl(const _Args&... __args) noexcept
  {
    using _Exts = decltype(_Level::extents(__args...));

    if constexpr (_Exts::rank_dynamic() == 0)
    {
      ::cuda::std::size_t __ret{1};
      for (::cuda::std::size_t __i = 0; __i < _Exts::rank(); ++__i)
      {
        __ret *= _Exts::static_extent(__i);
      }
      return __ret;
    }
    else
    {
      return ::cuda::std::dynamic_extent;
    }
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
