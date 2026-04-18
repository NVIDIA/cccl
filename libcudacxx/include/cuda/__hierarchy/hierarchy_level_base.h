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
#  include <cuda/__hierarchy/queries/count.h>
#  include <cuda/__hierarchy/queries/extents.h>
#  include <cuda/__hierarchy/queries/index.h>
#  include <cuda/__hierarchy/queries/rank.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/is_integer.h>

#  if defined(_CUDAX_GROUP)
#    include <cuda/experimental/__group/concepts.cuh>
#    include <cuda/experimental/__group/fwd.cuh>
#    include <cuda/experimental/__group/queries.cuh>
#  endif // _CUDAX_GROUP

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
  [[nodiscard]] _CCCL_API static constexpr auto count_as(const _InLevel&, const _Hierarchy& __hier) noexcept
  {
    return __count_query<_Level, _InLevel>::template __call<_Tp>(::cuda::__unpack_hierarchy_if_needed(__hier));
  }

#  if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto index_as(const _InLevel&, const _Hierarchy& __hier) noexcept
  {
    return __index_query<_Level, _InLevel>::template __call<_Tp>(::cuda::__unpack_hierarchy_if_needed(__hier));
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Hierarchy)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND
                   __is_or_has_hierarchy_member_v<_Hierarchy>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto rank_as(const _InLevel&, const _Hierarchy& __hier) noexcept
  {
    return __rank_query<_Level, _InLevel>::template __call<_Tp>(::cuda::__unpack_hierarchy_if_needed(__hier));
  }
#  endif // _CCCL_CUDA_COMPILATION()

#  if defined(_CUDAX_GROUP)
#    if _CCCL_CUDA_COMPILATION()

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::is_group<_Group>)
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t static_count(const _Group&) noexcept
  {
    return ::cuda::experimental::__static_count_query_group<_Level, _Group>();
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::is_group<_Group>)
  [[nodiscard]] _CCCL_API static constexpr auto count(const _Group& __group) noexcept
  {
    return count_as<__default_1d_query_type<typename _Group::unit_type>>(__group);
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::is_group<_Group>)
  [[nodiscard]] _CCCL_API static auto rank(const _Group& __group) noexcept
  {
    return rank_as<__default_1d_query_type<typename _Group::unit_type>>(__group);
  }

  _CCCL_TEMPLATE(class _Tp, class _Group)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND ::cuda::experimental::is_group<_Group>)
  [[nodiscard]] _CCCL_API static constexpr _Tp count_as(const _Group& __group) noexcept
  {
    return ::cuda::experimental::__count_query_group<_Tp, _Level>(__group);
  }

  _CCCL_TEMPLATE(class _Tp, class _Group)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND ::cuda::experimental::is_group<_Group>)
  [[nodiscard]] _CCCL_API static _Tp rank_as(const _Group& __group) noexcept
  {
    return ::cuda::experimental::__rank_query_group<_Tp, _Level>(__group);
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::is_group<_Group>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_root_rank(const _Group& __group) noexcept
  {
    return _Level::rank(__group) == 0;
  }

  _CCCL_TEMPLATE(class _Group)
  _CCCL_REQUIRES(::cuda::experimental::is_group<_Group>)
  [[nodiscard]] _CCCL_API static constexpr bool is_part_of(const _Group& __group) noexcept
  {
    // todo: static_assert that the _Level <= _Group::unit_type
    return ::cuda::experimental::__is_part_of_group<_Level>(__group);
  }
#    endif // _CCCL_CUDA_COMPILATION()
#  endif // _CUDAX_GROUP

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
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_HIERARCHY_LEVEL_BASE_H
