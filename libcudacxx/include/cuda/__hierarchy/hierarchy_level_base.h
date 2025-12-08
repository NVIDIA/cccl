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

#include <cuda/__fwd/hierarchy.h>
#include <cuda/__hierarchy/hierarchy_query_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Level>
struct hierarchy_level_base
{
  using level_type = _Level;

  // todo: use this once cuda::hierarchy is being implemented
  //
  // _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  // _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  // [[nodiscard]] _CCCL_DEVICE_API static constexpr auto dims(const _InLevel& __level, const _Hierarchy& __hier)
  // noexcept
  // {
  //   return __dims_impl(__level, __hier);
  // }

  // _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  // _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  // [[nodiscard]] _CCCL_DEVICE_API static constexpr auto static_dims(const _InLevel& __level, const _Hierarchy& __hier)
  // noexcept
  // {
  //   return __static_dims_impl(__level, __hier);
  // }

  // _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  // _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  // [[nodiscard]] _CCCL_DEVICE_API static constexpr auto extents(const _InLevel& __level, const _Hierarchy& __hier)
  // noexcept {}

  // _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  // _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  // [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t
  // count(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  // {
  //   return __count_impl(__level, __hier);
  // }

  // _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  // _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  // [[nodiscard]] _CCCL_DEVICE_API static constexpr auto index(const _InLevel& __level, const _Hierarchy& __hier)
  // noexcept
  // {}

  // _CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
  // _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_hierarchy_v<_Hierarchy>)
  // [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t
  // rank(const _InLevel& __level, const _Hierarchy& __hier) noexcept
  // {
  //   return __rank_impl(__level, __hier);
  // }

private:
  template <class>
  friend struct __native_hierarchy_level_base;

  template <class... _Args>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto __dims_impl(const _Args&... __args) noexcept
  {
    auto __exts = _Level::extents(__args...);
    using _Exts = decltype(__exts);

    hierarchy_query_result<typename _Exts::index_type, _Exts::rank()> __ret{};
    for (::cuda::std::size_t __i = 0; __i < _Exts::rank(); ++__i)
    {
      __ret[__i] = __exts.extent(__i);
    }
    return __ret;
  }

  template <class... _Args>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto __static_dims_impl(const _Args&... __args) noexcept
  {
    using _Exts = decltype(_Level::extents(__args...));

    hierarchy_query_result<::cuda::std::size_t, _Exts::rank()> __ret{};
    for (::cuda::std::size_t __i = 0; __i < _Exts::rank(); ++__i)
    {
      __ret[__i] = _Exts::static_extent(__i);
    }
    return __ret;
  }

  template <class... _Args>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t __count_impl(const _Args&... __args) noexcept
  {
    const auto __exts = _Level::extents(__args...);

    ::cuda::std::size_t __ret = 1;
    for (::cuda::std::size_t __i = 0; __i < __exts.rank(); ++__i)
    {
      __ret *= __exts.extent(__i);
    }
    return __ret;
  }

  template <class... _Args>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t __rank_impl(const _Args&... __args) noexcept
  {
    const auto __exts = _Level::extents(__args...);
    const auto __idcs = _Level::index(__args...);

    ::cuda::std::size_t __ret = 0;
    for (::cuda::std::size_t __i = 0; __i < __exts.rank(); ++__i)
    {
      ::cuda::std::size_t __inc = __idcs[__i];
      for (::cuda::std::size_t __j = __i + 1; __j < __exts.rank(); ++__j)
      {
        __inc *= __exts.extent(__j);
      }
      __ret += __inc;
    }
    return __ret;
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___HIERARCHY_HIERARCHY_LEVEL_BASE_H
