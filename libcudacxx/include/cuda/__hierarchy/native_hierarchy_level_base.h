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

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/hierarchy_level_base.h>
#  include <cuda/__hierarchy/hierarchy_query_result.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/is_integer.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// cudafe++ makes the queries (that are device only) return void when compiling for host, which causes host compilers
// to warn about applying [[nodiscard]] to a function that returns void.
_CCCL_DIAG_PUSH
#  if _CCCL_CUDA_COMPILER(NVCC)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
_CCCL_DIAG_SUPPRESS_NVHPC(nodiscard_doesnt_apply)
#  endif // _CCCL_CUDA_COMPILER(NVCC)

template <class _Level>
struct __native_hierarchy_level_base : hierarchy_level_base<_Level>
{
  template <class _InLevel>
  using __default_md_query_type = unsigned;
  template <class _InLevel>
  using __default_1d_query_type = ::cuda::std::size_t;

  using __base_type = hierarchy_level_base<_Level>;
  using __base_type::count;
  using __base_type::count_as;
  using __base_type::dims;
  using __base_type::dims_as;
  using __base_type::extents;
  using __base_type::extents_as;
  using __base_type::static_dims;

#  if _CCCL_CUDA_COMPILATION()
  using __base_type::index;
  using __base_type::index_as;
  using __base_type::rank;
  using __base_type::rank_as;

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto dims(const _InLevel& __level) noexcept
  {
    return _Level::template dims_as<__default_md_query_type<_InLevel>>(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto static_dims(const _InLevel& __level) noexcept
  {
    return __base_type::__static_dims_impl(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto extents(const _InLevel& __level) noexcept
  {
    return _Level::template extents_as<__default_md_query_type<_InLevel>>(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto count(const _InLevel& __level) noexcept
  {
    return _Level::template count_as<__default_1d_query_type<_InLevel>>(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto index(const _InLevel& __level) noexcept
  {
    return _Level::template index_as<__default_md_query_type<_InLevel>>(__level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto rank(const _InLevel& __level) noexcept
  {
    return _Level::template rank_as<__default_1d_query_type<_InLevel>>(__level);
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto dims_as(const _InLevel& __level) noexcept
  {
    return __base_type::template __dims_as_impl<_Tp>(__level);
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto extents_as(const _InLevel& __level) noexcept
  {
    static_assert(__is_natively_reachable_hierarchy_level_v<_Level, _InLevel>,
                  "_InLevel must be reachable from _Level");

    using _NextLevel = typename _Level::__next_native_level;
    auto __next_exts = _NextLevel::template extents_as<_Tp>(__level);
    auto __curr_exts = _Level::template extents_as<_Tp>(_NextLevel{});
    return ::cuda::__hierarchy_extents_mul(__curr_exts, __next_exts);
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto count_as(const _InLevel& __level) noexcept
  {
    return __base_type::template __count_as_impl<_Tp>(__level);
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static auto index_as(const _InLevel& __level) noexcept
  {
    static_assert(__is_natively_reachable_hierarchy_level_v<_Level, _InLevel>,
                  "_InLevel must be reachable from _Level");

    using _NextLevel       = typename _Level::__next_native_level;
    const auto __curr_exts = _Level::template extents_as<_Tp>(_NextLevel{});
    const auto __next_idx  = _NextLevel::template index_as<_Tp>(__level);
    const auto __curr_idx  = _Level::template index_as<_Tp>(_NextLevel{});

    hierarchy_query_result<_Tp> __ret{};
    for (::cuda::std::size_t __i = 0; __i < 3; ++__i)
    {
      __ret[__i] = __curr_idx[__i] + ((__i < __curr_exts.rank()) ? __curr_exts.extent(__i) : 1) * __next_idx[__i];
    }
    return __ret;
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(__is_native_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static _Tp rank_as(const _InLevel& __level) noexcept
  {
    static_assert(__is_natively_reachable_hierarchy_level_v<_Level, _InLevel>,
                  "_InLevel must be reachable from _Level");

    using _NextLevel = typename _Level::__next_native_level;

    const auto __curr_exts = _Level::template extents_as<_Tp>(_NextLevel{});
    const auto __curr_idx  = _Level::template index_as<_Tp>(_NextLevel{});

    _Tp __ret = 0;
    if constexpr (!::cuda::std::is_same_v<_InLevel, _NextLevel>)
    {
      __ret = _NextLevel::template rank_as<_Tp>(__level) * _Level::template count_as<_Tp>(_NextLevel{});
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
};

_CCCL_DIAG_POP

template <>
struct __native_hierarchy_level_base<grid_level> : hierarchy_level_base<grid_level>
{};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_NATIVE_HIERARCHY_LEVEL_BASE_H
