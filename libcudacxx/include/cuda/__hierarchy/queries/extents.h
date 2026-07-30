//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_QUERIES_EXTENTS_H
#define _CUDA___HIERARCHY_QUERIES_EXTENTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/array>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// helpers

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
      __ret[__i] = static_cast<_Index>(_Ret::static_extent(__i));
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
      __ret[__i] = static_cast<_Index>(_OrgExts::static_extent(__i));
    }
  }
  return ::cuda::std::extents<_Index, _StaticExts...>{__ret};
}

template <class _Tp, class _Unit, class _Level, class _Hierarchy>
[[nodiscard]] _CCCL_API constexpr auto __extents_query_generic(const _Hierarchy& __hier) noexcept
{
  static_assert(__has_bottom_unit_or_level_v<_Unit, _Hierarchy> || __is_native_hierarchy_level_v<_Unit>,
                "_Hierarchy doesn't contain _Unit");
  static_assert(_Hierarchy::has_level(_Level{}) || __is_native_hierarchy_level_v<_Level>,
                "_Hierarchy doesn't contain _Level");

  using _NextLevel = __next_hierarchy_level_t<_Unit, _Hierarchy>;
  using _CurrExts  = decltype(::cuda::__hierarchy_extents_cast<_Tp>(__hier.level(_NextLevel{}).extents()));

  // Remove dependency on runtime storage. This makes the queries work for hierarchy levels with all static extents
  // in constant evaluated context.
  _CurrExts __curr_exts{};
  if constexpr (_CurrExts::rank_dynamic() > 0)
  {
    __curr_exts = ::cuda::__hierarchy_extents_cast<_Tp>(__hier.level(_NextLevel{}).extents());
  }

  if constexpr (!::cuda::std::is_same_v<_NextLevel, _Level>)
  {
    const auto __next_exts = __extents_query<_NextLevel, _Level>::template __call<_Tp>(__hier);
    return ::cuda::__hierarchy_extents_mul(__curr_exts, __next_exts);
  }
  else
  {
    return __curr_exts;
  }
}

// native hierarchy queries

#  if _CCCL_CUDA_COMPILATION()

// cudafe++ makes the queries (that are device only) return void when compiling for host, which causes host compilers
// to warn about applying [[nodiscard]] to a function that returns void.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(nodiscard_doesnt_apply)
#    if _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
#    endif // _CCCL_CUDA_COMPILER(NVCC, <, 13, 0)

template <class _Unit, class _Level>
struct __extents_query_native
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static auto __call() noexcept
  {
    static_assert(__is_natively_reachable_hierarchy_level_v<_Unit, _Level>, "_Level must be reachable from _Unit");

    using _NextLevel       = typename _Unit::__next_native_level;
    const auto __next_exts = __extents_query_native<_NextLevel, _Level>::template __call<_Tp>();
    const auto __curr_exts = __extents_query_native<_Unit, _NextLevel>::template __call<_Tp>();
    return ::cuda::__hierarchy_extents_mul(__curr_exts, __next_exts);
  }
};

template <>
struct __extents_query_native<thread_level, warp_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::extents<_Tp, 32> __call() noexcept
  {
    return {};
  }
};

template <>
struct __extents_query_native<thread_level, block_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, _Tp> __call() noexcept
  {
    return ::cuda::std::dims<3, _Tp>{
      static_cast<_Tp>(blockDim.x), static_cast<_Tp>(blockDim.y), static_cast<_Tp>(blockDim.z)};
  }
};

template <>
struct __extents_query_native<warp_level, block_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<1, _Tp> __call() noexcept
  {
    const auto __thread_count = blockDim.x * blockDim.y * blockDim.z;
    return ::cuda::std::dims<1, _Tp>{static_cast<_Tp>(::cuda::ceil_div(__thread_count, 32))};
  }
};

template <>
struct __extents_query_native<block_level, cluster_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, _Tp> __call() noexcept
  {
    ::dim3 __dims{1u, 1u, 1u};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__dims = ::__clusterDim();))
    return ::cuda::std::dims<3, _Tp>{static_cast<_Tp>(__dims.x), static_cast<_Tp>(__dims.y), static_cast<_Tp>(__dims.z)};
  }
};

template <>
struct __extents_query_native<block_level, grid_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, _Tp> __call() noexcept
  {
    return ::cuda::std::dims<3, _Tp>{
      static_cast<_Tp>(gridDim.x), static_cast<_Tp>(gridDim.y), static_cast<_Tp>(gridDim.z)};
  }
};

template <>
struct __extents_query_native<cluster_level, grid_level>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, _Tp> __call() noexcept
  {
    ::dim3 __dims{gridDim};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__dims = ::__clusterGridDimInClusters();))
    return ::cuda::std::dims<3, _Tp>{static_cast<_Tp>(__dims.x), static_cast<_Tp>(__dims.y), static_cast<_Tp>(__dims.z)};
  }
};

_CCCL_DIAG_POP
#  endif // _CCCL_CUDA_COMPILATION()

// hierarchy queries

template <class _Unit, class _Level>
struct __extents_query
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_API static constexpr auto __call(const _Hierarchy& __hier) noexcept
  {
    return ::cuda::__extents_query_generic<_Tp, _Unit, _Level>(__hier);
  }
};

template <>
struct __extents_query<thread_level, warp_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::extents<_Tp, 32> __call(const _Hierarchy&) noexcept
  {
    static_assert(__has_bottom_unit_or_level_v<thread_level, _Hierarchy>, "_Hierarchy doesn't contain thread_level");
    static_assert(_Hierarchy::template has_level<block_level>(), "_Hierarchy doesn't contain block_level");
    return {};
  }
};

template <class _Level>
struct __extents_query<warp_level, _Level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_API static constexpr auto __call(const _Hierarchy& __hier) noexcept
  {
    auto __block_exts = __extents_query<thread_level, block_level>::template __call<_Tp>(__hier);
    using _BlockExts  = decltype(__block_exts);

    if constexpr (_BlockExts::rank_dynamic() == 0)
    {
      constexpr auto __static_thread_count =
        _BlockExts::static_extent(0) * _BlockExts::static_extent(1) * _BlockExts::static_extent(2);
      static_assert(__static_thread_count >= 32, "_Hierarchy doesn't contain enough threads to fill a single warp");

      constexpr auto __static_warp_count = ::cuda::ceil_div(__static_thread_count, 32);
      ::cuda::std::extents<_Tp, __static_warp_count> __curr_exts{};
      if constexpr (::cuda::std::is_same_v<_Level, block_level>)
      {
        return __curr_exts;
      }
      else
      {
        const auto __next_exts = __extents_query<block_level, _Level>::template __call<_Tp>(__hier);
        return ::cuda::__hierarchy_extents_mul(__curr_exts, __next_exts);
      }
    }
    else
    {
      const auto __thread_count = __block_exts.extent(0) * __block_exts.extent(1) * __block_exts.extent(2);
      _CCCL_ASSERT(__thread_count >= 32, "_Hierarchy doesn't contain enough threads to fill a single warp");

      const auto __warp_count = static_cast<_Tp>(::cuda::ceil_div(__thread_count, 32));
      ::cuda::std::dims<1, _Tp> __curr_exts{__warp_count};
      if constexpr (::cuda::std::is_same_v<_Level, block_level>)
      {
        return __curr_exts;
      }
      else
      {
        const auto __next_exts = __extents_query<block_level, _Level>::template __call<_Tp>(__hier);
        return ::cuda::__hierarchy_extents_mul(__curr_exts, __next_exts);
      }
    }
  }
};

template <>
struct __extents_query<block_level, cluster_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_API static constexpr auto __call(const _Hierarchy& __hier) noexcept
  {
    if constexpr (_Hierarchy::template has_level<cluster_level>())
    {
      return ::cuda::__extents_query_generic<_Tp, block_level, cluster_level>(__hier);
    }
    else
    {
      static_assert(__has_bottom_unit_or_level_v<block_level, _Hierarchy>, "_Hierarchy doesn't contain block_level");
      static_assert(_Hierarchy::template has_level<grid_level>(), "_Hierarchy doesn't contain grid_level");
      return ::cuda::std::extents<_Tp, 1, 1, 1>{};
    }
  }
};

template <>
struct __extents_query<cluster_level, grid_level>
{
  template <class _Tp, class _Hierarchy>
  [[nodiscard]] _CCCL_API static constexpr auto __call(const _Hierarchy& __hier) noexcept
  {
    if constexpr (_Hierarchy::template has_level<cluster_level>())
    {
      return ::cuda::__extents_query_generic<_Tp, cluster_level, grid_level>(__hier);
    }
    else
    {
      return __extents_query<block_level, grid_level>::template __call<_Tp>(__hier);
    }
  }
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_QUERIES_EXTENTS_H
