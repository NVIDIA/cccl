//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_QUERIES_CUH
#define _CUDA_EXPERIMENTAL___GROUP_QUERIES_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__hierarchy/queries/count.h>
#include <cuda/__hierarchy/queries/rank.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Unit, class _Group>
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::size_t __static_count_query_group() noexcept
{
  using _GroupUnit          = typename _Group::unit_type;
  using _GroupMappingResult = typename _Group::__mapping_result_type;

  constexpr auto __group_unit_count = _GroupMappingResult::static_count();

  if constexpr (::cuda::std::is_same_v<_Unit, _GroupUnit>)
  {
    return __group_unit_count;
  }
  else
  {
    using _UnitExts = decltype(_Unit::extents(_GroupUnit{}, ::cuda::std::declval<typename _Group::hierarchy_type>()));

    if constexpr (_UnitExts::rank_dynamic() == 0 && __group_unit_count != ::cuda::std::dynamic_extent)
    {
      auto __ret = __group_unit_count;
      for (::cuda::std::size_t __i = 0; __i < _UnitExts::rank(); ++__i)
      {
        __ret *= _UnitExts::static_extent(__i);
      }
      return __ret;
    }
    else
    {
      return ::cuda::std::dynamic_extent;
    }
  }
}

template <class _Tp, class _Unit, class _Group>
[[nodiscard]] _CCCL_DEVICE_API constexpr _Tp __count_query_group(const _Group& __group) noexcept
{
  using _GroupUnit = typename _Group::unit_type;

  // todo(dabayer): This optimization segfaults the compiler.
  // constexpr auto __static_count = ::cuda::experimental::__static_count_query_group<_Unit, _Group>();
  // if constexpr (__static_count != ::cuda::std::dynamic_extent)
  // {
  //   return static_cast<_Tp>(__static_count);
  // }
  // else
  {
    const auto __group_unit_count = static_cast<_Tp>(__group.__mapping_result().count());
    if constexpr (::cuda::std::is_same_v<_Unit, _GroupUnit>)
    {
      return __group_unit_count;
    }
    else
    {
      const auto __unit_count = __count_query<_Unit, _GroupUnit>::template __call<_Tp>(__group.hierarchy());
      return static_cast<_Tp>(__unit_count * __group_unit_count);
    }
  }
}

template <class _Tp, class _Unit, class _Group>
[[nodiscard]] _CCCL_DEVICE_API _Tp __rank_query_group(const _Group& __group) noexcept
{
  using _GroupUnit = typename _Group::unit_type;

  const auto __group_unit_rank = static_cast<_Tp>(__group.__mapping_result().rank());
  if constexpr (::cuda::std::is_same_v<_Unit, _GroupUnit>)
  {
    return __group_unit_rank;
  }
  else
  {
    const auto __unit_rank        = __rank_query<_Unit, _GroupUnit>::template __call<_Tp>(__group.hierarchy());
    const auto __group_unit_count = ::cuda::experimental::__count_query_group<_Tp, _Unit>(__group);
    return static_cast<_Tp>(__group_unit_rank * __group_unit_count + __unit_rank);
  }
}

template <class _Unit, class _Group>
[[nodiscard]] _CCCL_DEVICE_API bool __is_part_of_group(const _Group& __group) noexcept
{
  return __group.__mapping_result().is_valid();
}
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_QUERIES_CUH
