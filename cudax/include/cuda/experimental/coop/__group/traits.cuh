//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_TRAITS_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_TRAITS_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/experimental/coop/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
template <class _HierarchyLike>
using __hierarchy_type_of _CCCL_NODEBUG =
  ::cuda::std::remove_cvref_t<decltype(::cuda::__unpack_hierarchy_if_needed(::cuda::std::declval<_HierarchyLike>()))>;

template <class _Group>
extern _Group (*__viewed_group_impl)();
template <class _Unit, class _Group, class _MappingResult>
extern _Group (*__viewed_group_impl<group_view<_Unit, _Group, _MappingResult>>)();

template <class _Group>
using __viewed_group_t _CCCL_NODEBUG = decltype(__viewed_group_impl<_Group>());

template <class _Arg, class _ParentGroup>
inline constexpr bool __is_parent_group_query_v = false;
template <class _ParentGroup>
inline constexpr bool __is_parent_group_query_v<_ParentGroup, _ParentGroup> = true;
template <class _ParentGroup>
inline constexpr bool __is_parent_group_query_v<parent_group_t, _ParentGroup> = true;
template <class _ViewUnit, class _ViewMappingResult, class _ParentGroup>
inline constexpr bool __is_parent_group_query_v<group_view<_ViewUnit, _ParentGroup, _ViewMappingResult>, _ParentGroup> =
  ::cuda::std::is_same_v<_ViewUnit, typename _ParentGroup::unit_type>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_TRAITS_CUH
