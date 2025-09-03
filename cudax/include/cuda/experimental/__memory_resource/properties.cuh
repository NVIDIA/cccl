//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH
#define _CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/std/__type_traits/decay.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

using ::cuda::mr::device_accessible;
using ::cuda::mr::host_accessible;

//! @brief A type representing a list of memory resource properties
//! @tparam _Properties The properties to be included in the list
//! It has a member template `rebind` that allows constructing a type by combining
//! a template and type arguments with the properties from this list. The properties
//! are appended after the type arguments in the resulting type.
template <class... _Properties>
struct properties_list
{
  //! @brief A type alias for a type template instantiated with the properties
  //! from this list appended to the type arguments.
  template <template <class...> class _Fn, class... _ExtraArgs>
  using rebind = _Fn<_ExtraArgs..., _Properties...>;
};

template <class _T>
inline constexpr bool __is_queries_list = false;

template <class... _T>
inline constexpr bool __is_queries_list<properties_list<_T...>> = true;

template <typename _Tp>
_CCCL_CONCEPT __has_default_queries =
  _CCCL_REQUIRES_EXPR((_Tp))(requires(__is_queries_list<typename ::cuda::std::decay_t<_Tp>::default_queries>));

template <typename _Resource, bool _HasDefaultQueries = __has_default_queries<_Resource>>
struct __copy_default_queries;

template <typename _Resource>
struct __copy_default_queries<_Resource, true>
{
  using default_queries = typename _Resource::default_queries;
};

template <typename _Resource>
struct __copy_default_queries<_Resource, false>
{};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH
