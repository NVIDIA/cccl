//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_PROPERTIES_H
#define _CUDA___MEMORY_RESOURCE_PROPERTIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/std/__type_traits/decay.h>
#  include <cuda/std/__type_traits/type_set.h>
#  include <cuda/std/cstddef>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_MR

//! @brief The default alignment by a cudaMalloc{...} call
inline constexpr size_t default_cuda_malloc_alignment = 256;

//! @brief The default alignment by a cudaMallocHost{...} call
inline constexpr size_t default_cuda_malloc_host_alignment = alignof(::cuda::std::max_align_t);

//! @brief The device_accessible property signals that the allocated memory is device accessible
struct device_accessible
{};

//! @brief The device_accessible property signals that the allocated memory is host accessible
struct host_accessible
{};

//! @brief determines whether a set of properties signals host accessible memory.
template <class... _Properties>
inline constexpr bool __is_host_accessible =
  ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_Properties...>, host_accessible>;

//! @brief determines whether a set of properties signals device accessible memory.
template <class... _Properties>
inline constexpr bool __is_device_accessible =
  ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_Properties...>, device_accessible>;

//! @brief determines whether a set of properties signals host device accessible memory.
template <class... _Properties>
inline constexpr bool __is_host_device_accessible =
  ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_Properties...>, host_accessible, device_accessible>;

//! @brief verifies that a set of properties contains at least one execution space property
template <class... _Properties>
inline constexpr bool __contains_execution_space_property =
  __is_host_accessible<_Properties...> || __is_device_accessible<_Properties...>;

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

  template <class _QueryProperty>
  _CCCL_HOST_API static constexpr bool has_property([[maybe_unused]] _QueryProperty)
  {
    return ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_Properties...>, _QueryProperty>;
  }
};

template <class _Tp>
inline constexpr bool __is_queries_list = false;

template <class... _Tp>
inline constexpr bool __is_queries_list<properties_list<_Tp...>> = true;

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

enum class __memory_accessability
{
  __host,
  __device,
  __host_device,
};

template <class... _Properties>
struct __memory_accessability_from_properties
{
  static constexpr __memory_accessability value =
    ::cuda::mr::__is_host_device_accessible<_Properties...> ? __memory_accessability::__host_device
    : ::cuda::mr::__is_device_accessible<_Properties...>
      ? __memory_accessability::__device
      : __memory_accessability::__host;
};

_CCCL_END_NAMESPACE_CUDA_MR

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___MEMORY_RESOURCE_PROPERTIES_H
