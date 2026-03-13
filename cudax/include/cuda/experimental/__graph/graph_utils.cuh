//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_GRAPH_UTILS_CUH
#define _CUDAX__GRAPH_GRAPH_UTILS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/experimental/__graph/path_builder.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <::cuda::std::size_t _Count, ::cuda::std::size_t... _Idx>
[[nodiscard]] _CCCL_HOST_API auto
__replicate_impl(const path_builder& __source, ::cuda::std::index_sequence<_Idx...>)
  -> ::cuda::std::array<path_builder, _Count>
{
  const auto __dev   = __source.get_device();
  const auto __graph = __source.get_native_graph_handle();
  return ::cuda::std::array<path_builder, _Count>{((void) _Idx, path_builder(__dev, __graph))...};
}

template <::cuda::std::size_t _Count, ::cuda::std::size_t... _Idx>
[[nodiscard]] _CCCL_HOST_API auto __replicate_prepend_impl(
  path_builder&& __source, device_ref __dev, cudaGraph_t __graph, ::cuda::std::index_sequence<_Idx...>)
  -> ::cuda::std::array<path_builder, _Count + 1>
{
  return ::cuda::std::array<path_builder, _Count + 1>{
    ::cuda::std::move(__source), ((void) _Idx, path_builder(__dev, __graph))...};
}

//! @brief Create a fixed-size group of peer path builders with the same graph/device as `__source`.
//! @note This function only creates path builders; it does not add synchronization dependencies.
template <::cuda::std::size_t _Count>
[[nodiscard]] _CCCL_HOST_API auto replicate(const path_builder& __source)
  -> ::cuda::std::array<path_builder, _Count>
{
  return __replicate_impl<_Count>(__source, ::cuda::std::make_index_sequence<_Count>{});
}

//! @brief Create a runtime-sized group of peer path builders with the same graph/device as `__source`.
//! @note This function only creates path builders; it does not add synchronization dependencies.
[[nodiscard]] _CCCL_HOST_API inline auto replicate(const path_builder& __source, ::cuda::std::size_t __count)
  -> ::std::vector<path_builder>
{
  const auto __dev   = __source.get_device();
  const auto __graph = __source.get_native_graph_handle();
  ::std::vector<path_builder> __replicas;
  __replicas.reserve(__count);
  for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
  {
    __replicas.emplace_back(__dev, __graph);
  }
  return __replicas;
}

//! @brief Create a runtime-sized group of peer path builders with the same graph/device as `__source` and prepend
//! `__source` at index 0.
//! @note This function only creates path builders; it does not add synchronization dependencies.
[[nodiscard]] _CCCL_HOST_API inline auto replicate_prepend(path_builder&& __source, ::cuda::std::size_t __count)
  -> ::std::vector<path_builder>
{
  const auto __dev   = __source.get_device();
  const auto __graph = __source.get_native_graph_handle();
  ::std::vector<path_builder> __replicas;
  __replicas.reserve(__count + 1);
  __replicas.emplace_back(::cuda::std::move(__source));
  for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
  {
    __replicas.emplace_back(__dev, __graph);
  }
  return __replicas;
}

template <::cuda::std::size_t _Count>
//! @brief Create a fixed-size group of peer path builders with the same graph/device as `__source` and prepend
//! `__source` at index 0.
//! @note This function only creates path builders; it does not add synchronization dependencies.
[[nodiscard]] _CCCL_HOST_API auto replicate_prepend(path_builder&& __source)
  -> ::cuda::std::array<path_builder, _Count + 1>
{
  const auto __dev   = __source.get_device();
  const auto __graph = __source.get_native_graph_handle();
  return __replicate_prepend_impl<_Count>(
    ::cuda::std::move(__source), __dev, __graph, ::cuda::std::make_index_sequence<_Count>{});
}

template <class _Range>
_CCCL_CONCEPT __path_builder_join_range =
  ::cuda::std::is_same_v<::cuda::std::ranges::range_value_t<const _Range&>, path_builder>;

template <class _ToRange, class _FromRange>
_CCCL_HOST_API void __join_impl(_ToRange& __to_builders, const _FromRange& __from_builders)
{
  // TODO: Consider adding dependency deduplication in join to avoid duplicate graph edges.
  for (auto& __to : __to_builders)
  {
    for (const auto& __from : __from_builders)
    {
      __to.depends_on(__from.get_dependencies());
    }
  }
}

//! @brief Synchronize one group of path builders with another by adding dependencies from all `from` into all `to`.
_CCCL_TEMPLATE(class _ToRange, class _FromRange)
_CCCL_REQUIRES(
  ::cuda::std::ranges::forward_range<_ToRange&> _CCCL_AND ::cuda::std::ranges::forward_range<const _FromRange&>
  _CCCL_AND __path_builder_join_range<_ToRange> _CCCL_AND __path_builder_join_range<_FromRange>)
_CCCL_HOST_API void join(_ToRange& __to_builders, const _FromRange& __from_builders)
{
  __join_impl(__to_builders, __from_builders);
}

//! @brief Synchronize a single target path builder with a group of source path builders.
_CCCL_TEMPLATE(class _FromRange)
_CCCL_REQUIRES(
  ::cuda::std::ranges::forward_range<const _FromRange&> _CCCL_AND __path_builder_join_range<_FromRange>)
_CCCL_HOST_API void join(path_builder& __to_builder, const _FromRange& __from_builders)
{
  auto __to_span = ::cuda::std::span<path_builder>(&__to_builder, 1);
  __join_impl(__to_span, __from_builders);
}

//! @brief Synchronize a group of target path builders with a single source path builder.
_CCCL_TEMPLATE(class _ToRange)
_CCCL_REQUIRES(::cuda::std::ranges::forward_range<_ToRange&> _CCCL_AND __path_builder_join_range<_ToRange>)
_CCCL_HOST_API void join(_ToRange& __to_builders, const path_builder& __from_builder)
{
  __join_impl(__to_builders, ::cuda::std::span<const path_builder>(&__from_builder, 1));
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__GRAPH_GRAPH_UTILS_CUH
