//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_DEPENDS_ON
#define __CUDAX_GRAPH_DEPENDS_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/array>

#include <cuda/experimental/__graph/fwd.cuh>
#include <cuda/experimental/__graph/graph_node_ref.cuh>

#include <cuda_runtime_api.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! \brief Builds an array of graph nodes that represent dependencies. It is for use as a
//!        parameter to the `graph_builder::add` function.
//!
//! \tparam _Nodes Variadic template parameter representing the types of the graph nodes.
//!         Each type must be either `graph_node_ref` or `cudaGraphNode_t`.
//! \param __nodes The graph nodes to add as dependencies to a new node.
//! \return A object of type `cuda::std::array<cudaGraphNode_t, sizeof...(_Nodes)>`
//!         containing the references to the provided graph nodes.
//!
//! \note A static assertion ensures that all provided arguments are convertible to
//!       `graph_node_ref`. If this condition is not met, a compilation error will occur.
// TODO graph_node_ref needs a graph argument if this function would accept cudaGraphNode_t
// TODO we should consider defining a type that also wraps a device and a graph and making it a graph_inserter,
//      and then we could return it here. It would serve as a non-advancing alternative to path_builder.
template <class... _Nodes>
_CCCL_NODEBUG_HOST_API constexpr auto depends_on(const _Nodes&... __nodes) noexcept
  -> ::cuda::std::array<cudaGraphNode_t, sizeof...(_Nodes)>
{
  return ::cuda::std::array<cudaGraphNode_t, sizeof...(_Nodes)>{{graph_node_ref(__nodes).get()...}};
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_DEPENDS_ON
