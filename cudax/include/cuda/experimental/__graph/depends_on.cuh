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

#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/array>

#include <cuda/experimental/__graph/fwd.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
//! \brief Builds a tuple of graph nodes that represent dependencies. It is for use as a
//!        parameter to the `graph::add` function.
//!
//! \tparam _Nodes Variadic template parameter representing the types of the graph nodes.
//!         Each type must be either `graph_node` or `graph_node_ref`.
//! \param __nodes The graph nodes to add as dependencies to a new node.
//! \return A object of type `cuda::std::array<cudaGraphNode_t, sizeof...(_Nodes)>`
//!         containing the references to the provided graph nodes.
//!
//! \note A static assertion ensures that all provided arguments are convertible to
//!       `graph_node_ref`. If this condition is not met, a compilation error will occur.
template <class... _Nodes>
_CCCL_TRIVIAL_HOST_API constexpr auto depends_on(const _Nodes&... __nodes) noexcept
  -> _CUDA_VSTD::array<cudaGraphNode_t, sizeof...(_Nodes)>
{
  static_assert((_CUDA_VSTD::is_base_of_v<graph_node_ref, _Nodes> && ...),
                "depends_on() requires graph_node arguments");
  return _CUDA_VSTD::array<cudaGraphNode_t, sizeof...(_Nodes)>{{__nodes.get()...}};
}
} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_GRAPH_DEPENDS_ON
