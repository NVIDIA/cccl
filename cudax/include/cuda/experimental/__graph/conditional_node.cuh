//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_CONDITIONAL_NODE_CUH
#define _CUDAX__GRAPH_CONDITIONAL_NODE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(12, 4)

#  include <cuda/experimental/__driver/driver_api.cuh>
#  include <cuda/experimental/__graph/graph_builder_ref.cuh>
#  include <cuda/experimental/__graph/graph_node_ref.cuh>
#  include <cuda/experimental/__graph/path_builder.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! \brief A thin, non-owning wrapper around a `cudaGraphConditionalHandle`.
//!
//! A conditional handle is a graph-scoped token whose value at runtime controls whether
//! the body of an `if` or `while` conditional node executes.  The handle is owned by
//! the graph — there is no destroy API — so this wrapper is trivially copyable and
//! safe to pass by value into device kernels.
//!
//! Users can construct a handle directly, or let `make_if_node` / `make_while_node`
//! create one automatically.
//!
//! \rst
//! .. _cudax-graph-conditional-handle:
//! \endrst
struct conditional_handle
{
  //! \brief Creates a conditional handle for the given graph.
  //!
  //! \param __graph       Graph in which the conditional node will be inserted.
  //! \param __default_val Initial value of the handle (true = execute body, false = skip).
  //! \throws cuda::std::cuda_error if `cudaGraphConditionalHandleCreate` fails.
  _CCCL_HOST_API explicit conditional_handle(graph_builder_ref __graph, bool __default_val = true)
      : __handle_(::cuda::experimental::__driver::__graphConditionalHandleCreate(
          __graph.get(), __graph.get_device().__primary_context(), __default_val, ::cudaGraphCondAssignDefault))
  {}

  //! \brief Sets the runtime value of the conditional handle from device code.
  //!
  //! \param __value Non-zero to execute the body, zero to skip.
  _CCCL_DEVICE void set_value(bool __value) const noexcept
  {
    ::cudaGraphSetConditional(__handle_, __value);
  }

  //! \brief Convenience: enables execution of the conditional body (sets the handle to 1).
  _CCCL_DEVICE void enable() const noexcept
  {
    set_value(true);
  }

  //! \brief Convenience: disables execution of the conditional body (sets the handle to 0).
  _CCCL_DEVICE void disable() const noexcept
  {
    set_value(false);
  }

  //! \brief Returns the underlying `cudaGraphConditionalHandle`.
  [[nodiscard]] _CCCL_NODEBUG_HOST_API ::cudaGraphConditionalHandle get() const noexcept
  {
    return __handle_;
  }

private:
  ::cudaGraphConditionalHandle __handle_{};
};

//! \brief Result of adding a conditional node.
//!
//! Contains the newly created conditional node, the body graph that should be
//! populated by the caller, and the conditional handle to pass into body kernels.
struct conditional_node_result
{
  graph_node_ref node; //!< The conditional node in the parent graph.
  graph_builder_ref body_graph; //!< The body graph to populate with operations.
  conditional_handle handle; //!< The handle to control execution from device code.
};

_CCCL_HOST_API inline conditional_node_result
__make_conditional_node(path_builder& __pb, conditional_handle __handle, ::CUgraphConditionalNodeType __type)
{
  auto __deps = __pb.get_dependencies();

  ::CUgraphNodeParams __params{};
  __params.type               = ::CU_GRAPH_NODE_TYPE_CONDITIONAL;
  __params.conditional.handle = __handle.get();
  __params.conditional.type   = __type;
  __params.conditional.size   = 1;
  __params.conditional.ctx    = __pb.get_device().__primary_context();

  auto __node = ::cuda::experimental::__driver::__graphAddNode(
    __pb.get_native_graph_handle(), __deps.data(), __deps.size(), &__params);

  __pb.__clear_and_set_dependency_node(__node);

  return {graph_node_ref{__node, __pb.get_native_graph_handle()},
          graph_builder_ref{__params.conditional.phGraph_out[0], __pb.get_device()},
          __handle};
}
//! \brief Adds an `if`-conditional node to a CUDA graph path.
//!
//! At runtime, if the value of the handle is non-zero the body graph executes once;
//! otherwise it is skipped entirely.
//!
//! The caller must populate the returned `body_graph` with all operations that should
//! run conditionally before the parent graph is instantiated.
//!
//! \param __pb          Path builder to insert the node into.
//! \param __default_val Initial handle value (true = execute, false = skip). Ignored when
//!                      \p __handle is provided.
//! \return A `conditional_node_result` containing the node ref, body graph, and handle.
//! \throws cuda::std::cuda_error if node creation fails.
_CCCL_HOST_API inline conditional_node_result make_if_node(path_builder& __pb, bool __default_val = true)
{
  conditional_handle __handle{__pb.get_graph(), __default_val};
  return __make_conditional_node(__pb, __handle, ::CU_GRAPH_COND_TYPE_IF);
}

//! \brief Adds an `if`-conditional node reusing an existing conditional handle.
//!
//! \param __pb     Path builder to insert the node into.
//! \param __handle An existing conditional handle (e.g. shared with another node).
//! \return A `conditional_node_result` containing the node ref, body graph, and handle.
//! \throws cuda::std::cuda_error if node creation fails.
_CCCL_HOST_API inline conditional_node_result make_if_node(path_builder& __pb, conditional_handle __handle)
{
  return __make_conditional_node(__pb, __handle, ::CU_GRAPH_COND_TYPE_IF);
}

//! \brief Adds a `while`-conditional node to a CUDA graph path.
//!
//! At runtime, the body graph is executed repeatedly as long as the handle value
//! is non-zero at the start of each iteration (including the first).
//!
//! The caller must populate the returned `body_graph` before instantiating the parent
//! graph. The body is responsible for calling `handle.set_value(false)` or `handle.disable()`
//! to terminate the loop.
//!
//! \param __pb          Path builder to insert the node into.
//! \param __default_val Initial handle value (true = enter loop, false = skip).
//! \return A `conditional_node_result` containing the node ref, body graph, and handle.
//! \throws cuda::std::cuda_error if node creation fails.
_CCCL_HOST_API inline conditional_node_result make_while_node(path_builder& __pb, bool __default_val = true)
{
  conditional_handle __handle{__pb.get_graph(), __default_val};
  return __make_conditional_node(__pb, __handle, ::CU_GRAPH_COND_TYPE_WHILE);
}

//! \brief Adds a `while`-conditional node reusing an existing conditional handle.
//!
//! \param __pb     Path builder to insert the node into.
//! \param __handle An existing conditional handle (e.g. shared with another node).
//! \return A `conditional_node_result` containing the node ref, body graph, and handle.
//! \throws cuda::std::cuda_error if node creation fails.
_CCCL_HOST_API inline conditional_node_result make_while_node(path_builder& __pb, conditional_handle __handle)
{
  return __make_conditional_node(__pb, __handle, ::CU_GRAPH_COND_TYPE_WHILE);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(12, 4)

#endif // _CUDAX__GRAPH_CONDITIONAL_NODE_CUH
