//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH_NODE
#define __CUDAX_GRAPH_GRAPH_NODE

#include <cuda/std/detail/__config>

#include "cuda/std/__cccl/dialect.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/cstddef>

#include <cuda/experimental/__graph/depends_on.cuh>
#include <cuda/experimental/__graph/fwd.cuh>
#include <cuda/experimental/__graph/graph_node_ref.cuh>
#include <cuda/experimental/__graph/graph_node_type.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
//! \brief Represents a node in a CUDA graph, providing RAII-style management of the
//!        underlying CUDA graph node.
//!
//! The `graph_node` class is a wrapper around a CUDA graph node (`cudaGraphNode_t`) and
//! its associated graph (`cudaGraph_t`). It ensures proper resource management by
//! releasing the CUDA graph node when the object is destroyed or reset.
//!
//! This class inherits from `graph_node_ref` and extends its functionality with additional
//! resource management features.
//!
//! ## Key Features:
//! - Default constructor for creating an empty `graph_node`.
//! - Move constructor and move assignment operator for transferring ownership of a CUDA
//!   graph node.
//! - Destructor that automatically releases the CUDA graph node if it is still valid.
//! - `release()` method to release ownership of the CUDA graph node without destroying it.
//! - `reset()` method to explicitly destroy the CUDA graph node.
//! - Static factory method `from_native_handle()` to create a `graph_node` from a native
//!   CUDA graph node handle.
//!
//! ## Usage:
//! This class is designed to be used in CUDA graph-based programming to manage graph nodes
//! safely and efficiently. It ensures that CUDA graph nodes are properly destroyed when
//! they are no longer needed, preventing resource leaks.
//!
//! ## Thread Safety:
//! This class is not thread-safe. Proper synchronization is required if used in a
//! multi-threaded environment.
//!
//! \rst
//! .. _cudax-graph-graph-node:
//! \endrst
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_node : graph_node_ref
{
  _CCCL_HIDE_FROM_ABI graph_node() = default;

  /// Disallow construction from an `int`, e.g., `0`.
  graph_node(int, int = 0) = delete;

  /// Disallow construction from `nullptr`.
  graph_node(_CUDA_VSTD::nullptr_t, _CUDA_VSTD::nullptr_t = nullptr) = delete;

  //! \brief Move constructor for the `graph_node` class.
  //! \param __other The `graph_node` instance to move from. After the move,
  //!                `__other` will be left in the empty state.
  //! \throws None
  //! \post `__other.get() == nullptr`
  _CCCL_HOST_API constexpr graph_node(graph_node&& __other) noexcept
      : graph_node_ref{_CUDA_VSTD::exchange(__other.__node_, {}), __other.__graph_}
  {}

  //! \brief Destructor for the `graph_node` class.
  //!
  //! This destructor calls the `reset()` method to destroy the wrapped `cudaGraphNode_t`,
  //! if it is not null.
  _CCCL_HOST_API _CCCL_CONSTEXPR_CXX20 ~graph_node()
  {
    reset();
  }

  //! \brief Move assignment operator for the `graph_node` class.
  //!
  //! \param __other The `graph_node` instance to move from.
  //! \return A reference to the current `graph_node` instance after assignment.
  //! \note After the move, the source object is left in the empty state.
  //! \post `__other.get() == nullptr`
  _CCCL_HOST_API constexpr auto operator=(graph_node&& __other) noexcept -> graph_node&
  {
    swap(__other);
    __other.reset();
    return *this;
  }

  //! \brief Releases ownership of the current CUDA graph node.
  //!
  //! This function releases the ownership of the CUDA graph node managed by this object
  //! and returns a `graph_node_ref` object. After calling this function, the internal
  //! node pointer is set to `nullptr`, and the caller assumes responsibility for managing
  //! the returned node.
  //!
  //! \return graph_node_ref A non-owning wrapper to the CUDA graph node that was
  //! previously managed by this object. If the internal node pointer was already
  //! `nullptr`, this function returns a null `graph_node_ref`.
  //! \post `get() == nullptr`
  _CCCL_TRIVIAL_HOST_API constexpr auto release() noexcept -> graph_node_ref
  {
    auto __node  = _CUDA_VSTD::exchange(__node_, nullptr);
    auto __graph = _CUDA_VSTD::exchange(__graph_, nullptr);
    return graph_node_ref{__node, __graph};
  }

  //! \brief Resets the graph node by destroying the current CUDA graph node, if it exists.
  //!
  //! This function ensures that the current CUDA graph node is destroyed and the internal
  //! pointer is set to `nullptr`. It uses `cudaGraphDestroyNode` to release the resources
  //! associated with the node. If the destruction fails, an assertion is triggered.
  //! \post `get() == nullptr`
  _CCCL_HOST_API constexpr void reset() noexcept
  {
    __graph_ = nullptr;
    if (auto __old = _CUDA_VSTD::exchange(__node_, nullptr))
    {
      _CCCL_ASSERT_CUDA_API(cudaGraphDestroyNode, "cudaGraphDestroy failed", __old);
    }
  }

  //! \brief Creates a `graph_node` object from a native CUDA graph node handle and graph
  //!        handle.
  //!
  //! \param __node The native CUDA graph node handle (`cudaGraphNode_t`).
  //! \param __graph The native CUDA graph handle (`cudaGraph_t`) to which the node belongs.
  //! \return A `graph_node` object representing the specified CUDA graph node.
  //! \pre The `__node` and `__graph` parameters must both be non-null, or both must be
  //!      null. If one is null and the other is not, an assertion will be triggered.
  //! \post `get() == __node`
  [[nodiscard]] _CCCL_HOST_API static _CCCL_CONSTEXPR_CXX20 auto
  from_native_handle(cudaGraphNode_t __node, cudaGraph_t __graph) noexcept -> graph_node
  {
    return graph_node{__node, __graph};
  }

private:
  friend struct graph;

  //! \brief Constructs a `graph_node` object from a native CUDA graph node handle and graph
  //!        handle.
  //! \param __node The native CUDA graph node handle (`cudaGraphNode_t`).
  //! \param __graph The native CUDA graph handle (`cudaGraph_t`) to which the node belongs.
  //! \throws None
  //! \post `get() == __node`
  _CCCL_HOST_API explicit constexpr graph_node(cudaGraphNode_t __node, cudaGraph_t __graph) noexcept
      : graph_node_ref{__node, __graph}
  {}
};

} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_GRAPH_GRAPH_NODE
