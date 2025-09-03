//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH_BUILDER_REF
#define __CUDAX_GRAPH_GRAPH_BUILDER_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cuda/experimental/__graph/graph.cuh>
#include <cuda/experimental/__graph/graph_node_ref.cuh>

#include <cuda_runtime_api.h>

#include <cuda/std/__cccl/prologue.h>

// work around breathe "_CUDAX_CONSTEXPR_FRIEND friend" bug.
// See: https://github.com/breathe-doc/breathe/issues/916
#if defined(_CCCL_DOXYGEN_INVOKED)
#  define _CUDAX_CONSTEXPR_FRIEND friend
#else
#  define _CUDAX_CONSTEXPR_FRIEND constexpr friend
#endif

namespace cuda::experimental
{
//! \brief An owning wrapper type for a cudaGraph_t handle
//!
//! The `graph_builder` class provides a high-level interface for creating, managing, and
//! manipulating CUDA graphs. It ensures proper resource management and simplifies the
//! process of working with CUDA graph APIs.
//!
//! Features:
//! - Supports construction, destruction, and copying of CUDA graphs.
//! - Provides methods for adding nodes and dependencies to the graph.
//! - Allows instantiation of the graph into an executable form.
//! - Ensures proper cleanup of CUDA resources.
//!
//! Usage:
//! - Create an instance of `graph_builder` to represent a CUDA graph.
//! - Use the `add` methods to add nodes and dependencies to the graph.
//! - Instantiate the graph using the `instantiate` method to obtain an executable graph.
//! - Use the `reset` method to release resources when the graph is no longer needed.
//!
//! Thread Safety:
//! - This class is not thread-safe. Concurrent access to the same `graph_builer` object
//!   must be synchronized externally.
//!
//! Exception Safety:
//! - Methods that interact with CUDA APIs may throw ``cuda::std::cuda_error`` if the
//!   underlying CUDA operation fails.
//! - Move operations leave the source object in a valid but unspecified state.
//!
//! \rst
//! .. _cudax-graph-graph-builder:
//! \endrst
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_builder_ref
{
  //! \brief Constructs a new, empty CUDA graph.
  //! \param __dev The device on which graph nodes will execute.
  //! \throws cuda::std::cuda_error if `cudaGraphCreate` fails.
  _CCCL_HOST_API constexpr graph_builder_ref(cudaGraph_t __graph, device_ref __dev) noexcept
      : __dev_{__dev}
      , __graph_{__graph}
  {}

  //! \brief Compares two `graph_builder` objects for equality.
  //!
  //! \param __lhs The left-hand side `graph_builder` object to compare.
  //! \param __rhs The right-hand side `graph_builder` object to compare.
  //! \return `true` if both `graph_builder` objects are equal, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator==(const graph_builder_ref& __lhs, const graph_builder_ref& __rhs) noexcept
  {
    return __lhs.__graph_ == __rhs.__graph_;
  }

  //! \brief Compares two `graph_builder` objects for inequality.
  //!
  //! \param __lhs The left-hand side `graph_builder` object to compare.
  //! \param __rhs The right-hand side `graph_builder` object to compare.
  //! \return `true` if both `graph_builder` objects are not equal, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator!=(const graph_builder_ref& __lhs, const graph_builder_ref& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  //! \brief Test whether a `graph_builder` object is null.
  //! \return `true` if `__rhs` is null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator==(::cuda::std::nullptr_t, const graph_builder_ref& __rhs) noexcept
  {
    return !static_cast<bool>(__rhs);
  }

  //! \brief Test whether a `graph_builder` object is null.
  //! \return `true` if `__rhs` is null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator==(const graph_builder_ref& __lhs, ::cuda::std::nullptr_t) noexcept
  {
    return !static_cast<bool>(__lhs);
  }

  //! \brief Test whether a `graph_builder` object is not null.
  //! \return `true` if `__rhs` is not null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator!=(::cuda::std::nullptr_t, const graph_builder_ref& __rhs) noexcept
  {
    return static_cast<bool>(__rhs);
  }

  //! \brief Test whether a `graph_builder` object is not null.
  //! \return `true` if `__lhs` is not null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator!=(const graph_builder_ref& __lhs, ::cuda::std::nullptr_t) noexcept
  {
    return static_cast<bool>(__lhs);
  }

  //! \brief Checks if the graph handle is valid.
  //!
  //! \details This operator allows the graph builder to be used in a
  //! boolean context to determine if it is valid. A valid graph builder
  //! is one where the internal node pointer is not `nullptr`.
  //!
  //! \return `true` if the internal node pointer is not `nullptr`, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_API explicit constexpr operator bool() const noexcept
  {
    return __graph_ != nullptr;
  }

  //! \brief Checks if the graph is not null.
  //! \return `true` if the internal graph handle is null, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_API constexpr auto operator!() const noexcept -> bool
  {
    return !static_cast<bool>(*this);
  }

  //! \brief Swaps the contents of this `graph_builder` with another.
  //! \param __other The `graph_builder` object to swap with.
  //! \throws None
  _CCCL_HOST_API constexpr void swap(graph_builder_ref& __other) noexcept
  {
    ::cuda::std::swap(__graph_, __other.__graph_);
  }

  //! \brief Retrieves the underlying CUDA graph object.
  //! \return The `cudaGraph_t` handle.
  //! \throws None
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get() const noexcept -> cudaGraph_t
  {
    return __graph_;
  }

  //! \brief Retrieves the device on which the graph is built.
  //! \return The device on which the graph is built.
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get_device() const noexcept -> device_ref
  {
    return __dev_;
  }

  //! \brief Adds a new root node to the graph.
  //! \tparam _Node The type of the node to add.
  //! \param __node The descriptor of the node to add to the graph.
  //! \return A `graph_node_ref` representing the added node. The graph object owns the
  //! new node.
  //! \throws cuda::std::cuda_error if adding the node fails.
  template <class _Node>
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto add(_Node __node) -> graph_node_ref
  {
    return add(_CCCL_MOVE(__node), ::cuda::std::span<cudaGraphNode_t, 0>{});
  }

  //! \brief Adds a new node to the graph with specified dependencies.
  //!
  //! This function creates a new node in the graph and establishes dependencies
  //! between the newly created node and the provided dependency nodes.
  //!
  //! \tparam _Node The type of the node to be added.
  //! \tparam _Extent The extent of the span representing the dependencies.
  //!
  //! \param __node The descriptor of the node to be added to the graph.
  //! \param __deps An array of `cudaGraphNode_t` handles representing the dependencies of
  //! the new node. Each node in this span will become a dependency of the newly created
  //! node.
  //!
  //! \return A `graph_node_ref` object representing the newly created node in the graph.
  //! The graph object owns the new node.
  //!
  //! \throws cuda::std::cuda_error If the CUDA API call `cudaGraphAddDependencies` fails.
  //!
  //! \details
  //! - The function first creates a new node in the graph using the provided `_Node` object.
  //! - It initializes an array of "dependant" nodes, where all dependant nodes correspond
  //!   to the newly created node.
  //! - The function then uses the CUDA API `cudaGraphAddDependencies` to establish the
  //!   dependencies between the newly created node and the nodes provided in the `__deps`
  //!   span.
  //! - If the number of dependencies is small, a stack-allocated buffer is used;
  //!   otherwise, a dynamically allocated array is used to store the dependant nodes.
  template <class _Node, size_t _Np>
  _CCCL_HOST_API constexpr auto add(_Node __node, ::cuda::std::array<cudaGraphNode_t, _Np> __deps) -> graph_node_ref
  {
    return add(_CCCL_MOVE(__node), ::cuda::std::span{__deps});
  }

  //! \overload
  template <class _Node, size_t _Extent>
  _CCCL_HOST_API constexpr auto add(_Node __node, ::cuda::std::span<cudaGraphNode_t, _Extent> __deps) -> graph_node_ref
  {
    // assert that the node descriptor returns a graph_node_ref object:
    static_assert(::cuda::std::_IsSame<decltype(__node.__add_to_graph(__graph_, __deps)), graph_node_ref>::value,
                  "node descriptors must return a graph_node_ref");
    return __node.__add_to_graph(__graph_, __deps);
  }

  //! \brief Retrieves the number of nodes in the graph.
  //! \return The number of nodes in the graph.
  //! \throws cuda::std::cuda_error if `cudaGraphGetNodes` fails.
  [[nodiscard]] _CCCL_HOST_API size_t node_count() const
  {
    size_t __count = 0;
    _CCCL_TRY_CUDA_API(cudaGraphGetNodes, "cudaGraphGetNodes failed", __graph_, nullptr, &__count);
    return __count;
  }

  //! \brief Instantiates the CUDA graph into a `graph_exec` object.
  //! \return A `graph_exec` object representing the instantiated graph.
  //! \throws cuda::std::cuda_error if `cudaGraphInstantiate` fails.
  _CCCL_HOST_API auto instantiate() -> graph
  {
    _CCCL_ASSERT(__graph_ != nullptr, "cannot instantiate a NULL graph");
    graph __exec;
    _CCCL_TRY_CUDA_API(
      cudaGraphInstantiate,
      "cudaGraphInstantiate failed",
      &__exec.__exec_, // output
      __graph_, // graph to instantiate
      0); // flags
    return __exec;
  }

private:
  friend struct graph_builder;

  //! \brief Adds this graph as a child graph to the parent graph.
  //! \param __parent The parent graph to which this graph will be added.
  //! \return A `graph_node_ref` representing the added child graph.
  //! \throws cuda::std::cuda_error if `cudaGraphAddChildGraphNode` fails.
  template <size_t _Extent>
  [[nodiscard]] _CCCL_HOST_API auto
  __add_to_graph(cudaGraph_t __parent, ::cuda::std::span<cudaGraphNode_t, _Extent> __deps) -> graph_node_ref
  {
    graph_node_ref __child;
    __child.__graph_ = __graph_;
    _CCCL_ASSERT_CUDA_API(
      cudaGraphAddChildGraphNode,
      "cudaGraphAddChildGraphNode failed",
      &__child.__node_, // output
      __parent, // graph to which we are adding the child graph
      __deps.data(), // dependencies
      __deps.size(), // number of dependencies
      __graph_); // the child graph to add
    return __child;
  }

  device_ref __dev_; //!< The device on which the graph is built.
  cudaGraph_t __graph_ = nullptr; //!< The underlying CUDA graph handle.
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_GRAPH_BUILDER_REF
