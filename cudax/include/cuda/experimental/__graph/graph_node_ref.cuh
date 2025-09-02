//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH_NODE_REF
#define __CUDAX_GRAPH_GRAPH_NODE_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cuda/experimental/__graph/fwd.cuh>
#include <cuda/experimental/__graph/graph_node_type.cuh>

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
//! \brief A reference wrapper for a CUDA graph node.
//!
//! This structure provides an interface to manage and interact with a CUDA graph node
//! within a CUDA graph. It includes functionality for swapping, retrieving node information,
//! and managing dependencies between nodes.
//!
//! \rst
//! .. _cudax-graph-graph-node-ref:
//! \endrst
struct graph_node_ref
{
  //! \brief Default constructor.
  _CCCL_HIDE_FROM_ABI graph_node_ref() = default;

  /// Disallow construction from an `int`, e.g., `0`.
  graph_node_ref(int, int = 0) = delete;

  /// Disallow construction from `nullptr`.
  graph_node_ref(::cuda::std::nullptr_t, ::cuda::std::nullptr_t = nullptr) = delete;

  //! \brief Constructs a graph_node_ref with a given CUDA graph node and graph.
  //! \param __node The CUDA graph node.
  //! \param __graph The CUDA graph containing the node.
  //! \pre Both of __node and __graph are non-null.
  //! \post `get() == __node`
  _CCCL_NODEBUG_HOST_API explicit constexpr graph_node_ref(cudaGraphNode_t __node, cudaGraph_t __graph) noexcept
      : __node_{__node}
      , __graph_{__graph}
  {
    _CCCL_ASSERT(__node_ && __graph_, "construction of a graph_node_ref from a null cudaGraphNode_t handle");
  }

  //! \brief Compares two `graph_node_ref` objects for equality.
  //!
  //! \param __lhs The left-hand side `graph_node_ref` object to compare.
  //! \param __rhs The right-hand side `graph_node_ref` object to compare.
  //! \return `true` if both `graph_node_ref` objects are equal, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator==(const graph_node_ref& __lhs, const graph_node_ref& __rhs) noexcept
  {
    return __lhs.__node_ == __rhs.__node_ && __lhs.__graph_ == __rhs.__graph_;
  }

  //! \brief Compares two `graph_node_ref` objects for inequality.
  //!
  //! \param __lhs The left-hand side `graph_node_ref` object to compare.
  //! \param __rhs The right-hand side `graph_node_ref` object to compare.
  //! \return `true` if both `graph_node_ref` objects are not equal, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator!=(const graph_node_ref& __lhs, const graph_node_ref& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  //! \brief Test whether a `graph_node_ref` object is null.
  //! \return `true` if `__rhs` is null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator==(::cuda::std::nullptr_t, const graph_node_ref& __rhs) noexcept
  {
    return !static_cast<bool>(__rhs);
  }

  //! \brief Test whether a `graph_node_ref` object is null.
  //! \return `true` if `__rhs` is null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator==(const graph_node_ref& __lhs, ::cuda::std::nullptr_t) noexcept
  {
    return !static_cast<bool>(__lhs);
  }

  //! \brief Test whether a `graph_node_ref` object is not null.
  //! \return `true` if `__rhs` is not null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator!=(::cuda::std::nullptr_t, const graph_node_ref& __rhs) noexcept
  {
    return static_cast<bool>(__rhs);
  }

  //! \brief Test whether a `graph_node_ref` object is not null.
  //! \return `true` if `__lhs` is not null, `false` otherwise.
  [[nodiscard]] _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND bool
  operator!=(const graph_node_ref& __lhs, ::cuda::std::nullptr_t) noexcept
  {
    return static_cast<bool>(__lhs);
  }

  //! \brief Checks if the graph node reference is valid.
  //!
  //! \details This operator allows the graph node reference to be used in a
  //! boolean context to determine if it is valid. A valid graph node reference
  //! is one where the internal node pointer is not null.
  //!
  //! \return `true` if the internal node pointer is not null, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_API explicit constexpr operator bool() const noexcept
  {
    return __node_ != nullptr;
  }

  //! \brief Checks if the graph node reference is not null.
  //! \return `true` if the internal node pointer is null, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_API constexpr auto operator!() const noexcept -> bool
  {
    return !static_cast<bool>(*this);
  }

  //! \brief Swaps the contents of this graph_node_ref with another.
  //! \param __other The other graph_node_ref to swap with.
  _CCCL_HOST_API constexpr void swap(graph_node_ref& __other) noexcept
  {
    ::cuda::std::swap(__node_, __other.__node_);
    ::cuda::std::swap(__graph_, __other.__graph_);
  }

  //! \brief Swaps the contents of two graph_node_ref objects.
  //! \param __left The first graph_node_ref.
  //! \param __right The second graph_node_ref.
  _CCCL_HOST_API _CUDAX_CONSTEXPR_FRIEND void swap(graph_node_ref& __left, graph_node_ref& __right) noexcept
  {
    __left.swap(__right);
  }

  //! \brief Retrieves the underlying CUDA graph node.
  //! \return The CUDA graph node.
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get() const noexcept -> cudaGraphNode_t
  {
    return __node_;
  }

  //! \brief Retrieves the CUDA graph this node belongs to.
  //! \return The CUDA graph.
  // internal for now because of a clash with get_graph() in path_builder. We could store the device in the
  // graph_node_ref, but that feels like going a bit too far.
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get_native_graph_handle() const noexcept -> cudaGraph_t
  {
    return __graph_;
  }

  //! \brief Retrieves the type of the CUDA graph node.
  //! \return The type of the graph node as a graph_node_type.
  //! \pre The internal graph node handle is not null.
  //! \throws If the CUDA API call to retrieve the node type fails.
  [[nodiscard]] _CCCL_HOST_API auto type() const -> graph_node_type
  {
    _CCCL_ASSERT(__node_ != nullptr, "cannot get the type of a null graph node");
    cudaGraphNodeType __type;
    _CCCL_ASSERT_CUDA_API(cudaGraphNodeGetType, "cudaGraphNodeGetType failed", __node_, &__type);
    return static_cast<graph_node_type>(__type);
  }

  //! \brief Establishes dependencies between this node and other nodes.
  //! This function sets up dependencies such that this node depends on the provided nodes.
  //!
  //! \tparam _Nodes Variadic template parameter for the types of the dependent nodes.
  //! \param __nodes The nodes that this node depends on.
  //! \pre The internal graph node handle is not null.
  //! \throws If the CUDA API call to add dependencies fails.
  template <class... _Nodes>
  _CCCL_HOST_API constexpr void depends_on(const _Nodes&... __nodes)
  {
    cudaGraphNode_t __deps[]{__nodes.get()...};
    return depends_on(::cuda::std::span{__deps});
  }

  //! \brief Establishes dependencies between this node and other nodes.
  //! This function sets up dependencies such that this node depends on the provided nodes.
  //!
  //! \tparam _Node The type of the node to be added.
  //! \tparam _Extent The extent of the span representing the dependencies.
  //!
  //! \param __deps A span of `cudaGraphNode_t` representing the dependencies of this node.
  //!               Each node in the span will become a dependency of this node.
  //!
  //! \throws cuda::std::cuda_error If the CUDA API call `cudaGraphAddDependencies` fails.
  //!
  //! \details
  //! - This function first initializes an array of "dependant" nodes, where all dependant
  //!   nodes correspond to this node.
  //! - The function then uses the CUDA API `cudaGraphAddDependencies` to establish the
  //!   dependencies between this node and the nodes provided in the `__deps` span.
  //! - If the number of dependencies is small, a stack-allocated buffer is used; otherwise,
  //!   a dynamically allocated array is used to store the dependant nodes.
  template <size_t _Extent>
  _CCCL_HOST_API _CCCL_CONSTEXPR_CXX23 void depends_on(::cuda::std::span<cudaGraphNode_t, _Extent> __deps)
  {
    _CCCL_ASSERT(__node_ != nullptr, "cannot add dependencies to a null graph node");
    if (!__deps.empty())
    {
      // Initialize an array of "dependant" nodes that correspond to the dependencies. All
      // dependant nodes are __node_; thus, each node in __deps becomes a dependency of the
      // newly created node.
      using __src_arr_t = ::cuda::std::unique_ptr<cudaGraphNode_t[], void (*)(cudaGraphNode_t*) noexcept>;
      cudaGraphNode_t __small_buffer[_Extent == ::cuda::std::dynamic_extent ? 4 : _Extent];
      bool const __is_small = __deps.size() <= ::cuda::std::ranges::size(__small_buffer);
      auto const __src_arr  = __is_small ? __src_arr_t{__small_buffer, &__noop_deleter}
                                         : __src_arr_t{::new cudaGraphNode_t[__deps.size()], &__array_deleter};
      ::cuda::std::fill(__src_arr.get(), __src_arr.get() + __deps.size(), __node_);

      // Add the dependencies using __src_arr array and the span of dependencies.
#if _CCCL_CTK_AT_LEAST(13, 0)
      _CCCL_TRY_CUDA_API(
        cudaGraphAddDependencies,
        "cudaGraphAddDependencies failed",
        __graph_,
        __deps.data(), // dependencies
        __src_arr.get(), // dependant nodes
        __nullptr, // no edge data
        __deps.size()); // number of dependencies
#else
      _CCCL_TRY_CUDA_API(
        cudaGraphAddDependencies,
        "cudaGraphAddDependencies failed",
        __graph_,
        __deps.data(), // dependencies
        __src_arr.get(), // dependant nodes
        __deps.size()); // number of dependencies
#endif
    }
  }

private:
  friend struct graph_builder_ref;

  template <class... _Nodes>
  friend _CCCL_NODEBUG_HOST_API constexpr auto depends_on(const _Nodes&...) noexcept
    -> ::cuda::std::array<cudaGraphNode_t, sizeof...(_Nodes)>;

  _CCCL_NODEBUG_HOST_API explicit constexpr graph_node_ref(cudaGraphNode_t __node) noexcept
      : __node_{__node}
  {}

  _CCCL_HOST_API static constexpr void __noop_deleter(cudaGraphNode_t*) noexcept {}
  _CCCL_HOST_API static _CCCL_CONSTEXPR_CXX20_ALLOCATION void __array_deleter(cudaGraphNode_t* __ptr) noexcept
  {
    delete[] __ptr;
  }

  cudaGraphNode_t __node_ = nullptr; ///< The CUDA graph node.
  cudaGraph_t __graph_    = nullptr; ///< The CUDA graph containing the node.
};
} // namespace cuda::experimental

#undef _CUDAX_CONSTEXPR_FRIEND

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_GRAPH_NODE_REF
