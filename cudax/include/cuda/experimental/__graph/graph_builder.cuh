//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH_BUILDER
#define __CUDAX_GRAPH_GRAPH_BUILDER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/exchange.h>

#include <cuda/experimental/__graph/graph_builder_ref.cuh>

#include <cuda/std/__cccl/prologue.h>

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
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_builder : graph_builder_ref
{
  //! \brief Constructs a new, empty CUDA graph.
  //! \param __dev The device on which graph nodes will execute.
  //! \throws cuda::std::cuda_error if `cudaGraphCreate` fails.
  _CCCL_HOST_API explicit graph_builder(device_ref __dev)
      : graph_builder_ref(nullptr, __dev)
  {
    _CCCL_TRY_CUDA_API(cudaGraphCreate, "cudaGraphCreate failed", &__graph_, 0);
  }

  //! \brief Constructs a new, empty CUDA graph.
  //! \details The nodes in the graph will execute on the default device 0.
  //! \throws cuda::std::cuda_error if `cudaGraphCreate` fails.
  _CCCL_HOST_API explicit graph_builder()
      : graph_builder(device_ref{0})
  {}

  /// Disallow construction from an `int`, e.g., `0`.
  graph_builder(int) = delete;

  /// Disallow construction from `nullptr`.
  graph_builder(::cuda::std::nullptr_t) = delete;

  //! \brief Constructs an uninitialized CUDA graph.
  //! \param __dev The device on which graph nodes will execute, default to device 0.
  //! \throws None
  _CCCL_HOST_API explicit constexpr graph_builder(no_init_t, device_ref __dev = device_ref{0}) noexcept
      : graph_builder_ref(nullptr, __dev)
  {}

  //! \brief Move constructor for `graph_builder`.
  //! \param __other The `graph_builder` object to move from.
  //! \note After the move, the source object is left in the empty state.
  //! \throws None
  //! \post `__other.get() == nullptr`
  _CCCL_HOST_API constexpr graph_builder(graph_builder&& __other) noexcept
      : graph_builder_ref(::cuda::std::exchange(__other.__graph_, nullptr), __other.__dev_)
  {}

  //! \brief Copy constructor for `graph_builder`.
  //! \param __other The `graph_builder` object to copy from.
  //! \throws cuda::std::cuda_error if `cudaGraphClone` fails.
  //! \post `get() == __other.get()`
  _CCCL_HOST_API constexpr graph_builder(graph_builder_ref __other)
      : graph_builder_ref(nullptr, __other.__dev_)
  {
    if (__other.__graph_)
    {
      _CCCL_TRY_CUDA_API(cudaGraphClone, "cudaGraphClone failed", &__graph_, __other.__graph_);
    }
  }

  //! \brief Destructor for `graph_builder`.
  //! \details Ensures proper cleanup of the CUDA graph object.
  //! \throws None
  _CCCL_HOST_API _CCCL_CONSTEXPR_CXX20 ~graph_builder()
  {
    reset();
  }

  //! \brief Move assignment operator for `graph_builder`.
  //! \param __other The `graph_builder` object to move from.
  //! \return A reference to the current object.
  //! \note After the move, the source object is left in the empty state.
  //! \post `__other.get() == nullptr`
  //! \throws None
  _CCCL_HOST_API constexpr auto operator=(graph_builder&& __other) noexcept -> graph_builder&
  {
    if (this != &__other)
    {
      swap(__other);
      __other.reset();
    }
    return *this;
  }

  //! \brief Copy assignment operator for `graph_builder`.
  //! \param __other The `graph_builder` object to copy from.
  //! \return A reference to the current object.
  //! \post `get() == __other.get()`
  //! \throws cuda::std::cuda_error if `cudaGraphClone` fails.
  _CCCL_HOST_API _CCCL_CONSTEXPR_CXX20 auto operator=(graph_builder_ref __other) -> graph_builder&
  {
    if (this != &__other)
    {
      operator=(graph_builder(__other));
    }
    return *this;
  }

  //! \brief Releases ownership of the CUDA graph object.
  //! \return The `cudaGraph_t` handle, leaving this object in a null state.
  //! \throws None
  //! \post `get() == nullptr`
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto release() noexcept -> cudaGraph_t
  {
    return ::cuda::std::exchange(__graph_, nullptr);
  }

  //! \brief Resets the `graph_builder` object, destroying the underlying CUDA graph object.
  //! \throws cuda::std::cuda_error if `cudaGraphDestroy` fails.
  //! \post `get() == nullptr`
  _CCCL_HOST_API constexpr void reset() noexcept
  {
    if (auto __graph = ::cuda::std::exchange(__graph_, nullptr))
    {
      _CCCL_ASSERT_CUDA_API(cudaGraphDestroy, "cudaGraphDestroy failed", __graph);
    }
  }

  //! \brief Constructs a `graph_builder` object from a native CUDA graph handle.
  //! \param __graph The native CUDA graph handle to construct the `graph_builder` object from.
  //! \param __dev The device on which graph nodes will execute, default to device 0.
  //! \throws None
  //! \post `get() == __graph`
  [[nodiscard]] _CCCL_HOST_API static _CCCL_CONSTEXPR_CXX20 auto
  from_native_handle(cudaGraph_t __graph, device_ref __dev) noexcept -> graph_builder
  {
    return graph_builder{__graph, __dev};
  }

private:
  //! \brief Constructs a `graph_builder` object from a native CUDA graph handle.
  //! \param __graph The native CUDA graph handle to construct the `graph_builder` object from.
  //! \param __dev The device on which graph nodes will execute, default to device 0.
  //! \throws None
  _CCCL_HOST_API explicit constexpr graph_builder(cudaGraph_t __graph, device_ref __dev) noexcept
      : graph_builder_ref(__graph, __dev)
  {}
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_GRAPH_BUILDER
