//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH
#define __CUDAX_GRAPH_GRAPH

#include <cuda/std/detail/__config>

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

#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda_runtime_api.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! \brief An owning wrapper for a cudaGraphExec_t handle
//!
//! The `graph` class provides a safe and convenient interface for managing
//! the lifecycle of a `cudaGraphExec_t` object, ensuring proper cleanup and
//! resource management. It supports move semantics, resource release, and
//! launch of the CUDA graph.
//!
//! \note The `graph` object is not directly constructible. One is obtained
//!       by calling the `instantiate()` method on a `graph_builder` object.
//! \sa cuda::experimental::graph_builder
//!
//! \rst
//! .. _cudax-graph-graph:
//! \endrst
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph
{
  //! \brief Move constructor for `graph`.
  //! \param __other The `graph` object to move from.
  //! \note After the move, the source object is left in the empty state.
  //! \post `__other.get() == nullptr`
  _CCCL_HOST_API constexpr graph(graph&& __other) noexcept
      : __exec_{::cuda::std::exchange(__other.__exec_, nullptr)}
  {}

  //! \brief Destructor for `graph`.
  //! \details Ensures proper cleanup of the CUDA graph execution object.
  //! \throws None
  _CCCL_HOST_API _CCCL_CONSTEXPR_CXX20 ~graph()
  {
    reset();
  }

  //! \brief Move assignment operator for `graph`.
  //! \param __other The `graph` object to move from.
  //! \return A reference to the current object.
  //! \note After the move, the source object is left in the empty state.
  //! \throws None
  //! \post `__other.get() == nullptr`
  _CCCL_HOST_API constexpr auto operator=(graph&& __other) noexcept -> graph&
  {
    swap(__other);
    __other.reset();
    return *this;
  }

  //! \brief Swaps the contents of this `graph` with another.
  //! \param __other The `graph` object to swap with.
  //! \throws None
  _CCCL_HOST_API constexpr void swap(graph& __other) noexcept
  {
    ::cuda::std::swap(__exec_, __other.__exec_);
  }

  //! \brief Retrieves the underlying CUDA graph execution object.
  //! \return The `cudaGraphExec_t` handle.
  //! \throws None
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get() const noexcept -> cudaGraphExec_t
  {
    return __exec_;
  }

  //! \brief Releases ownership of the CUDA graph execution object.
  //! \return The `cudaGraphExec_t` handle, leaving this object in a null state.
  //! \throws None
  //! \post `get() == nullptr`
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto release() noexcept -> cudaGraphExec_t
  {
    return ::cuda::std::exchange(__exec_, nullptr);
  }

  //! \brief Resets the `graph` object, destroying the underlying CUDA graph execution object.
  //! \throws cuda::std::cuda_error if `cudaGraphExecDestroy` fails.
  //! \post `get() == nullptr`
  _CCCL_HOST_API constexpr void reset() noexcept
  {
    if (auto __exec = ::cuda::std::exchange(__exec_, nullptr))
    {
      _CCCL_ASSERT_CUDA_API(cudaGraphExecDestroy, "cudaGraphDestroy failed", __exec);
    }
  }

  //! \brief Constructs a `graph` object from a native CUDA graph execution handle.
  //! \param __exec The native CUDA graph execution handle to construct the `graph` object from.
  //! \throws None
  //! \post `get() == __exec`
  [[nodiscard]] _CCCL_HOST_API static _CCCL_CONSTEXPR_CXX20 auto from_native_handle(cudaGraphExec_t __exec) noexcept
    -> graph
  {
    return graph{__exec};
  }

  //! \brief Launches the CUDA graph execution object on the specified stream.
  //! \param __stream The stream on which to launch the graph.
  //! \throws cuda::std::cuda_error if `cudaGraphLaunch` fails.
  _CCCL_HOST_API void launch(stream_ref __stream)
  {
    _CCCL_TRY_CUDA_API(cudaGraphLaunch, "cudaGraphLaunch failed", __exec_, __stream.get());
  }

private:
  friend struct graph_builder_ref;

  _CCCL_HIDE_FROM_ABI graph() = default;

  //! \brief Constructs a `graph` object from a native CUDA graph execution handle.
  //! \param __exec The native CUDA graph execution handle to construct the `graph` object from.
  //! \throws None
  _CCCL_HOST_API explicit constexpr graph(cudaGraphExec_t __exec) noexcept
      : __exec_{__exec}
  {}

  cudaGraphExec_t __exec_ = nullptr; //!< The underlying CUDA graph execution handle.
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_GRAPH
