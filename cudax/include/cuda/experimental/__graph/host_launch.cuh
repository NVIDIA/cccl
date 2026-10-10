//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_HOST_LAUNCH_CUH
#define _CUDAX__GRAPH_HOST_LAUNCH_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(12, 2)

#  include <cuda/__launch/host_launch.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__functional/reference_wrapper.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__type_traits/is_function.h>
#  include <cuda/std/__type_traits/is_move_constructible.h>
#  include <cuda/std/__type_traits/is_pointer.h>
#  include <cuda/std/__type_traits/remove_pointer.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/experimental/__driver/driver_api.cuh>
#  include <cuda/experimental/__graph/graph_node_ref.cuh>
#  include <cuda/experimental/__graph/path_builder.cuh>

#  include <memory>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// Launcher for a plain no-arg function pointer.
template <class _FuncPtr>
_CCCL_HOST_API inline void CUDA_CB __graph_func_ptr_launcher(void* __callable_ptr)
{
  reinterpret_cast<_FuncPtr>(__callable_ptr)();
}

// Launcher for a heap-allocated callable + argument pack. The graph host node callback
// signature is void(void*), unlike the stream callback which is void(CUstream, CUresult, void*),
// so we can't reuse ::cuda::__stream_callback_launcher here.
template <class _CallbackData>
_CCCL_HOST_API inline void CUDA_CB __graph_callback_launcher(void* __data_ptr)
{
  auto* __data = static_cast<_CallbackData*>(__data_ptr);
  // Copy, not move — the graph may be launched multiple times.
  // Data lifetime is managed by a graph user object.
  ::cuda::std::apply(__data->__callable_, __data->__args_);
}

template <class _CallbackData>
_CCCL_HOST_API inline void CUDA_CB __graph_callback_data_destroyer(void* __data_ptr)
{
  delete static_cast<_CallbackData*>(__data_ptr);
}

//! \brief Adds a host node to a CUDA graph path that invokes a callable on the host.
//!
//! The callable and its arguments are copied into a heap allocation whose lifetime is
//! tied to the graph via a CUDA user object. The graph can be launched multiple times.
//! The rules and restrictions match `cuda::host_launch`:
//!   - The callable must not call into CUDA Runtime or Driver APIs.
//!   - It must not depend on another thread that could block on asynchronous CUDA work.
//!
//! Three dispatch paths (mirroring `cuda::host_launch`):
//!   1. A bare no-arg function pointer; no allocation.
//!   2. A `std::reference_wrapper` (no args) passes the address of the referenced object.
//!   3. Everything else is heap-allocated with lifetime managed by a graph user object.
//!
//! \param __pb       Path builder to insert the node into.
//! \param __callable Callable to execute on the host.
//! \param __args     Arguments to forward to the callable.
//! \return A `graph_node_ref` for the newly added host node.
//! \throws cuda::std::cuda_error if node creation fails.
template <class _Callable, class... _Args>
_CCCL_HOST_API graph_node_ref host_launch(path_builder& __pb, _Callable __callable, _Args... __args)
{
  static_assert(::cuda::std::is_invocable_v<_Callable, _Args...>,
                "Callable can't be called with the supplied arguments");
  static_assert(::cuda::std::is_move_constructible_v<_Callable>, "The callable must be move constructible");
  static_assert((::cuda::std::is_move_constructible_v<_Args> && ...),
                "All callback arguments must be move constructible");

  constexpr bool __has_args = sizeof...(_Args) > 0;

  ::CUhostFn __fn   = nullptr;
  void* __user_data = nullptr;

  if constexpr (!__has_args && ::cuda::std::is_pointer_v<_Callable>
                && ::cuda::std::is_function_v<::cuda::std::remove_pointer_t<_Callable>>)
  {
    __fn        = __graph_func_ptr_launcher<_Callable>;
    __user_data = reinterpret_cast<void*>(__callable);
  }
  else if constexpr (!__has_args && ::cuda::std::__is_cuda_std_reference_wrapper_v<_Callable>)
  {
    __fn        = ::cuda::__host_func_launcher<typename _Callable::type>;
    __user_data = static_cast<void*>(::cuda::std::addressof(__callable.get()));
  }
  else
  {
    // Heap-allocate the callback data. Lifetime is tied to the graph via a user object.
    using _CallbackData = ::cuda::__stream_callback_data<_Callable, _Args...>;
    auto __data         = new _CallbackData{::cuda::std::move(__callable), {::cuda::std::move(__args)...}};
    __fn                = __graph_callback_launcher<_CallbackData>;
    __user_data         = __data;
    ::cuda::experimental::__driver::__graphRetainUserObject(
      __pb.get_native_graph_handle(), __data, __graph_callback_data_destroyer<_CallbackData>);
  }

  auto __deps = __pb.get_dependencies();
  ::CUgraphNodeParams __params{};
  __params.type          = ::CU_GRAPH_NODE_TYPE_HOST;
  __params.host.fn       = __fn;
  __params.host.userData = __user_data;
  auto __node            = ::cuda::experimental::__driver::__graphAddNode(
    __pb.get_native_graph_handle(), __deps.data(), __deps.size(), &__params);

  __pb.__clear_and_set_dependency_node(__node);
  return graph_node_ref{__node, __pb.get_native_graph_handle()};
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(12, 2)

#endif // _CUDAX__GRAPH_HOST_LAUNCH_CUH
