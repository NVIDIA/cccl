//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_PATH_BUILDER_CUH
#define _CUDAX__GRAPH_PATH_BUILDER_CUH

#include <cuda/std/detail/__config>

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__exception/cuda_error.h>

#include <cuda/experimental/__graph/concepts.cuh>
#include <cuda/experimental/__graph/graph_builder.cuh>
#include <cuda/experimental/__graph/graph_node_ref.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <vector>

#include <cuda_runtime.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental
{

//! \brief A builder for a path in a CUDA graph.
//!
//! This class allows for the creation of a path in a CUDA graph, which is a sequence of nodes that are executed in
//! order. The path builder can be used to add nodes to the path, and to set the dependencies between nodes. Thanks to
//! the sequential nature of the path builder, it is possible to write single code path that uses either a stream or a
//! path builder to result in either eager stream execution or construction of a lazy graph.
//!
//! \rst
//! .. _cudax-graph-path-builder:
//! \endrst
struct path_builder
{
  //! \brief Construct a path builder that will insert nodes into a graph builder.
  //! \param __builder The graph builder to create the path builder for.
  _CCCL_HOST_API explicit path_builder(graph_builder_ref __builder)
      : __dev_{__builder.get_device()}
      , __graph_{__builder.get()}
  {}

  //! \brief Construct a path builder that will insert nodes into a graph.
  //! \param __dev The device on which nodes inserted into the graph will execute.
  //! \param __graph The graph to create the path builder for.
  path_builder(device_ref __dev, cudaGraph_t __graph)
      : __dev_{__dev}
      , __graph_{__graph}
  {}

#if _CCCL_CTK_AT_LEAST(12, 3)
  //! \brief Capture the nodes into the path builder from a legacy stream capture.
  //! \param __stream The stream to use for the capture.
  //! \param __capture_fn A function that will be called with the stream to capture the nodes to.
  template <typename _Fn>
  _CCCL_HOST_API void legacy_stream_capture(stream_ref __stream, _Fn&& __capture_fn)
  {
    _CCCL_TRY_CUDA_API(
      ::cudaStreamBeginCaptureToGraph,
      "Failed to begin stream capture",
      __stream.get(),
      __graph_,
      __nodes_.data(),
      nullptr,
      __nodes_.size(),
      cudaStreamCaptureModeGlobal);

    __capture_fn(__stream.get());

    cudaGraph_t __graph_out = nullptr;

    cudaStreamCaptureStatus __capture_status;
    const cudaGraphNode_t* __last_captured_node = nullptr;
    size_t __num_nodes                          = 0;

#  if _CCCL_CTK_AT_LEAST(13, 0)
    _CCCL_TRY_CUDA_API(
      ::cudaStreamGetCaptureInfo,
      "Failed to get stream capture info",
      __stream.get(),
      &__capture_status,
      nullptr,
      nullptr,
      &__last_captured_node,
      nullptr,
      &__num_nodes);
#  else // _CCCL_CTK_AT_LEAST(13, 0)
    _CCCL_TRY_CUDA_API(
      ::cudaStreamGetCaptureInfo,
      "Failed to get stream capture info",
      __stream.get(),
      &__capture_status,
      nullptr,
      nullptr,
      &__last_captured_node,
      &__num_nodes);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

    if (__capture_status != cudaStreamCaptureStatusActive)
    {
      __throw_cuda_error(cudaErrorInvalidValue, "Stream capture no longer active", "cudaStreamGetCaptureInfo");
    }
    _CCCL_TRY_CUDA_API(::cudaStreamEndCapture, "Failed to end stream capture", __stream.get(), &__graph_out);
    assert(__graph_out == __graph_);
    assert(__num_nodes == 1);
    __nodes_.clear();
    __nodes_.push_back(__last_captured_node[0]);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 3)

  //! \brief Clear the path builder and set the dependency node.
  //! Used by most APIs that operate on a path builder to insert a new node into the path.
  //! \param __node The node to set as the dependency node.
  _CCCL_HOST_API void __clear_and_set_dependency_node(cudaGraphNode_t __node)
  {
    __nodes_.clear(); // Clear existing nodes
    __nodes_.push_back(__node);
  }

  //! \brief Get the dependencies of the path builder.
  //! \return A span of the dependencies of the path builder.
  [[nodiscard]] _CCCL_NODEBUG_HOST_API auto get_dependencies() const noexcept
    -> ::cuda::std::span<const cudaGraphNode_t>
  {
    return ::cuda::std::span(__nodes_.data(), __nodes_.size());
  }

  //! \brief Add the dependencies of another path builder to this path builder.
  //! \param __other The path builder to add dependencies from.
  //! Named wait to match the stream/stream_ref wait function
  _CCCL_HOST_API void wait(const path_builder& __other)
  {
    __nodes_.insert(__nodes_.end(), __other.__nodes_.begin(), __other.__nodes_.end());
  }

  template <typename... Nodes>
  static constexpr bool __all_dependencies = (graph_dependency<Nodes> && ...);

  //! \brief Add the dependencies of another path builder or single nodes to this path builder.
  //! \param __nodes The nodes or path builders to add to the path builder as dependencies.
  _CCCL_TEMPLATE(typename... Nodes)
  _CCCL_REQUIRES(__all_dependencies<Nodes...>)
  _CCCL_HOST_API void depends_on(Nodes&&... __nodes)
  {
    (
      [this](auto&& __arg) {
        if constexpr (::cuda::std::is_same_v<::cuda::std::decay_t<decltype(__arg)>, graph_node_ref>)
        {
          __nodes_.push_back(__arg.get());
        }
        else
        {
          __nodes_.insert(__nodes_.end(), __arg.__nodes_.begin(), __arg.__nodes_.end());
        }
      }(static_cast<Nodes&&>(__nodes)),
      ...);
  }

  //! \brief Get the graph that the path builder is building.
  //! \return The graph that the path builder is building.
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get_graph() const noexcept -> graph_builder_ref
  {
    return graph_builder_ref(__graph_, __dev_);
  }

  //! \internal
  //! Internal graph handle getter to match graph_node_ref::__get_graph().
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get_native_graph_handle() const noexcept -> cudaGraph_t
  {
    return __graph_;
  }

  //! \brief Retrieves the device on which graph nodes inserted by the path builder will execute.
  //! \return The device on which graph nodes inserted by the path builder will execute.
  [[nodiscard]] _CCCL_NODEBUG_HOST_API constexpr auto get_device() const noexcept -> device_ref
  {
    return __dev_;
  }

private:
  device_ref __dev_;
  cudaGraph_t __graph_;
  // TODO should this be a custom class that does inline storage for small counts?
  ::std::vector<cudaGraphNode_t> __nodes_;
};

//! \brief Create a new path builder for a graph builder.
//! \param __gb The graph builder to create the path builder for.
//! \param __nodes The nodes the path builder will depend on.
//! \return A new path builder for the graph builder.
template <typename... Nodes>
[[nodiscard]] _CCCL_HOST_API path_builder start_path(graph_builder_ref __gb, Nodes... __nodes)
{
  path_builder __pb(__gb);
  if constexpr (sizeof...(__nodes) > 0)
  {
    __pb.depends_on(__nodes...);
  }
  return __pb;
}

//! \brief Create a new path builder for a device and a first node.
//! \param __dev The device to create the path builder for.
//! \param __first_node At least one node that the path builder will depend on.
//! \param __nodes Additional nodes that the path builder will depend on.
//! \return A new path builder for the device and the first node.
template <typename _FirstNode, typename... _Nodes>
[[nodiscard]] _CCCL_HOST_API path_builder start_path(device_ref __dev, _FirstNode __first_node, _Nodes... __nodes)
{
  path_builder __pb(__dev, __first_node.get_native_graph_handle());
  __pb.depends_on(__first_node, __nodes...);
  return __pb;
}

} // namespace cuda::experimental

#endif // _CUDAX__GRAPH_PATH_BUILDER_CUH
