//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_FILL_BYTES_CUH
#define _CUDAX__GRAPH_FILL_BYTES_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(12, 2)

#  include <cuda/__algorithm/common.h>
#  include <cuda/__stream/launch_transform.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/span>

#  include <cuda/experimental/__driver/driver_api.cuh>
#  include <cuda/experimental/__graph/concepts.cuh>
#  include <cuda/experimental/__graph/graph_node_ref.cuh>
#  include <cuda/experimental/__graph/path_builder.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _DstTy, ::cuda::std::size_t _DstSize>
_CCCL_HOST_API graph_node_ref
__fill_bytes_graph_impl(path_builder& __pb, ::cuda::std::span<_DstTy, _DstSize> __dst, ::cuda::std::uint8_t __value)
{
  static_assert(!::cuda::std::is_const_v<_DstTy>, "Fill destination can't be const");
  static_assert(::cuda::std::is_trivially_copyable_v<_DstTy>,
                "Fill destination element type must be trivially copyable");

  auto __deps = __pb.get_dependencies();
  ::CUgraphNodeParams __params{};
  __params.type               = ::CU_GRAPH_NODE_TYPE_MEMSET;
  __params.memset.dst         = reinterpret_cast<::CUdeviceptr>(__dst.data());
  __params.memset.pitch       = __dst.size_bytes();
  __params.memset.value       = __value;
  __params.memset.elementSize = 1;
  __params.memset.width       = __dst.size_bytes();
  __params.memset.height      = 1;
  __params.memset.ctx         = __pb.get_device().__primary_context();
  auto __node                 = ::cuda::experimental::__driver::__graphAddNode(
    __pb.get_native_graph_handle(), __deps.data(), __deps.size(), &__params);

  __pb.__clear_and_set_dependency_node(__node);
  return graph_node_ref{__node, __pb.get_native_graph_handle()};
}

template <typename _DstElem, typename _DstExtents, typename _DstLayout, typename _DstAccessor>
_CCCL_HOST_API graph_node_ref __fill_bytes_graph_impl(
  path_builder& __pb,
  ::cuda::std::mdspan<_DstElem, _DstExtents, _DstLayout, _DstAccessor> __dst,
  ::cuda::std::uint8_t __value)
{
  if (!__dst.is_exhaustive())
  {
    _CCCL_THROW(::std::invalid_argument, "fill_bytes supports only exhaustive mdspans");
  }

  return __fill_bytes_graph_impl(
    __pb, ::cuda::std::span(__dst.data_handle(), __dst.mapping().required_span_size()), __value);
}
//! \brief Adds a memset node to a CUDA graph path that bytewise-fills the destination.
//!
//! This overload is selected when the destination (after applying `launch_transform`) is
//! a contiguous range convertible to `cuda::std::span`. The element type must be trivially
//! copyable and non-const. The pointer captured in the node must remain valid until the
//! graph executes.
//!
//! \param __pb    Path builder to insert the node into.
//! \param __dst   Destination memory to fill.
//! \param __value Byte value to write to every byte of the destination.
//! \return A `graph_node_ref` for the newly added memset node.
//! \throws cuda::std::cuda_error if node creation fails.
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(::cuda::__spannable<::cuda::transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API graph_node_ref fill_bytes(path_builder& __pb, _DstTy&& __dst, ::cuda::std::uint8_t __value)
{
  return __fill_bytes_graph_impl(
    __pb,
    ::cuda::std::span(
      ::cuda::launch_transform(::cuda::stream_ref{::cuda::invalid_stream}, ::cuda::std::forward<_DstTy>(__dst))),
    __value);
}

//! @overload
//! This overload is selected when the destination (after applying `launch_transform`) is
//! a `cuda::std::mdspan`. The mdspan must be exhaustive. The element type must be trivially
//! copyable and non-const. The pointer captured in the node must remain valid until the
//! graph executes.
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(::cuda::__mdspannable<::cuda::transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API graph_node_ref fill_bytes(path_builder& __pb, _DstTy&& __dst, ::cuda::std::uint8_t __value)
{
  return __fill_bytes_graph_impl(
    __pb,
    ::cuda::__as_mdspan(
      ::cuda::launch_transform(::cuda::stream_ref{::cuda::invalid_stream}, ::cuda::std::forward<_DstTy>(__dst))),
    __value);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(12, 2)

#endif // _CUDAX__GRAPH_FILL_BYTES_CUH
