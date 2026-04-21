//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_COPY_BYTES_CUH
#define _CUDAX__GRAPH_COPY_BYTES_CUH

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
#  include <cuda/std/cstddef>
#  include <cuda/std/span>

#  include <cuda/experimental/__driver/driver_api.cuh>
#  include <cuda/experimental/__graph/graph_node_ref.cuh>
#  include <cuda/experimental/__graph/path_builder.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _SrcTy, typename _DstTy>
_CCCL_HOST_API graph_node_ref
__copy_bytes_graph_impl(path_builder& __pb, ::cuda::std::span<_SrcTy> __src, ::cuda::std::span<_DstTy> __dst)
{
  static_assert(!::cuda::std::is_const_v<_DstTy>, "Copy destination can't be const");
  static_assert(::cuda::std::is_trivially_copyable_v<_SrcTy> && ::cuda::std::is_trivially_copyable_v<_DstTy>,
                "Copy source and destination element types must be trivially copyable");

  if (__src.size_bytes() > __dst.size_bytes())
  {
    _CCCL_THROW(::std::invalid_argument, "Copy destination is too small to fit the source data");
  }

  if (__src.size_bytes() == 0)
  {
    return graph_node_ref{};
  }

  auto __deps = __pb.get_dependencies();
  ::CUgraphNodeParams __params{};
  __params.type                            = ::CU_GRAPH_NODE_TYPE_MEMCPY;
  __params.memcpy.copyCtx                  = __pb.get_device().__primary_context();
  __params.memcpy.copyParams.srcMemoryType = ::CU_MEMORYTYPE_UNIFIED;
  __params.memcpy.copyParams.srcDevice     = reinterpret_cast<::CUdeviceptr>(__src.data());
  __params.memcpy.copyParams.dstMemoryType = ::CU_MEMORYTYPE_UNIFIED;
  __params.memcpy.copyParams.dstDevice     = reinterpret_cast<::CUdeviceptr>(__dst.data());
  __params.memcpy.copyParams.WidthInBytes  = __src.size_bytes();
  __params.memcpy.copyParams.Height        = 1;
  __params.memcpy.copyParams.Depth         = 1;
  auto __node                              = ::cuda::experimental::__driver::__graphAddNode(
    __pb.get_native_graph_handle(), __deps.data(), __deps.size(), &__params);

  __pb.__clear_and_set_dependency_node(__node);
  return graph_node_ref{__node, __pb.get_native_graph_handle()};
}

template <typename _SrcElem,
          typename _SrcExtents,
          typename _SrcLayout,
          typename _SrcAccessor,
          typename _DstElem,
          typename _DstExtents,
          typename _DstLayout,
          typename _DstAccessor>
_CCCL_HOST_API graph_node_ref __copy_bytes_graph_impl(
  path_builder& __pb,
  ::cuda::std::mdspan<_SrcElem, _SrcExtents, _SrcLayout, _SrcAccessor> __src,
  ::cuda::std::mdspan<_DstElem, _DstExtents, _DstLayout, _DstAccessor> __dst)
{
  static_assert(::cuda::std::is_constructible_v<_DstExtents, _SrcExtents>,
                "Multidimensional copy requires both source and destination extents to be compatible");
  static_assert(::cuda::std::is_same_v<_SrcLayout, _DstLayout>,
                "Multidimensional copy requires both source and destination layouts to match");

  if (!__dst.is_exhaustive())
  {
    _CCCL_THROW(::std::invalid_argument, "copy_bytes supports only exhaustive mdspans");
  }

  if (__src.extents() != __dst.extents())
  {
    _CCCL_THROW(::std::invalid_argument, "Copy destination size differs from the source");
  }

  return __copy_bytes_graph_impl(
    __pb,
    ::cuda::std::span(__src.data_handle(), __src.mapping().required_span_size()),
    ::cuda::std::span(__dst.data_handle(), __dst.mapping().required_span_size()));
}
//! \brief Adds a memcpy node to a CUDA graph path that copies bytes from source to destination.
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(::cuda::__spannable<::cuda::transformed_device_argument_t<_SrcTy>>
                 _CCCL_AND ::cuda::__spannable<::cuda::transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API graph_node_ref copy_bytes(path_builder& __pb, _SrcTy&& __src, _DstTy&& __dst)
{
  return __copy_bytes_graph_impl(
    __pb,
    ::cuda::std::span(
      ::cuda::launch_transform(::cuda::stream_ref{::cuda::invalid_stream}, ::cuda::std::forward<_SrcTy>(__src))),
    ::cuda::std::span(
      ::cuda::launch_transform(::cuda::stream_ref{::cuda::invalid_stream}, ::cuda::std::forward<_DstTy>(__dst))));
}

//! \brief Adds a memcpy node for mdspan source and destination.
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(::cuda::__mdspannable<::cuda::transformed_device_argument_t<_SrcTy>>
                 _CCCL_AND ::cuda::__mdspannable<::cuda::transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API graph_node_ref copy_bytes(path_builder& __pb, _SrcTy&& __src, _DstTy&& __dst)
{
  return __copy_bytes_graph_impl(
    __pb,
    ::cuda::__as_mdspan(
      ::cuda::launch_transform(::cuda::stream_ref{::cuda::invalid_stream}, ::cuda::std::forward<_SrcTy>(__src))),
    ::cuda::__as_mdspan(
      ::cuda::launch_transform(::cuda::stream_ref{::cuda::invalid_stream}, ::cuda::std::forward<_DstTy>(__dst))));
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(12, 2)

#endif // _CUDAX__GRAPH_COPY_BYTES_CUH
