//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY_DEVICE_CUH
#define __CUDAX__CONTAINER_MDARRAY_DEVICE_CUH

#include <cuda/std/detail/__config>

#include <cuda/__cccl_config>
#include <cuda/__driver/driver_api.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__memory_resource/device_memory_pool.h>
#include <cuda/__memory_resource/shared_resource.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/experimental/__container/mdarray_base.cuh>

#include <cuda/std/__cccl/prologue.h>

// see also https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/mdarray.hpp
namespace cuda::experimental
{
template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy,
          typename _Allocator = ::cuda::mr::shared_resource<::cuda::device_memory_pool>>
class device_mdarray
    : public __base_mdarray<
        device_mdarray<_ElementType, _Extents, _LayoutPolicy, _Allocator>,
        ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy, ::cuda::std::default_accessor<_ElementType>>,
        ::cuda::
          device_mdspan<_ElementType const, _Extents, _LayoutPolicy, ::cuda::std::default_accessor<_ElementType const>>,
        _Allocator>
{
  static_assert(::cuda::has_property<_Allocator, ::cuda::mr::device_accessible>);

  using base_type = __base_mdarray<
    device_mdarray<_ElementType, _Extents, _LayoutPolicy, _Allocator>,
    ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy, ::cuda::std::default_accessor<_ElementType>>,
    ::cuda::device_mdspan<_ElementType const, _Extents, _LayoutPolicy, ::cuda::std::default_accessor<_ElementType const>>,
    _Allocator>;

  friend base_type;

  _CCCL_HOST_API static _Allocator __get_default_allocator()
  {
    return __construct_allocator<_Allocator>::__do(::cuda::__devices()[0]);
  }

public:
  using allocator_type = _Allocator;
  using mapping_type   = typename base_type::mapping_type;
  using extents_type   = typename base_type::extents_type;

  using view_type =
    ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy, ::cuda::std::default_accessor<_ElementType>>;

  using const_view_type =
    ::cuda::device_mdspan<_ElementType const, _Extents, _LayoutPolicy, ::cuda::std::default_accessor<_ElementType const>>;

  _CCCL_HOST_API device_mdarray() = default;

  _CCCL_HOST_API device_mdarray(mapping_type __mapping)
      : base_type{__mapping}
  {}

  _CCCL_HOST_API device_mdarray(mapping_type __mapping, allocator_type __allocator)
      : base_type{__mapping, __allocator}
  {}

  _CCCL_HOST_API device_mdarray(extents_type __ext)
      : base_type{__ext}
  {}

  _CCCL_HOST_API device_mdarray(extents_type __ext, allocator_type __allocator)
      : base_type{__ext, __allocator}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts)
      : base_type{__exts}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts, allocator_type __allocator)
      : base_type{__exts, __allocator}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts)
      : base_type{__exts}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts, allocator_type __allocator)
      : base_type{__exts, __allocator}
  {}

  _CCCL_HOST_API device_mdarray(const device_mdarray&) = default;

  _CCCL_HOST_API device_mdarray(device_mdarray&&) noexcept = default;

  _CCCL_HOST_API device_mdarray& operator=(const device_mdarray&) = default;

  _CCCL_HOST_API device_mdarray& operator=(device_mdarray&&) noexcept = default;

  _CCCL_HOST_API ~device_mdarray() noexcept = default;
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_DEVICE_CUH
