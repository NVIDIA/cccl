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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__memory_resource/device_memory_pool.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

// see also https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/mdarray.hpp
namespace cuda::experimental
{
// Helper base class to ensure allocator and pointer are initialized before mdspan
template <typename _Allocator, typename _ElementType>
struct __device_mdarray_storage
{
  using pointer        = _ElementType*;
  using allocator_type = _Allocator;

  allocator_type __allocator{::cuda::device_ref{0}};
  pointer __ptr{nullptr};

  _CCCL_HIDE_FROM_ABI __device_mdarray_storage() = default;

  _CCCL_HOST_API __device_mdarray_storage(::cuda::device_ref __device, ::cuda::std::size_t __size)
      : __allocator{__device}
      , __ptr{static_cast<pointer>(__allocator.allocate_sync(__size))}
  {}
};

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy,
          typename _Allocator = ::cuda::device_memory_pool>
class device_mdarray
    : private __device_mdarray_storage<_Allocator, _ElementType>
    , public ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>
{
  static_assert(cuda::has_property<_Allocator, cuda::mr::device_accessible>);

public:
  using base_type = ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>;

private:
  using __storage_base = __device_mdarray_storage<_Allocator, _ElementType>;

public:
  using allocator_type = _Allocator;
  using extents_type   = typename base_type::extents_type;
  using mapping_type   = typename base_type::mapping_type;
  using layout_type    = _LayoutPolicy;
  using accessor_type  = ::cuda::std::default_accessor<_ElementType>;
  using element_type   = _ElementType;
  using pointer        = element_type*;

  using view_type       = ::cuda::device_mdspan<element_type, extents_type, layout_type, accessor_type>;
  using const_view_type = ::cuda::device_mdspan<element_type const, extents_type, layout_type, accessor_type>;

  _CCCL_HIDE_FROM_ABI device_mdarray() = default;

  _CCCL_HOST_API device_mdarray(mapping_type __mapping, ::cuda::device_ref __device = ::cuda::device_ref{0})
      : __storage_base{__device, __mapping.required_span_size()}
      , base_type{this->__ptr, __mapping}
  {}

  _CCCL_HOST_API device_mdarray(extents_type __ext, ::cuda::device_ref __device = ::cuda::device_ref{0})
      : device_mdarray{mapping_type{__ext}, __device}
  {}

  template <typename... _OtherIndexTypes>
  _CCCL_HOST_API device_mdarray(::cuda::device_ref __device, _OtherIndexTypes... __exts)
      : device_mdarray{extents_type{static_cast<::cuda::std::size_t>(__exts)...}, __device}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts,
                                ::cuda::device_ref __device = ::cuda::device_ref{0})
      : device_mdarray{extents_type{__exts}, __device}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API
  device_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts, ::cuda::device_ref __device = ::cuda::device_ref{0})
      : device_mdarray{extents_type{__exts}, __device}
  {}

  // cuda::device_memory_pool does not support copy construction
  _CCCL_HIDE_FROM_ABI device_mdarray(const device_mdarray&)            = delete;
  _CCCL_HIDE_FROM_ABI device_mdarray& operator=(const device_mdarray&) = delete;
  // TODO
  _CCCL_HIDE_FROM_ABI device_mdarray(device_mdarray&&)            = delete;
  _CCCL_HIDE_FROM_ABI device_mdarray& operator=(device_mdarray&&) = delete;

  _CCCL_HOST_API ~device_mdarray() noexcept
  {
    this->__allocator.deallocate_sync(this->__ptr, this->mapping().required_span_size());
    this->__ptr = nullptr;
  }

  [[nodiscard]] _CCCL_HOST_API view_type view() noexcept
  {
    return static_cast<view_type>(*this);
  }

  [[nodiscard]] _CCCL_HOST_API const_view_type view() const noexcept
  {
    return static_cast<const_view_type>(*this);
  }

  [[nodiscard]] _CCCL_HOST_API operator view_type() noexcept
  {
    return static_cast<view_type>(*this);
  }

  [[nodiscard]] _CCCL_HOST_API operator const_view_type() noexcept
  {
    return static_cast<const_view_type>(*this);
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_DEVICE_CUH
