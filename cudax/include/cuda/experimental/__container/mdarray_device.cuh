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

#include <cub/device/dispatch/dispatch_copy_mdspan.cuh>

#include <cuda/__device/device_ref.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__memory_resource/device_memory_pool.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

// see also https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/mdarray.hpp
namespace cuda::experimental
{
template <class _Allocator>
struct __device_mdarray_allocator
{
  _Allocator __allocator_{::cuda::device_ref{0}};

  _CCCL_HIDE_FROM_ABI __device_mdarray_allocator() = default;

  _CCCL_HOST_API __device_mdarray_allocator(::cuda::device_ref __device)
      : __allocator_{__device}
  {}

  [[nodiscard]] _CCCL_HOST_API _Allocator& __get_allocator() noexcept
  {
    return __allocator_;
  }

  [[nodiscard]] _CCCL_HOST_API const _Allocator& __get_allocator() const noexcept
  {
    return __allocator_;
  }
};

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy,
          typename _Allocator = ::cuda::device_memory_pool>
class device_mdarray
    : private __device_mdarray_allocator<_Allocator>
    , public ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>
{
public:
  using base_type      = ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>;
  using allocator_type = _Allocator;
  using extents_type   = typename base_type::extents_type;
  using mapping_type   = typename base_type::mapping_type;
  using layout_type    = _LayoutPolicy;
  using accessor_type  = ::cuda::std::default_accessor<_ElementType>;
  using element_type   = _ElementType;
  using pointer        = element_type*;

  using view_type       = ::cuda::device_mdspan<element_type, extents_type, layout_type, accessor_type>;
  using const_view_type = ::cuda::
    device_mdspan<element_type const, extents_type, layout_type, ::cuda::std::default_accessor<element_type const>>;

private:
  using __allocator_base = __device_mdarray_allocator<_Allocator>;

  static_assert(::cuda::has_property<_Allocator, ::cuda::mr::device_accessible>);

  ::cuda::device_ref __device_{::cuda::device_ref{0}};

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __mapping_size_bytes() const noexcept
  {
    return this->mapping().required_span_size() * sizeof(element_type);
  }

  _CCCL_HOST_API void __release_storage() noexcept
  {
    if (this->data_handle() != nullptr)
    {
      (this->__get_allocator().deallocate_sync(this->data_handle(), __mapping_size_bytes()));
      this->template __get<0>() = nullptr;
    }
  }

public:
  _CCCL_HIDE_FROM_ABI device_mdarray() = default;

  [[nodiscard]] _CCCL_HOST_API
  device_mdarray(mapping_type __mapping, ::cuda::device_ref __device = ::cuda::device_ref{0})
      : __allocator_base{__device}
      , base_type{static_cast<pointer>(__allocator_base::__get_allocator().allocate_sync(
                    __mapping.required_span_size() * sizeof(element_type))),
                  __mapping}
      , __device_{__device}
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

  _CCCL_HOST_API device_mdarray(const device_mdarray& __other)
      : device_mdarray{__other.mapping(), __other.__device_}
  {
    if (__other.data_handle() != nullptr)
    {
      cub::detail::copy_mdspan::copy(__other.view(), view(), ::cudaStream_t{nullptr});
    }
    else
    {
      this->template __get<0>() = nullptr;
    }
  }

  // only mdarray with the same extents can be assigned
  _CCCL_HOST_API device_mdarray& operator=(const device_mdarray& __other)
  {
    if (this == &__other)
    {
      return *this;
    }
    //_CCCL_THROW_IF(this->extents() != __other.extents(), ::std::invalid_argument{"Extents do not match"});
    _CCCL_VERIFY(this->extents() == __other.extents(), "Extents do not match");
    if (__other.data_handle() != nullptr)
    {
      cub::detail::copy_mdspan::copy(__other.view(), view(), ::cudaStream_t{nullptr});
    }
    return *this;
  }

  _CCCL_HOST_API device_mdarray(device_mdarray&& __other) noexcept
      : __allocator_base{__other.__device_}
      , base_type{::cuda::std::exchange(static_cast<base_type&>(__other), base_type{})}
      , __device_{__other.__device_}
  {}

  // only mdarray with the same extents can be moved assigned
  _CCCL_HIDE_FROM_ABI device_mdarray& operator=(device_mdarray&& __other) noexcept
  {
    if (this == &__other)
    {
      return *this;
    }
    //_CCCL_THROW_IF(this->extents() != __other.extents(), ::std::invalid_argument{"Extents do not match"});
    _CCCL_VERIFY(this->extents() == __other.extents(), "Extents do not match");
    __release_storage();
    static_cast<base_type&>(*this) = ::cuda::std::exchange(static_cast<base_type&>(__other), base_type{});
    __device_                      = __other.__device_;
    return *this;
  }

  _CCCL_HOST_API ~device_mdarray() noexcept
  {
    __release_storage();
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

  [[nodiscard]] _CCCL_HOST_API operator const_view_type() const noexcept
  {
    return static_cast<const_view_type>(*this);
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_DEVICE_CUH
