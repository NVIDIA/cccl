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

#include <cub/device/dispatch/dispatch_copy_mdspan.cuh>

#include <cuda/__device/device_ref.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__memory_resource/device_memory_pool.h>
#include <cuda/__memory_resource/shared_resource.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

// see also https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/mdarray.hpp
namespace cuda::experimental
{
template <typename _Alloc>
struct __construct_allocator
{
  [[nodiscard]] _CCCL_HOST_API static _Alloc __do(::cuda::device_ref)
  {
    return _Alloc{};
  }
};

template <typename _Resource>
struct __construct_allocator<::cuda::mr::shared_resource<_Resource>>
{
  [[nodiscard]] _CCCL_HOST_API static ::cuda::mr::shared_resource<_Resource> __do(::cuda::device_ref __device)
  {
    return ::cuda::mr::make_shared_resource<_Resource>(__device);
  }
};

template <typename _Allocator>
struct __device_mdarray_allocator
{
  _Allocator __allocator_;

  _CCCL_HIDE_FROM_ABI __device_mdarray_allocator() = default;

  _CCCL_HOST_API explicit __device_mdarray_allocator(const _Allocator& __allocator)
      : __allocator_{__allocator}
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
          typename _Allocator = ::cuda::mr::shared_resource<::cuda::device_memory_pool>>
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

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __mapping_size_bytes() const noexcept
  {
    return this->mapping().required_span_size() * sizeof(element_type);
  }

  _CCCL_HOST_API void __release_storage() noexcept
  {
    if (this->data_handle() != nullptr)
    {
      (this->__get_allocator().deallocate_sync(this->data_handle(), __mapping_size_bytes()));
      static_cast<base_type&>(*this) = base_type{nullptr, this->mapping()};
    }
  }

  _CCCL_HOST_API static _Allocator __get_default_allocator() noexcept
  {
    return __construct_allocator<_Allocator>::__do(::cuda::device_ref{::cuda::__driver::__ctxGetDevice()});
  }

public:
  _CCCL_HOST_API device_mdarray()
      : device_mdarray{mapping_type{}, __get_default_allocator()}
  {}

  _CCCL_HOST_API device_mdarray(mapping_type __mapping)
      : device_mdarray{__mapping, __get_default_allocator()}
  {}

  _CCCL_HOST_API device_mdarray(mapping_type __mapping, _Allocator& __allocator)
      : __allocator_base{__allocator}
      , base_type{static_cast<pointer>(__allocator.allocate_sync(__mapping.required_span_size() * sizeof(element_type))),
                  __mapping}
  {}

  _CCCL_HOST_API device_mdarray(extents_type __ext)
      : device_mdarray{mapping_type{__ext}, __get_default_allocator()}
  {}

  _CCCL_HOST_API device_mdarray(extents_type __ext, _Allocator& __allocator)
      : device_mdarray{mapping_type{__ext}, __allocator}
  {}

  // template <typename... _OtherIndexTypes>
  //_CCCL_HOST_API device_mdarray(_OtherIndexTypes... __exts)
  //     : device_mdarray{extents_type{static_cast<::cuda::std::size_t>(__exts)...}}
  //{}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts)
      : device_mdarray{extents_type{__exts}, __get_default_allocator()}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts, _Allocator& __allocator)
      : device_mdarray{extents_type{__exts}, __allocator}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts)
      : device_mdarray{extents_type{__exts}, __get_default_allocator()}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API device_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts, _Allocator& __allocator)
      : device_mdarray{extents_type{__exts}, __allocator}
  {}

  _CCCL_HOST_API device_mdarray(const device_mdarray& __other)
      : device_mdarray{__other, __other.__get_allocator()}
  {}

  _CCCL_HOST_API device_mdarray(const device_mdarray& __other, ::cuda::std::type_identity_t<_Allocator>& __allocator)
      : __allocator_base{__allocator}
      , base_type{nullptr, __other.mapping()}
  {
    if (__other.data_handle() != nullptr)
    {
      auto __new_data = static_cast<pointer>(this->__get_allocator().allocate_sync(__other.__mapping_size_bytes()));
      static_cast<base_type&>(*this) = base_type{__new_data, __other.mapping()};
      cub::detail::copy_mdspan::copy(__other.view(), view(), ::cudaStream_t{nullptr});
    }
  }

  _CCCL_HOST_API device_mdarray& operator=(const device_mdarray& __other)
  {
    if (this == &__other)
    {
      return *this;
    }
    const bool __realloc = this->__mapping_size_bytes() != __other.__mapping_size_bytes()
                        || this->__get_allocator() != __other.__get_allocator();
    if (__realloc)
    {
      __release_storage();
      this->__get_allocator() = __other.__get_allocator();
      if (__other.data_handle() != nullptr)
      {
        auto __new_data = static_cast<pointer>(this->__get_allocator().allocate_sync(__other.__mapping_size_bytes()));
        static_cast<base_type&>(*this) = base_type{__new_data, __other.mapping()};
      }
      else
      {
        static_cast<base_type&>(*this) = base_type{nullptr, __other.mapping()};
      }
    }
    else // no reallocation needed
    {
      pointer __new_data = this->data_handle();
      if (__other.data_handle() == nullptr)
      {
        __release_storage();
        __new_data = nullptr;
      }
      else if (this->data_handle() == nullptr)
      {
        __new_data = static_cast<pointer>(this->__get_allocator().allocate_sync(__other.__mapping_size_bytes()));
      }
      static_cast<base_type&>(*this) = base_type{__new_data, __other.mapping()};
    }
    if (__other.data_handle() != nullptr)
    {
      cub::detail::copy_mdspan::copy(__other.view(), view(), ::cudaStream_t{nullptr});
    }
    return *this;
  }

  _CCCL_HOST_API device_mdarray(device_mdarray&& __other) noexcept
      : __allocator_base{::cuda::std::move(static_cast<__allocator_base&>(__other))}
      , base_type{::cuda::std::exchange(static_cast<base_type&>(__other), base_type{})}
  {}

  _CCCL_HOST_API
  device_mdarray(device_mdarray&& __other, const ::cuda::std::type_identity_t<_Allocator>& __allocator) noexcept
      : __allocator_base{__allocator}
      , base_type{::cuda::std::exchange(static_cast<base_type&>(__other), base_type{})}
  {}

  _CCCL_HIDE_FROM_ABI device_mdarray& operator=(device_mdarray&& __other) noexcept
  {
    if (this == &__other)
    {
      return *this;
    }
    __release_storage();
    static_cast<base_type&>(*this) = ::cuda::std::exchange(static_cast<base_type&>(__other), base_type{});
    this->__get_allocator()        = ::cuda::std::move(__other.__get_allocator());
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
