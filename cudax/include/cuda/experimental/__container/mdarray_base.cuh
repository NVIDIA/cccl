//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY_BASE_CUH
#define __CUDAX__CONTAINER_MDARRAY_BASE_CUH

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

#include <cuda/std/__utility/exchange.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

// see also https://github.com/rapidsai/raft/blob/main/cpp/include/raft/core/mdarray.hpp
namespace cuda::experimental
{
template <typename T, typename R>
void __copy(T&& src, R&& dst, ::cudaStream_t stream)
{
  // TODO: implementation
  cub::detail::copy_mdspan::copy(src, dst, stream);
}

template <typename _Alloc>
struct __construct_allocator
{
  [[nodiscard]] _CCCL_HOST_API static _Alloc __do()
  {
    return _Alloc{};
  }

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

// __mdarray_allocator_wrapper allows to initialize the allocator before allocating the memory
template <typename _Allocator>
struct __mdarray_allocator_wrapper
{
  _Allocator __allocator_;

  _CCCL_HIDE_FROM_ABI __mdarray_allocator_wrapper() = default;

  _CCCL_HOST_API explicit __mdarray_allocator_wrapper(const _Allocator& __allocator)
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

template <typename _Derived, typename _ViewType, typename _ConstViewType, typename _Allocator>
class __base_mdarray
    : private __mdarray_allocator_wrapper<_Allocator>
    , public _ViewType
{
public:
  using mdspan_type    = _ViewType;
  using allocator_type = _Allocator;
  using extents_type   = typename _ViewType::extents_type;
  using mapping_type   = typename _ViewType::mapping_type;
  using element_type   = typename _ViewType::element_type;
  using pointer        = element_type*;

  using view_type       = _ViewType;
  using const_view_type = _ConstViewType;

private:
  using __allocator_base = __mdarray_allocator_wrapper<allocator_type>;

  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t __mapping_size_bytes() const noexcept
  {
    return this->mapping().required_span_size() * sizeof(element_type);
  }

  _CCCL_HOST_API void __release_storage() noexcept
  {
    if (this->data_handle() != nullptr)
    {
      (this->__get_allocator().deallocate_sync(this->data_handle(), __mapping_size_bytes()));
      static_cast<mdspan_type&>(*this) = mdspan_type{nullptr, this->mapping()};
    }
  }

public:
  _CCCL_HOST_API __base_mdarray()
      : __base_mdarray{mapping_type{}, _Derived::__get_default_allocator()}
  {}

  _CCCL_HOST_API __base_mdarray(mapping_type __mapping)
      : __base_mdarray{__mapping, _Derived::__get_default_allocator()}
  {}

  _CCCL_HOST_API __base_mdarray(mapping_type __mapping, allocator_type __allocator)
      : __allocator_base{__allocator}
      , mdspan_type{
          static_cast<pointer>(__allocator.allocate_sync(__mapping.required_span_size() * sizeof(element_type))),
          __mapping}
  {}

  _CCCL_HOST_API __base_mdarray(extents_type __ext)
      : __base_mdarray{mapping_type{__ext}, _Derived::__get_default_allocator()}
  {}

  _CCCL_HOST_API __base_mdarray(extents_type __ext, allocator_type __allocator)
      : __base_mdarray{mapping_type{__ext}, __allocator}
  {}

  // template <typename... _OtherIndexTypes>
  //_CCCL_HOST_API __base_mdarray(_OtherIndexTypes... __exts)
  //     : __base_mdarray{extents_type{static_cast<::cuda::std::size_t>(__exts)...}}
  //{}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API __base_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts)
      : __base_mdarray{extents_type{__exts}, _Derived::__get_default_allocator()}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API __base_mdarray(const ::cuda::std::array<_OtherIndexType, _Size>& __exts, allocator_type __allocator)
      : __base_mdarray{extents_type{__exts}, __allocator}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API __base_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts)
      : __base_mdarray{extents_type{__exts}, _Derived::__get_default_allocator()}
  {}

  template <typename _OtherIndexType, ::cuda::std::size_t _Size>
  _CCCL_HOST_API __base_mdarray(::cuda::std::span<_OtherIndexType, _Size> __exts, allocator_type __allocator)
      : __base_mdarray{extents_type{__exts}, __allocator}
  {}

  _CCCL_HOST_API __base_mdarray(const __base_mdarray& __other)
      : __base_mdarray{__other, __other.__get_allocator()}
  {}

  _CCCL_HOST_API __base_mdarray(const __base_mdarray& __other, ::cuda::std::type_identity_t<allocator_type> __allocator)
      : __allocator_base{__allocator}
      , mdspan_type{nullptr, __other.mapping()}
  {
    if (__other.data_handle() != nullptr)
    {
      auto __new_data = static_cast<pointer>(this->__get_allocator().allocate_sync(__other.__mapping_size_bytes()));
      static_cast<mdspan_type&>(*this) = mdspan_type{__new_data, __other.mapping()};
      ::cuda::experimental::__copy(__other.view(), view(), ::cudaStream_t{nullptr});
    }
  }

  _CCCL_HOST_API __base_mdarray& operator=(const __base_mdarray& __other)
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
        static_cast<mdspan_type&>(*this) = mdspan_type{__new_data, __other.mapping()};
      }
      else
      {
        static_cast<mdspan_type&>(*this) = mdspan_type{nullptr, __other.mapping()};
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
      static_cast<mdspan_type&>(*this) = mdspan_type{__new_data, __other.mapping()};
    }
    if (__other.data_handle() != nullptr)
    {
      ::cuda::experimental::__copy(__other.view(), view(), ::cudaStream_t{nullptr});
    }
    return *this;
  }

  _CCCL_HOST_API __base_mdarray(__base_mdarray&& __other) noexcept
      : __allocator_base{::cuda::std::move(static_cast<__allocator_base&>(__other))}
      , mdspan_type{::cuda::std::exchange(static_cast<mdspan_type&>(__other), mdspan_type{})}
  {}

  _CCCL_HOST_API
  __base_mdarray(__base_mdarray&& __other, const ::cuda::std::type_identity_t<allocator_type> __allocator) noexcept
      : __allocator_base{__allocator}
      , mdspan_type{::cuda::std::exchange(static_cast<mdspan_type&>(__other), mdspan_type{})}
  {}

  _CCCL_HIDE_FROM_ABI __base_mdarray& operator=(__base_mdarray&& __other) noexcept
  {
    if (this == &__other)
    {
      return *this;
    }
    __release_storage();
    static_cast<mdspan_type&>(*this) = ::cuda::std::exchange(static_cast<mdspan_type&>(__other), mdspan_type{});
    this->__get_allocator()          = ::cuda::std::move(__other.__get_allocator());
    return *this;
  }

  _CCCL_HOST_API ~__base_mdarray() noexcept
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

#endif //__CUDAX__CONTAINER_MDARRAY_BASE_CUH
