//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_SHARED_RESOURCE_H
#define _CUDAX__MEMORY_RESOURCE_SHARED_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// If the memory resource header was included without the experimental flag,
// tell the user to define the experimental flag.
#if defined(_CUDA_MEMORY_RESOURCE) && !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  error "To use the experimental memory resource, define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
#endif

// cuda::mr is unavable on MSVC 2017
#if defined(_CCCL_COMPILER_MSVC_2017)
#  error "The shared_resource header is not supported on MSVC 2017"
#endif

#if !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#endif

#include <cuda/__memory_resource/resource.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/config.cuh>

#include <memory>

namespace cuda::experimental::mr
{

//! @rst
//! .. _cudax-memory-resource-shared-resource:
//!
//! Resource wrapper to share ownership of a resource
//! --------------------------------------------------
//!
//! ``shared_resource`` holds a reference counted instance of a memory resource.
//! This allows the user to pass a resource around with reference semantics while
//! avoiding lifetime issues. It can be used directly, like so:
//!
//! .. code-block:: cpp
//!   auto mr = shared_resource< MyCustomResource >{ args... };
//!
//!
//! It can also be used as a mixin when defining a custom resource. When used
//! as a mixin, the implementation of the resource should be in a nested type
//! named ``impl`` as shown below.
//!
//! .. code-block:: cpp
//!   struct MyCustomSharedResource : shared_resource< MyCustomSharedResource >
//!   {
//!     MyCustomSharedResource( Arg arg )
//!         : shared_resource( arg )
//!     {}
//!
//!   private:
//!     friend shared_resource;
//!
//!     struct impl // put the implementation of the resource here
//!     {
//!       impl( Arg arg )
//!       impl(impl&&) = delete; // immovable impls are ok
//!
//!       void* allocate(size_t, size_t);
//!       void deallocate(void*, size_t, size_t);
//!       bool operator==(const impl&) const = default;
//!
//!     private:
//!       Arg arg;
//!     };
//!   };
//!
//! Instances of ``MyCustomSharedResource`` satisfy the ``cuda::mr::resource``
//! concept. Copies all share ownership of the underlying ``impl`` object.
//!
//! @note ``shared_resource`` satisfies the ``cuda::mr::async_resource`` concept
//! iff \tparam _Resource satisfies it.
//!
//! @tparam _Resource The resource type to hold, or the derived type when using
//! ``shared_resource`` as a mixin. When used directly, \c _Resource must
//! satisfy the :ref:`resource <libcudacxx-extended-api-memory-resources-resource>`
//! concept. When used as a mixin, there must be an accessible nested type
//! ``_Resource::impl`` that satisfies the
//! :ref:`resource <libcudacxx-extended-api-memory-resources-resource>`
//! concept.
//! @endrst
template <class _Resource>
struct shared_resource
{
  //! @brief Constructs a \c shared_resource refering to an object of type \c _Resource
  //! that has been constructed with arguments \c __args. The \c _Resource object is
  //! dynamically allocated with \c new.
  //! @param __args The arguments to be passed to the \c _Resource constructor.
  template <class... _Args>
  _CUDAX_HOST_API explicit shared_resource(_Args&&... __args)
      : __pimpl_(::std::make_shared<__impl>(_CUDA_VSTD::forward<_Args>(__args)...))
  {
    static_assert(_CUDA_VMR::resource<_Resource>, "The Resource type does not satisfy the cuda::mr::resource concept.");
  }

  //! @brief Swaps a \c shared_resource with another one.
  //! @param __other The other \c shared_resource.
  _CUDAX_HOST_API void swap(shared_resource& __other) noexcept
  {
    __pimpl_.swap(__other.__pimpl_);
  }

  //! @brief Swaps a \c shared_resource with another one.
  //! @param __other The other \c shared_resource.
  _CUDAX_TRIVIAL_HOST_API friend void swap(shared_resource& __left, shared_resource& __right) noexcept
  {
    __left.swap(__right);
  }

  //! @brief Allocate memory of size at least \p __bytes using the stored resource.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD _CUDAX_HOST_API void* allocate(size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    return __pimpl_->allocate(__bytes, __alignment);
  }

  //! @brief Deallocate memory pointed to by \p __ptr using the stored resource.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  _CUDAX_HOST_API void
  deallocate(void* __ptr, size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t)) noexcept
  {
    __pimpl_->deallocate(__ptr, __bytes, __alignment);
  }

  //! @brief Enqueues an allocation of memory of size at least \p __bytes using
  //! the wrapped resource. The allocation is performed asynchronously on stream \c __stream.
  //! @pre \c _Resource must satisfy \c async_resource.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @return Pointer to the newly allocated memory.
  //! @note The caller is responsible for ensuring that the memory is not accessed until the
  //! operation has completed.
  _LIBCUDACXX_TEMPLATE(class _ThisResource = _Resource)
  _LIBCUDACXX_REQUIRES(_CUDA_VMR::async_resource<_ThisResource>)
  _CCCL_NODISCARD _CUDAX_HOST_API void* async_allocate(size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return this->__pimpl_->async_allocate(__bytes, __alignment, __stream);
  }

  //! @brief Enqueues the deallocation of memory pointed to by \c __ptr. The deallocation is
  //! performed asynchronously on stream \c __stream.
  //! @pre \c _Resource must satisfy \c async_resource.
  //! @param __bytes The number of bytes that was passed to the `async_allocate` call that returned
  //! \p __ptr.
  //! @param __alignment The alignment that was passed to the `async_allocate` call that returned
  //! \p __ptr.
  //! @note The caller is responsible for ensuring that the memory is not accessed after the
  //! operation has completed.
  _LIBCUDACXX_TEMPLATE(class _ThisResource = _Resource)
  _LIBCUDACXX_REQUIRES(_CUDA_VMR::async_resource<_ThisResource>)
  _CUDAX_HOST_API void async_deallocate(void* __ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    this->__pimpl_->async_deallocate(__ptr, __bytes, __alignment, __stream);
  }

  //! @brief Equality comparison between two \c shared_resource
  //! @param __lhs The first \c shared_resource
  //! @param __rhs The other \c shared_resource
  //! @return Checks whether the objects refer to resources that compare equal.
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API bool operator==(const shared_resource& __lhs, const shared_resource& __rhs)
  {
    if (__lhs.__pimpl_ == __rhs.__pimpl_)
    {
      return true;
    }

    if (__lhs.__pimpl_ == nullptr || __rhs.__pimpl_ == nullptr)
    {
      return false;
    }

    return *__lhs.__pimpl_ == *__rhs.__pimpl_;
  }

  //! @brief Equality comparison between two \c shared_resource
  //! @param __lhs The first \c shared_resource
  //! @param __rhs The other \c shared_resource
  //! @return Checks whether the objects refer to resources that compare unequal.
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API bool operator!=(const shared_resource& __lhs, const shared_resource& __rhs)
  {
    return !(__lhs == __rhs);
  }

  //! @brief Forwards the stateless properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND(has_property<_Resource, _Property>))
  _CUDAX_HOST_API friend void get_property(const shared_resource&, _Property) noexcept {}

  //! @brief Forwards the stateful properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(property_with_value<_Property> _LIBCUDACXX_AND(has_property<_Resource, _Property>))
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API __property_value_t<_Property>
  get_property(const shared_resource& __self, _Property) noexcept
  {
    return get_property(*__self.__pimpl_, _Property{});
  }

private:
  template <class _Ty>
  _CUDAX_HOST_API static _CUDA_VSTD::true_type __impl_test(int, typename _Ty::impl* = nullptr);
  template <class _Ty>
  _CUDAX_HOST_API static _CUDA_VSTD::false_type __impl_test(long);

  template <class _Ty>
  using __has_impl = decltype(shared_resource::__impl_test<_Ty>(0));

  template <bool HasImpl, class = void>
  struct __impl_base : _Resource::impl
  {
    template <class... _Args>
    _CUDAX_TRIVIAL_HOST_API __impl_base(_Args&&... __args)
        : _Resource::impl(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
  };

  template <class _Ty>
  struct __impl_base<false, _Ty> : _Resource
  {
    static_assert(!_CUDA_VSTD::is_base_of_v<shared_resource, _Resource>,
                  "It looks like shared_resource is being used as a mixin, but the specified Resource does not have an "
                  "accessible nested type Resource::impl");

    template <class... _Args>
    _CUDAX_TRIVIAL_HOST_API __impl_base(_Args&&... __args)
        : _Resource(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
  };

  struct __impl
      : __impl_base<__has_impl<_Resource>::value>
      , ::std::enable_shared_from_this<__impl>
  {
    template <class... _Args>
    _CUDAX_TRIVIAL_HOST_API __impl(_Args&&... __args)
        : __impl::__impl_base(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
  };

  ::std::shared_ptr<__impl> __pimpl_;
};

//! @rst
//! .. _cudax-memory-resource-make-shared-resource:
//!
//! Factory function for `shared_resource` objects
//! -----------------------------------------------
//!
//! ``make_any_resource`` constructs an :ref:`shared_resource <cudax-memory-resource-shared-resource>` object that wraps
//! a newly constructed instance of the given resource type. The resource type must satisfy the ``cuda::mr::resource``
//! concept and provide all of the properties specified in the template parameter pack.
//!
//! @param __args The arguments used to construct the instance of the resource type.
//!
//! @endrst
template <class _Resource, class... _Args>
_CUDAX_HOST_API auto make_shared_resource(_Args&&... __args) -> shared_resource<_Resource>
{
  static_assert(_CUDA_VMR::resource<_Resource>, "_Resource does not satisfy the cuda::mr::resource concept");
  return shared_resource<_Resource>{_CUDA_VSTD::forward<_Args>(__args)...};
}

} // namespace cuda::experimental::mr

#endif // _CUDAX__MEMORY_RESOURCE_SHARED_RESOURCE_H
