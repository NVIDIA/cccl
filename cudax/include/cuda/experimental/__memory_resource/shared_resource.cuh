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
#include <cuda/std/__new_>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/atomic>

namespace cuda::experimental::mr
{

//! @rst
//! .. _cudax-memory-resource-shared-resource:
//!
//! Resource wrapper to share ownership of a resource
//! --------------------------------------------------
//!
//! ``shared_resource`` holds a reference counted instance of a memory resource. This allows
//! the user to pass a resource around with reference semantics while avoiding lifetime issues.
//!
//! @note ``shared_resource`` satisfies the ``cuda::mr::async_resource`` concept iff \tparam _Resource satisfies it.
//! @tparam _Resource The resource type to hold.
//! @endrst
template <class _Resource>
struct shared_resource
{
  static_assert(_CUDA_VMR::resource<_Resource>, "");

  //! @brief Constructs a \c shared_resource refering to an object of type \c _Resource
  //! that has been constructed with arguments \c __args. The \c _Resource object is
  //! dynamically allocated with \c new.
  //! @param __args The arguments to be passed to the \c _Resource constructor.
  template <class... _Args>
  explicit shared_resource(_Args&&... __args)
      : __control_block(new _Control_block{_Resource{_CUDA_VSTD::forward<_Args>(__args)...}, 1})
  {}

  //! @brief Copy-constructs a \c shared_resource object resulting in an copy that shares
  //! ownership of the wrapped resource with \c __other.
  //! @param __other The \c shared_resource object to copy from.
  shared_resource(const shared_resource& __other) noexcept
      : __control_block(__other.__control_block)
  {
    if (__control_block)
    {
      __control_block->__ref_count.fetch_add(1, _CUDA_VSTD::memory_order_relaxed);
    }
  }

  //! @brief Move-constructs a \c shared_resource assuming ownership of the resource stored
  //! in \c __other.
  //! @param __other The \c shared_resource object to move from.
  //! @post \c __other is left in a valid but unspecified state.
  shared_resource(shared_resource&& __other) noexcept
      : __control_block(_CUDA_VSTD::exchange(__other.__control_block, nullptr))
  {}

  //! @brief Releases the reference held by this \c shared_resource object. If this is the
  //! last reference to the wrapped resource, the resource is deleted.
  ~shared_resource()
  {
    if (__control_block && __control_block->__ref_count.fetch_sub(1, _CUDA_VSTD::memory_order_acq_rel) == 1)
    {
      delete __control_block;
    }
  }

  //! @brief Copy-assigns from \c __other. Self-assignment is a no-op. Otherwise, the reference
  //! held by this \c shared_resource object is released and a new reference is acquired to the
  //! wrapped resource of \c __other, if any.
  //! @param __other The \c shared_resource object to copy from.
  shared_resource& operator=(const shared_resource& __other) noexcept
  {
    if (this != &__other)
    {
      shared_resource(__other).swap(*this);
    }

    return *this;
  }

  //! @brief Move-assigns from \c __other. Self-assignment is a no-op. Otherwise, the reference
  //! held by this \c shared_resource object is released, while the reference held by \c __other
  //! is transferred to this object.
  //! @param __other The \c shared_resource object to move from.
  /// @post \c __other is left in a valid but unspecified state.
  shared_resource& operator=(shared_resource&& __other) noexcept
  {
    if (this != &__other)
    {
      shared_resource(_CUDA_VSTD::move(__other)).swap(*this);
    }

    return *this;
  }

  //! @brief Swaps a \c shared_resource with another one.
  //! @param __other The other \c shared_resource.
  void swap(shared_resource& __other) noexcept
  {
    _CUDA_VSTD::swap(__control_block, __other.__control_block);
  }

  //! @brief Swaps a \c shared_resource with another one.
  //! @param __other The other \c shared_resource.
  friend void swap(shared_resource& __left, shared_resource& __right) noexcept
  {
    __left.swap(__right);
  }

  //! @brief Allocate memory of size at least \p __bytes using the stored resource.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD void* allocate(size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    return __control_block->__resource.allocate(__bytes, __alignment);
  }

  //! @brief Deallocate memory pointed to by \p __ptr using the stored resource.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  void deallocate(void* __ptr, size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t)) noexcept
  {
    __control_block->__resource.deallocate(__ptr, __bytes, __alignment);
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
  _CCCL_NODISCARD void* async_allocate(size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return this->__control_block->__resource.async_allocate(__bytes, __alignment, __stream);
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
  void async_deallocate(void* __ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    this->__control_block->__resource.async_deallocate(__ptr, __bytes, __alignment, __stream);
  }

  //! @brief Equality comparison between two \c shared_resource
  //! @param __lhs The first \c shared_resource
  //! @param __rhs The other \c shared_resource
  //! @return Checks whether the objects refer to resources that compare equal.
  _CCCL_NODISCARD_FRIEND bool operator==(const shared_resource& __lhs, const shared_resource& __rhs)
  {
    if (__lhs.__control_block == __rhs.__control_block)
    {
      return true;
    }

    if (__lhs.__control_block == nullptr || __rhs.__control_block == nullptr)
    {
      return false;
    }

    return __lhs.__control_block->__resource == __rhs.__control_block->__resource;
  }

  //! @brief Equality comparison between two \c shared_resource
  //! @param __lhs The first \c shared_resource
  //! @param __rhs The other \c shared_resource
  //! @return Checks whether the objects refer to resources that compare unequal.
  _CCCL_NODISCARD_FRIEND bool operator!=(const shared_resource& __lhs, const shared_resource& __rhs)
  {
    return !(__lhs == __rhs);
  }

  //! @brief Forwards the stateless properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND(has_property<_Resource, _Property>))
  friend void get_property(const shared_resource&, _Property) noexcept {}

  //! @brief Forwards the stateful properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(property_with_value<_Property> _LIBCUDACXX_AND(has_property<_Resource, _Property>))
  _CCCL_NODISCARD_FRIEND __property_value_t<_Property> get_property(const shared_resource& __self, _Property) noexcept
  {
    return get_property(__self.__control_block->__resource, _Property{});
  }

private:
  // Use a custom shared_ptr implementation because (a) we don't need to support weak_ptr so we only
  // need one pointer, not two, and (b) this implementation can work on device also.
  struct _Control_block
  {
    _Resource __resource;
    _CUDA_VSTD::atomic<int> __ref_count;
  };

  _Control_block* __control_block;
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
auto make_shared_resource(_Args&&... __args) -> shared_resource<_Resource>
{
  static_assert(_CUDA_VMR::resource<_Resource>, "_Resource does not satisfy the cuda::mr::resource concept");
  return shared_resource<_Resource>{_CUDA_VSTD::forward<_Args>(__args)...};
}

} // namespace cuda::experimental::mr

#endif // _CUDAX__MEMORY_RESOURCE_SHARED_RESOURCE_H
