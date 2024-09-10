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
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__memory_resource/any_resource.cuh>

#include <memory>

namespace cuda::experimental::mr
{

//! @rst
//! .. _cudax-memory-resource-shared-resource:
//!
//! Resource wrapper to share ownership of a resource
//! --------------------------------------------------
//!
//! ``shared_resource`` holds a reference counted :ref:`any_resource <cudax-memory-resource-basic-any-resource>`.
//! This allows the user to pass a resource around with reference semantics while avoiding lifetime issues.
//!
//! @endrst
template <class... _Properties>
class shared_resource
{
private:
  ::std::shared_ptr<::cuda::experimental::mr::any_resource<_Properties...>> __resource_;

  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    _CUDA_VSTD::__type_set_contains<_CUDA_VSTD::__make_type_set<_OtherProperties...>, _Properties...>;

  template <class _Resource>
  static constexpr bool __different_resource =
    !_CCCL_TRAIT(::cuda::std::is_same, ::cuda::std::remove_cvref_t<_Resource>, shared_resource);

public:
  //! @brief Constructs a \c shared_resource from a type that satisfies the \c resource concept as well as all
  //! properties.
  //! @param __res The resource to be wrapped within the \c shared_resource.
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES(
    __different_resource<_Resource> _LIBCUDACXX_AND(_CUDA_VMR::resource_with<_Resource, _Properties...>))
  shared_resource(_Resource&& __res)
      : __resource_(::std::make_shared<::cuda::experimental::mr::any_resource<_Properties...>>(_CUDA_VSTD::move(__res)))
  {}

  //! @brief Allocate memory of size at least \p __bytes using the stored resource.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD void* allocate(size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    return __resource_->allocate(__bytes, __alignment);
  }

  //! @brief Deallocate memory pointed to by \p __ptr using the stored resource.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  void deallocate(void* __ptr, size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t)) noexcept
  {
    return __resource_->deallocate(__ptr, __bytes, __alignment);
  }

  //! @brief Swaps a \c shared_resource with another one.
  //! @param __other The other \c shared_resource.
  void swap(shared_resource& __other) noexcept
  {
    _CUDA_VSTD::swap(__resource_, __other.__resource_);
  }

  //! @brief Equality comparison between two \c shared_resource
  //! @param __rhs The other \c shared_resource
  //! @return Checks whether both resources have the same equality function stored in their vtable and if so returns
  //! the result of that equality comparison. Otherwise returns false.
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator==(const shared_resource<_OtherProperties...>& __rhs) const
  {
    return (__resource_ && __rhs.__resource_)
           ? (*__resource_ == *__rhs.__resource_)
           : (__resource_ && __rhs.__resource_);
  }

  //! @brief Inequality comparison between two \c shared_resource
  //! @param __rhs The other \c shared_resource
  //! @return Checks whether both resources have the same equality function stored in their vtable and if so returns
  //! the inverse result of that equality comparison. Otherwise returns true.
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator!=(const shared_resource<_OtherProperties...>& __rhs) const
  {
    return __resource_ != __rhs.__resource_;
  }

  //! @brief Forwards the stateless properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND(_CUDA_VSTD::_One_of<_Property, _Properties...>))
  friend void get_property(const shared_resource&, _Property) noexcept {}

  //! @brief Forwards the stateful properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(property_with_value<_Property> _LIBCUDACXX_AND(_CUDA_VSTD::_One_of<_Property, _Properties...>))
  _CCCL_NODISCARD_FRIEND __property_value_t<_Property> get_property(const shared_resource& __res, _Property) noexcept
  {
    _CUDA_VMR::_Property_vtable<_Property> const& __prop = __res;
    return __prop.__property_fn(__res._Get_object());
  }
};
} // namespace cuda::experimental::mr

#endif // _CUDAX__MEMORY_RESOURCE_SHARED_RESOURCE_H
