//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
#define _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/any_resource.h>
#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/std/__concepts/_One_of.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>

#if _CCCL_STD_VER >= 2014

namespace cuda::experimental::mr
{

template <class>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_basic_any_resource = false;

template <_CUDA_VMR::_AllocType _Alloc_type, class... _Properties>
class basic_any_resource
    : public _CUDA_VMR::_Resource_ref_base<_Alloc_type, false>
    , private _CUDA_VMR::_Filtered_vtable<_Properties...>
{
private:
  template <_CUDA_VMR::_AllocType, class...>
  friend class basic_any_resource;

  template <class...>
  friend struct _CUDA_VMR::_Resource_vtable;

  using __vtable = _CUDA_VMR::_Filtered_vtable<_Properties...>;

  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    _CUDA_VSTD::__all_of<_CUDA_VSTD::_One_of<_Properties, _OtherProperties...>...>;

  void* __get_object() const noexcept
  {
    return this->__static_vtable->__is_small_fn() ? this->__object.__ptr_ : static_cast<void*>(this->__object.__buf_);
  }

public:
  //! @brief Constructs a \c basic_any_resource from a type that satisfies the \c resource or \c async_resource concept
  //! as well as all properties
  //! @param __res The resource to be wrapped within the \c basic_any_resource
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!__is_basic_any_resource<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Default)
                         _LIBCUDACXX_AND resource_with<_Resource, _Properties...>)
  basic_any_resource(_Resource& __res) noexcept
      : _Resource_ref_base<_Alloc_type, true>(_CUDA_VSTD::addressof(__res), &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Constructs a \c any_resource from a type that satisfies the \c async_resource concept  as well as all
  //! properties. This ignores the async interface of the passed in resource
  //! @param __res The resource to be wrapped within the \c any_resource
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!__is_basic_any_resource<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Async)
                         _LIBCUDACXX_AND async_resource_with<_Resource, _Properties...>)
  basic_any_resource(_Resource& __res) noexcept
      : _Resource_ref_base<_Alloc_type, true>(_CUDA_VSTD::addressof(__res), &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Constructs a \c basic_any_resource from a type that satisfies the \c resource or \c async_resource concept
  //! as well as all properties
  //! @param __res Pointer to a resource to be wrapped within the \c basic_any_resource
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!__is_basic_any_resource<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Default)
                         _LIBCUDACXX_AND resource_with<_Resource, _Properties...>)
  basic_any_resource(_Resource* __res) noexcept
      : _Resource_ref_base<_Alloc_type, true>(__res, &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Constructs a \c any_resource from a type that satisfies the \c async_resource concept  as well as all
  //! properties. This ignores the async interface of the passed in resource
  //! @param __res Pointer to a resource to be wrapped within the \c any_resource
  _LIBCUDACXX_TEMPLATE(class _Resource, _AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES((!__is_basic_any_resource<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _AllocType::_Async)
                         _LIBCUDACXX_AND async_resource_with<_Resource, _Properties...>)
  basic_any_resource(_Resource* __res) noexcept
      : _Resource_ref_base<_Alloc_type, true>(__res, &__alloc_vtable<_Alloc_type, _Resource>)
      , __vtable(__vtable::template _Create<_Resource>())
  {}

  //! @brief Conversion from a \c basic_any_resource with the same set of properties but in a different order
  //! @param __ref The other \c basic_any_resource
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES(__properties_match<_OtherProperties...>)
  basic_any_resource(basic_any_resource<_Alloc_type, _OtherProperties...> __ref) noexcept
      : _Resource_ref_base<_Alloc_type, true>(__ref.__object, __ref.__static_vtable)
      , __vtable(__ref)
  {}

  //! @brief Conversion from a \c async_any_resource with the same set of properties but in a different order to a
  //! \c any_resource
  //! @param __ref The other \c async_any_resource
  _LIBCUDACXX_TEMPLATE(_AllocType _OtherAllocType, class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((_OtherAllocType == _AllocType::_Async) _LIBCUDACXX_AND(_OtherAllocType != _Alloc_type)
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  basic_any_resource(basic_any_resource<_OtherAllocType, _OtherProperties...> __ref) noexcept
      : _Resource_ref_base<_Alloc_type, true>(__ref.__object, __ref.__static_vtable)
      , __vtable(__ref)
  {}

  basic_any_resource(basic_any_resource&)            = delete;
  basic_any_resource& operator=(basic_any_resource&) = delete;

  basic_any_resource(basic_any_resource&&) = delete;
  basic_any_resource& operator=(basic_any_resource&& __other) noexcept
  {
    // destroy the excisting resource
    this->~basic_any_resource();

    // construct from
    _CUDA_VSTD::__construct_at(this, __other);
  }

  //! @brief Destroys the stored resource
  ~basic_any_resource() noexcept
  {
    this->__static_vtable->__destroy_fn(this->__get_object());
    if (!this->__static_vtable->__is_small_fn()) // free the allocated storage
    {
      delete this->__get_object();
    }
  }

  //! @brief Equality comparison between two \c basic_any_resource
  //! @param __rhs The other \c basic_any_resource
  //! @return Checks whether both resources have the same equality function stored in their vtable and if so returns
  //! the result of that equality comparison. Otherwise returns false.
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator==(const basic_any_resource<_Alloc_type, _OtherProperties...>& __rhs) const
  {
    return (this->__static_vtable->__equal_fn == __rhs.__static_vtable->__equal_fn)
        && this->__static_vtable->__equal_fn(this->__object, __rhs.__object);
  }

  //! @brief Inequality comparison between two \c basic_any_resource
  //! @param __rhs The other \c basic_any_resource
  //! @return Checks whether both resources have the same equality function stored in their vtable and if so returns
  //! the inverse result of that equality comparison. Otherwise returns true.
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((sizeof...(_Properties) == sizeof...(_OtherProperties))
                         _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  _CCCL_NODISCARD bool operator!=(const basic_any_resource<_Alloc_type, _OtherProperties...>& __rhs) const
  {
    return !(*this == __rhs);
  }

  //! @brief Forwards the stateless properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>)
  friend void get_property(const basic_any_resource&, _Property) noexcept {}

  //! @brief Forwards the stateful properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(property_with_value<_Property> _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>)
  _CCCL_NODISCARD_FRIEND __property_value_t<_Property> get_property(const basic_any_resource& __res, _Property) noexcept
  {
    return __res._Property_vtable<_Property>::__property_fn(__res.__object);
  }
};

//! @brief Checks whether a passed in type is a specialization of basic_any_resource
template <_AllocType _Alloc_type, class... _Properties>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_basic_any_resource<basic_any_resource<_Alloc_type, _Properties...>> = true;

//! @brief Type erased wrapper around a `resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any resource wrapped within the `any_resource` needs to satisfy
template <class... _Properties>
using any_resource = basic_any_resource<_AllocType::_Default, _Properties...>;

//! @brief Type erased wrapper around a `async_resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any async resource wrapped within the `async_any_resource` needs to satisfy
template <class... _Properties>
using async_any_resource = basic_any_resource<_AllocType::_Async, _Properties...>;
} // namespace cuda::experimental::mr

#endif // _CCCL_STD_VER >= 2014
#endif //_CUDA__MEMORY_RESOURCE_ANY_RESOURCE_H
