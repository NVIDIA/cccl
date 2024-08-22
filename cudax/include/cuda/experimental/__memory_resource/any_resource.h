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

// If the memory resource header was included without the experimental flag,
// tell the user to define the experimental flag.
#if defined(_CUDA_MEMORY_RESOURCE) && !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  error "To use the experimental memory resource, define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
#endif

// cuda::mr is unavable on MSVC 2017
#if defined(_CCCL_COMPILER_MSVC_2017)
#  error "The any_resource header is not supported on MSVC 2017"
#endif

#if !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#endif

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__concepts/_One_of.h>
#include <cuda/std/__concepts/all_of.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>

namespace cuda::experimental::mr
{
template <class _Ty, class _Uy = _CUDA_VSTD::remove_cvref_t<_Ty>>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_basic_any_resource = false;

template <_CUDA_VMR::_AllocType _Alloc_type, class... _Properties>
class basic_any_resource
    : public _CUDA_VMR::_Resource_base<_Alloc_type, _CUDA_VMR::_WrapperType::_Owning>
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
    _CUDA_VSTD::__type_set_contains<_CUDA_VSTD::__type_set<_OtherProperties...>, _Properties...>;

  template <class _Resource, class __resource_t = _CUDA_VSTD::remove_cvref_t<_Resource>>
  basic_any_resource(_CUDA_VSTD::in_place_t, _Resource&& __res) noexcept(_CUDA_VMR::_IsSmall<__resource_t>())
      : _CUDA_VMR::_Resource_base<_Alloc_type, _CUDA_VMR::_WrapperType::_Owning>(
          nullptr, &_CUDA_VMR::__alloc_vtable<_Alloc_type, _CUDA_VMR::_WrapperType::_Owning, __resource_t>)
      , __vtable(__vtable::template _Create<__resource_t>())
  {
    if constexpr (_CUDA_VMR::_IsSmall<__resource_t>())
    {
      ::new (static_cast<void*>(this->__object.__buf_)) __resource_t(_CUDA_VSTD::forward<_Resource>(__res));
    }
    else
    {
      this->__object.__ptr_ = new __resource_t(_CUDA_VSTD::forward<_Resource>(__res));
    }
  }

public:
  //! @brief Constructs a \c basic_any_resource from a type that satisfies the \c resource or \c async_resource concept
  //! as well as all properties
  //! @param __res The resource to be wrapped within the \c basic_any_resource
  _LIBCUDACXX_TEMPLATE(class _Resource, _CUDA_VMR::_AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES(
    (!__is_basic_any_resource<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _CUDA_VMR::_AllocType::_Default)
      _LIBCUDACXX_AND _CUDA_VMR::resource_with<_CUDA_VSTD::remove_cvref_t<_Resource>, _Properties...>)
  basic_any_resource(_Resource&& __res) noexcept
      : basic_any_resource(_CUDA_VSTD::in_place, _CUDA_VSTD::forward<_Resource>(__res))
  {}

  //! @brief Constructs a \c basic_any_resource from a type that satisfies the \c async_resource concept as well as all
  //! properties. This ignores the async interface of the passed in resource
  //! @param __res The resource to be wrapped within the \c basic_any_resource
  _LIBCUDACXX_TEMPLATE(class _Resource, _CUDA_VMR::_AllocType _Alloc_type2 = _Alloc_type)
  _LIBCUDACXX_REQUIRES(
    (!__is_basic_any_resource<_Resource>) _LIBCUDACXX_AND(_Alloc_type2 == _CUDA_VMR::_AllocType::_Async)
      _LIBCUDACXX_AND _CUDA_VMR::async_resource_with<_CUDA_VSTD::remove_cvref_t<_Resource>, _Properties...>)
  basic_any_resource(_Resource&& __res) noexcept
      : basic_any_resource(_CUDA_VSTD::in_place, _CUDA_VSTD::forward<_Resource>(__res))
  {}

  //! @brief Conversion from a \c basic_any_resource with the same set of properties but in a different order
  //! @param __ref The other \c basic_any_resource
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES(__properties_match<_OtherProperties...>)
  basic_any_resource(basic_any_resource<_Alloc_type, _OtherProperties...> __other) noexcept
      : _CUDA_VMR::_Resource_base<_Alloc_type, _CUDA_VMR::_WrapperType::_Owning>(
          nullptr, _CUDA_VSTD::exchange(__other.__static_vtable, nullptr))
      , __vtable(__other)
  {
    _LIBCUDACXX_ASSERT(this->__static_vtable != nullptr, "copying from a moved-from object");
    this->__static_vtable->__move_fn(&this->__object, &__other.__object);
  }

  //! @brief Conversion from a \c async_any_resource with the same set of properties but in a different order to a
  //! \c any_resource
  //! @param __ref The other \c async_any_resource
  _LIBCUDACXX_TEMPLATE(_CUDA_VMR::_AllocType _OtherAllocType, class... _OtherProperties)
  _LIBCUDACXX_REQUIRES((_OtherAllocType == _CUDA_VMR::_AllocType::_Async) _LIBCUDACXX_AND(
    _OtherAllocType != _Alloc_type) _LIBCUDACXX_AND __properties_match<_OtherProperties...>)
  basic_any_resource(basic_any_resource<_OtherAllocType, _OtherProperties...> __other) noexcept
      : _CUDA_VMR::_Resource_base<_Alloc_type, _CUDA_VMR::_WrapperType::_Owning>(
          nullptr, _CUDA_VSTD::exchange(__other.__static_vtable, nullptr))
      , __vtable(__other)
  {
    _LIBCUDACXX_ASSERT(this->__static_vtable != nullptr, "copying from a moved-from object");
    this->__static_vtable->__move_fn(&this->__object, &__other.__object);
  }

  basic_any_resource(basic_any_resource&& __other) noexcept
      : _CUDA_VMR::_Resource_base<_Alloc_type, _CUDA_VMR::_WrapperType::_Owning>(
          nullptr, _CUDA_VSTD::exchange(__other.__static_vtable, nullptr))
      , __vtable(__other)
  {
    _LIBCUDACXX_ASSERT(this->__static_vtable != nullptr, "copying from a moved-from object");
    this->__static_vtable->__move_fn(&this->__object, &__other.__object);
  }

  basic_any_resource& operator=(basic_any_resource&& __other) noexcept
  {
    if (this->__static_vtable != nullptr)
    {
      this->__static_vtable->__destroy_fn(&this->__object);
      this->__static_vtable = nullptr;
    }

    if (__other.__static_vtable != nullptr)
    {
      this->__static_vtable = _CUDA_VSTD::exchange(__other.__static_vtable, nullptr);
      this->__static_vtable->__move_fn(&this->__object, &__other.__object);
    }

    return *this;
  }

  basic_any_resource(const basic_any_resource& __other)
      : _CUDA_VMR::_Resource_base<_Alloc_type, _CUDA_VMR::_WrapperType::_Owning>(nullptr, __other.__static_vtable)
      , __vtable(__other)
  {
    _LIBCUDACXX_ASSERT(this->__static_vtable != nullptr, "copying from a moved-from object");
    this->__static_vtable->__copy_fn(&this->__object, &__other.__object);
  }

  basic_any_resource& operator=(const basic_any_resource& __other)
  {
    return this == &__other ? *this : operator=(basic_any_resource(__other));
  }

  //! @brief Destroys the stored resource
  ~basic_any_resource() noexcept
  {
    if (this->__static_vtable != nullptr)
    {
      this->__static_vtable->__destroy_fn(&this->__object);
    }
  }

  _LIBCUDACXX_TEMPLATE(_CUDA_VMR::_AllocType _OtherAllocType, class... _OtherProperties)
  _LIBCUDACXX_REQUIRES(
    (_OtherAllocType == _CUDA_VMR::_AllocType::_Default || _OtherAllocType == _Alloc_type)
      _LIBCUDACXX_AND _CUDA_VSTD::__type_set_contains<_CUDA_VSTD::__type_set<_Properties...>, _OtherProperties...>)
  operator _CUDA_VMR::basic_resource_ref<_OtherAllocType, _OtherProperties...>() noexcept
  {
    return _CUDA_VMR::_Resource_ref_helper::_Construct<_Alloc_type, _OtherProperties...>(
      this->_Get_object(), this->__static_vtable, static_cast<const __vtable&>(*this));
  }

  void swap(basic_any_resource& __other) noexcept
  {
    auto __tmp = _CUDA_VSTD::move(__other);
    __other    = _CUDA_VSTD::move(*this);
    *this      = _CUDA_VSTD::move(__tmp);
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
        && this->__static_vtable->__equal_fn(this->_Get_object(), __rhs._Get_object());
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
    _CUDA_VMR::_Property_vtable<_Property> const& __prop = __res;
    return __prop.__property_fn(__res._Get_object());
  }
};

//! @brief Checks whether a passed in type is a specialization of basic_any_resource
template <class _Ty, _CUDA_VMR::_AllocType _Alloc_type, class... _Properties>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_basic_any_resource<_Ty, basic_any_resource<_Alloc_type, _Properties...>> =
  true;

//! @brief Type erased wrapper around a `resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any resource wrapped within the `any_resource` needs to satisfy
template <class... _Properties>
using any_resource = basic_any_resource<_CUDA_VMR::_AllocType::_Default, _Properties...>;

//! @brief Type erased wrapper around a `async_resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any async resource wrapped within the `async_any_resource` needs to satisfy
template <class... _Properties>
using async_any_resource = basic_any_resource<_CUDA_VMR::_AllocType::_Async, _Properties...>;

} // namespace cuda::experimental::mr

#endif //_CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
