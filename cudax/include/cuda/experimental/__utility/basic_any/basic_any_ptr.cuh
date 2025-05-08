//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_PTR_H
#define __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/basic_any_base.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_from.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_ref.cuh>
#include <cuda/experimental/__utility/basic_any/conversions.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/rtti.cuh>
#include <cuda/experimental/__utility/basic_any/virtual_tables.cuh>

_CCCL_PUSH_MACROS
#undef interface

namespace cuda::experimental
{
//!
//! basic_any<_Interface*>
//!
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any<_Interface*>
{
  using interface_type                 = _CUDA_VSTD::remove_const_t<_Interface>;
  static constexpr bool __is_const_ptr = _CUDA_VSTD::is_const_v<_Interface>;

  //!
  //! Constructors
  //!
  basic_any() = default;

  _CCCL_TRIVIAL_HOST_API basic_any(_CUDA_VSTD::nullptr_t) {}

  _CCCL_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _CCCL_REQUIRES((!__is_basic_any<_Tp>) _CCCL_AND __satisfies<_Up, interface_type> _CCCL_AND(
    __is_const_ptr || !_CUDA_VSTD::is_const_v<_Tp>))
  _CCCL_HOST_API basic_any(_Tp* __obj) noexcept
  {
    operator=(__obj);
  }

  _CCCL_HOST_API basic_any(basic_any const& __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<basic_any<_OtherInterface*>, basic_any<_Interface*>>)
  _CCCL_HOST_API basic_any(basic_any<_OtherInterface*> const& __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<basic_any<_OtherInterface>&, basic_any<_Interface&>>)
  _CCCL_HOST_API basic_any(basic_any<_OtherInterface>* __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<basic_any<_OtherInterface> const&, basic_any<_Interface&>>)
  _CCCL_HOST_API basic_any(basic_any<_OtherInterface> const* __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _CCCL_REQUIRES(__is_interface<_OtherInterface<_Super>> _CCCL_AND
                   _CUDA_VSTD::derived_from<basic_any<_Super>, _OtherInterface<_Super>> _CCCL_AND
                     _CUDA_VSTD::same_as<__normalized_interface_of<basic_any<_Super>*>, _Interface*>)
  _CCCL_HOST_API explicit basic_any(_OtherInterface<_Super>* __self) noexcept
  {
    __convert_from(basic_any_from(__self));
  }

  _CCCL_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _CCCL_REQUIRES(__is_interface<_OtherInterface<_Super>> _CCCL_AND
                   _CUDA_VSTD::derived_from<basic_any<_Super>, _OtherInterface<_Super>> _CCCL_AND
                     _CUDA_VSTD::same_as<__normalized_interface_of<basic_any<_Super> const*>, _Interface*>)
  _CCCL_HOST_API explicit basic_any(_OtherInterface<_Super> const* __self) noexcept
  {
    __convert_from(basic_any_from(__self));
  }

  //!
  //! Assignment operators
  //!
  _CCCL_HOST_API auto operator=(_CUDA_VSTD::nullptr_t) noexcept -> basic_any&
  {
    reset();
    return *this;
  }

  _CCCL_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _CCCL_REQUIRES((!__is_basic_any<_Tp>) _CCCL_AND //
                   __satisfies<_Up, interface_type> _CCCL_AND //
                 (__is_const_ptr || !_CUDA_VSTD::is_const_v<_Tp>))
  _CCCL_HOST_API auto operator=(_Tp* __obj) noexcept -> basic_any&
  {
    __vptr_for<interface_type> __vptr = experimental::__get_vtable_ptr_for<interface_type, _Up>();
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *this;
  }

  _CCCL_HOST_API auto operator=(basic_any const& __other) noexcept -> basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<basic_any<_OtherInterface*>, basic_any<_Interface*>>)
  _CCCL_HOST_API auto operator=(basic_any<_OtherInterface*> const& __other) noexcept -> basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<basic_any<_OtherInterface>&, basic_any<_Interface&>>)
  _CCCL_HOST_API auto operator=(basic_any<_OtherInterface>* __other) noexcept -> basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<basic_any<_OtherInterface> const&, basic_any<_Interface&>>)
  _CCCL_HOST_API auto operator=(basic_any<_OtherInterface> const* __other) noexcept -> basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  //!
  //! emplace
  //!
  _CCCL_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_pointer_t<_Tp>, class _Vp = _CUDA_VSTD::remove_const_t<_Up>)
  _CCCL_REQUIRES(__satisfies<_Vp, _Interface> _CCCL_AND(__is_const_ptr || !_CUDA_VSTD::is_const_v<_Up>))
  _CCCL_HOST_API auto emplace(_CUDA_VSTD::type_identity_t<_Up>* __obj) noexcept
    -> _CUDA_VSTD::__maybe_const<__is_const_ptr, _Vp>*&
  {
    __vptr_for<interface_type> __vptr = experimental::__get_vtable_ptr_for<interface_type, _Vp>();
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *static_cast<_CUDA_VSTD::__maybe_const<__is_const_ptr, _Vp>**>(static_cast<void*>(&__ref_.__optr_));
  }

#if !defined(_CCCL_NO_THREE_WAY_COMPARISON)
  [[nodiscard]] _CCCL_HOST_API auto operator==(basic_any const& __other) const noexcept -> bool
  {
    using __void_ptr_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__get_optr()) == *static_cast<__void_ptr_t>(__other.__get_optr());
  }
#else // ^^^ !_CCCL_NO_THREE_WAY_COMPARISON ^^^ / vvv _CCCL_NO_THREE_WAY_COMPARISON vvv
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_API auto operator==(basic_any const& __lhs, basic_any const& __rhs) noexcept -> bool
  {
    using __void_ptr_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__lhs.__get_optr()) == *static_cast<__void_ptr_t>(__rhs.__get_optr());
  }

  _CCCL_NODISCARD_FRIEND _CCCL_TRIVIAL_HOST_API auto operator!=(basic_any const& __lhs, basic_any const& __rhs) noexcept
    -> bool
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_NO_THREE_WAY_COMPARISON

  using __any_ref_t _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::__maybe_const<__is_const_ptr, basic_any<__ireference<_Interface>>>;

  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto operator->() const noexcept -> __any_ref_t*
  {
    return &__ref_;
  }

  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto operator*() const noexcept -> __any_ref_t&
  {
    return __ref_;
  }

  [[nodiscard]] _CCCL_HOST_API auto type() const noexcept -> _CUDA_VSTD::__type_info_ref
  {
    return __ref_.__vptr_ != nullptr
           ? (__is_const_ptr ? *__get_rtti()->__object_info_->__const_pointer_typeid_
                             : *__get_rtti()->__object_info_->__pointer_typeid_)
           : _CCCL_TYPEID(void);
  }

  [[nodiscard]] _CCCL_HOST_API auto interface() const noexcept -> _CUDA_VSTD::__type_info_ref
  {
    return __ref_.__vptr_ != nullptr ? *__get_rtti()->__interface_typeid_ : _CCCL_TYPEID(interface_type);
  }

  [[nodiscard]] _CCCL_HOST_API auto has_value() const noexcept -> bool
  {
    return __ref_.__vptr_ != nullptr;
  }

  _CCCL_HOST_API void reset() noexcept
  {
    __vptr_for<interface_type> __vptr = nullptr;
    __ref_.__set_ref(__vptr, nullptr);
  }

  [[nodiscard]] _CCCL_HOST_API explicit operator bool() const noexcept
  {
    return __ref_.__vptr_ != nullptr;
  }

#if !defined(_CCCL_DOXYGEN_INVOKED) // Do not document
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API static constexpr auto __in_situ() noexcept -> bool
  {
    return true;
  }
#endif // _CCCL_DOXYGEN_INVOKED

private:
  template <class>
  friend struct basic_any;
  friend struct __basic_any_access;

  template <class _SrcCvAny>
  _CCCL_HOST_API void __convert_from(_SrcCvAny* __other) noexcept
  {
    __other ? __ref_.__set_ref(__other->__get_vptr(), __other->__get_optr()) : reset();
  }

  template <class _OtherInterface>
  _CCCL_HOST_API void __convert_from(basic_any<_OtherInterface*> const& __other) noexcept
  {
    using __other_interface_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_OtherInterface>;
    auto __to_vptr = __try_vptr_cast<__other_interface_t, interface_type>(__other.__get_vptr());
    auto __to_optr = __to_vptr ? *__other.__get_optr() : nullptr;
    __ref_.__set_ref(__to_vptr, __to_optr);
  }

  [[nodiscard]] _CCCL_HOST_API auto __get_optr() noexcept -> _CUDA_VSTD::__maybe_const<__is_const_ptr, void>**
  {
    return &__ref_.__optr_;
  }

  [[nodiscard]] _CCCL_HOST_API auto __get_optr() const noexcept
    -> _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const*
  {
    return &__ref_.__optr_;
  }

  [[nodiscard]] _CCCL_HOST_API auto __get_vptr() const noexcept -> __vptr_for<interface_type>
  {
    return __ref_.__vptr_;
  }

  [[nodiscard]] _CCCL_HOST_API auto __get_rtti() const noexcept -> __rtti const*
  {
    return __ref_.__vptr_ ? __ref_.__vptr_->__query_interface(iunknown()) : nullptr;
  }

  mutable basic_any<__ireference<_Interface>> __ref_;
};

_CCCL_TEMPLATE(template <class...> class _Interface, class _Super)
_CCCL_REQUIRES(__is_interface<_Interface<_Super>>)
_CUDAX_PUBLIC_API basic_any(_Interface<_Super>*) //
  -> basic_any<__normalized_interface_of<basic_any<_Super>*>>;

_CCCL_TEMPLATE(template <class...> class _Interface, class _Super)
_CCCL_REQUIRES(__is_interface<_Interface<_Super>>)
_CUDAX_PUBLIC_API basic_any(_Interface<_Super> const*) //
  -> basic_any<__normalized_interface_of<basic_any<_Super> const*>>;

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_PTR_H
