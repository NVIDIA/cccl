//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_PTR_H
#define _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_base.h>
#include <cuda/__utility/__basic_any/basic_any_from.h>
#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/basic_any_ref.h>
#include <cuda/__utility/__basic_any/conversions.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/__utility/__basic_any/rtti.h>
#include <cuda/__utility/__basic_any/virtual_tables.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! __basic_any<_Interface*>
//!
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __basic_any<_Interface*>
{
  using interface_type                 = ::cuda::std::remove_const_t<_Interface>;
  static constexpr bool __is_const_ptr = ::cuda::std::is_const_v<_Interface>;

  //!
  //! Constructors
  //!
  __basic_any() = default;

  _CCCL_NODEBUG_API __basic_any(::cuda::std::nullptr_t) {}

  _CCCL_TEMPLATE(class _Tp, class _Up = ::cuda::std::remove_const_t<_Tp>)
  _CCCL_REQUIRES((!__is_basic_any<_Tp>) _CCCL_AND __satisfies<_Up, interface_type> _CCCL_AND(
    __is_const_ptr || !::cuda::std::is_const_v<_Tp>))
  _CCCL_API __basic_any(_Tp* __obj) noexcept
  {
    operator=(__obj);
  }

  _CCCL_API __basic_any(__basic_any const& __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<__basic_any<_OtherInterface*>, __basic_any<_Interface*>>)
  _CCCL_API __basic_any(__basic_any<_OtherInterface*> const& __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_OtherInterface>&, __basic_any<_Interface&>>)
  _CCCL_API __basic_any(__basic_any<_OtherInterface>* __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_OtherInterface> const&, __basic_any<_Interface&>>)
  _CCCL_API __basic_any(__basic_any<_OtherInterface> const* __other) noexcept
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _CCCL_REQUIRES(__is_interface<_OtherInterface<_Super>>
                   _CCCL_AND ::cuda::std::derived_from<__basic_any<_Super>, _OtherInterface<_Super>>
                     _CCCL_AND ::cuda::std::same_as<__normalized_interface_of<__basic_any<_Super>*>, _Interface*>)
  _CCCL_API explicit __basic_any(_OtherInterface<_Super>* __self) noexcept
  {
    __convert_from(__basic_any_from(__self));
  }

  _CCCL_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _CCCL_REQUIRES(__is_interface<_OtherInterface<_Super>>
                   _CCCL_AND ::cuda::std::derived_from<__basic_any<_Super>, _OtherInterface<_Super>>
                     _CCCL_AND ::cuda::std::same_as<__normalized_interface_of<__basic_any<_Super> const*>, _Interface*>)
  _CCCL_API explicit __basic_any(_OtherInterface<_Super> const* __self) noexcept
  {
    __convert_from(__basic_any_from(__self));
  }

  //!
  //! Assignment operators
  //!
  _CCCL_API auto operator=(::cuda::std::nullptr_t) noexcept -> __basic_any&
  {
    reset();
    return *this;
  }

  _CCCL_TEMPLATE(class _Tp, class _Up = ::cuda::std::remove_const_t<_Tp>)
  _CCCL_REQUIRES((!__is_basic_any<_Tp>) _CCCL_AND //
                   __satisfies<_Up, interface_type> _CCCL_AND //
                 (__is_const_ptr || !::cuda::std::is_const_v<_Tp>))
  _CCCL_API auto operator=(_Tp* __obj) noexcept -> __basic_any&
  {
    __vptr_for<interface_type> __vptr = ::cuda::__get_vtable_ptr_for<interface_type, _Up>();
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *this;
  }

  _CCCL_API auto operator=(__basic_any const& __other) noexcept -> __basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<__basic_any<_OtherInterface*>, __basic_any<_Interface*>>)
  _CCCL_API auto operator=(__basic_any<_OtherInterface*> const& __other) noexcept -> __basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_OtherInterface>&, __basic_any<_Interface&>>)
  _CCCL_API auto operator=(__basic_any<_OtherInterface>* __other) noexcept -> __basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_OtherInterface> const&, __basic_any<_Interface&>>)
  _CCCL_API auto operator=(__basic_any<_OtherInterface> const* __other) noexcept -> __basic_any&
  {
    __convert_from(__other);
    return *this;
  }

  //!
  //! emplace
  //!
  _CCCL_TEMPLATE(class _Tp, class _Up = ::cuda::std::remove_pointer_t<_Tp>, class _Vp = ::cuda::std::remove_const_t<_Up>)
  _CCCL_REQUIRES(__satisfies<_Vp, _Interface> _CCCL_AND(__is_const_ptr || !::cuda::std::is_const_v<_Up>))
  _CCCL_API auto emplace(::cuda::std::type_identity_t<_Up>* __obj) noexcept
    -> ::cuda::std::__maybe_const<__is_const_ptr, _Vp>*&
  {
    __vptr_for<interface_type> __vptr = ::cuda::__get_vtable_ptr_for<interface_type, _Vp>();
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *static_cast<::cuda::std::__maybe_const<__is_const_ptr, _Vp>**>(static_cast<void*>(&__ref_.__optr_));
  }

#if !defined(_CCCL_NO_THREE_WAY_COMPARISON)
  [[nodiscard]] _CCCL_API auto operator==(__basic_any const& __other) const noexcept -> bool
  {
    using __void_ptr_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__get_optr()) == *static_cast<__void_ptr_t>(__other.__get_optr());
  }
#else // ^^^ !_CCCL_NO_THREE_WAY_COMPARISON ^^^ / vvv _CCCL_NO_THREE_WAY_COMPARISON vvv
  [[nodiscard]] _CCCL_API friend auto operator==(__basic_any const& __lhs, __basic_any const& __rhs) noexcept -> bool
  {
    using __void_ptr_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__lhs.__get_optr()) == *static_cast<__void_ptr_t>(__rhs.__get_optr());
  }

  [[nodiscard]] _CCCL_NODEBUG_API friend auto operator!=(__basic_any const& __lhs, __basic_any const& __rhs) noexcept
    -> bool
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_NO_THREE_WAY_COMPARISON

  using __any_ref_t _CCCL_NODEBUG_ALIAS =
    ::cuda::std::__maybe_const<__is_const_ptr, __basic_any<__ireference<_Interface>>>;

  [[nodiscard]] _CCCL_NODEBUG_API auto operator->() const noexcept -> __any_ref_t*
  {
    return &__ref_;
  }

  [[nodiscard]] _CCCL_NODEBUG_API auto operator*() const noexcept -> __any_ref_t&
  {
    return __ref_;
  }

  [[nodiscard]] _CCCL_API auto type() const noexcept -> ::cuda::std::__type_info_ref
  {
    return __ref_.__vptr_ != nullptr
           ? (__is_const_ptr ? *__get_rtti()->__object_info_->__const_pointer_typeid_
                             : *__get_rtti()->__object_info_->__pointer_typeid_)
           : _CCCL_TYPEID(void);
  }

  [[nodiscard]] _CCCL_API auto interface() const noexcept -> ::cuda::std::__type_info_ref
  {
    return __ref_.__vptr_ != nullptr ? *__get_rtti()->__interface_typeid_ : _CCCL_TYPEID(interface_type);
  }

  [[nodiscard]] _CCCL_API auto has_value() const noexcept -> bool
  {
    return __ref_.__vptr_ != nullptr;
  }

  _CCCL_API void reset() noexcept
  {
    __vptr_for<interface_type> __vptr = nullptr;
    __ref_.__set_ref(__vptr, nullptr);
  }

  [[nodiscard]] _CCCL_API explicit operator bool() const noexcept
  {
    return __ref_.__vptr_ != nullptr;
  }

#if !defined(_CCCL_DOXYGEN_INVOKED) // Do not document
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto __in_situ() noexcept -> bool
  {
    return true;
  }
#endif // _CCCL_DOXYGEN_INVOKED

private:
  template <class>
  friend struct __basic_any;
  friend struct __basic_any_access;

  template <class _SrcCvAny>
  _CCCL_API void __convert_from(_SrcCvAny* __other) noexcept
  {
    __other ? __ref_.__set_ref(__other->__get_vptr(), __other->__get_optr()) : reset();
  }

  template <class _OtherInterface>
  _CCCL_API void __convert_from(__basic_any<_OtherInterface*> const& __other) noexcept
  {
    using __other_interface_t _CCCL_NODEBUG_ALIAS = ::cuda::std::remove_const_t<_OtherInterface>;
    auto __to_vptr = __try_vptr_cast<__other_interface_t, interface_type>(__other.__get_vptr());
    auto __to_optr = __to_vptr ? *__other.__get_optr() : nullptr;
    __ref_.__set_ref(__to_vptr, __to_optr);
  }

  [[nodiscard]] _CCCL_API auto __get_optr() noexcept -> ::cuda::std::__maybe_const<__is_const_ptr, void>**
  {
    return &__ref_.__optr_;
  }

  [[nodiscard]] _CCCL_API auto __get_optr() const noexcept -> ::cuda::std::__maybe_const<__is_const_ptr, void>* const*
  {
    return &__ref_.__optr_;
  }

  [[nodiscard]] _CCCL_API auto __get_vptr() const noexcept -> __vptr_for<interface_type>
  {
    return __ref_.__vptr_;
  }

  [[nodiscard]] _CCCL_API auto __get_rtti() const noexcept -> __rtti const*
  {
    return __ref_.__vptr_ ? __ref_.__vptr_->__query_interface(__iunknown()) : nullptr;
  }

  mutable __basic_any<__ireference<_Interface>> __ref_;
};

_CCCL_TEMPLATE(template <class...> class _Interface, class _Super)
_CCCL_REQUIRES(__is_interface<_Interface<_Super>>)
_CCCL_HOST_DEVICE __basic_any(_Interface<_Super>*) //
  -> __basic_any<__normalized_interface_of<__basic_any<_Super>*>>;

_CCCL_TEMPLATE(template <class...> class _Interface, class _Super)
_CCCL_REQUIRES(__is_interface<_Interface<_Super>>)
_CCCL_HOST_DEVICE __basic_any(_Interface<_Super> const*) //
  -> __basic_any<__normalized_interface_of<__basic_any<_Super> const*>>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_PTR_H
