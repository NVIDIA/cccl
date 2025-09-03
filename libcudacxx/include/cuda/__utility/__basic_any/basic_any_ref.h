//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_REF_H
#define _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_REF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_base.h>
#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/conversions.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/__utility/__basic_any/iset.h>
#include <cuda/__utility/__basic_any/rtti.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! \c __ireference<_Interface>
//!
//! For an interface `I`, `__ireference<I>` represents a reference to an object
//! that satisfies the given interface. When you dereference a `__basic_any<I*>`,
//! you get a `__basic_any<__ireference<I>>`. Also, `__basic_any<I&>` is implemented
//! in terms of `__basic_any<__ireference<I>>`.
//!
//! Note: a `__basic_any<__ireference<_Interface>> &&` is an rvalue reference,
//! whereas a `__basic_any<_Interface &> &&` is an lvalue reference.
template <class _Interface>
struct __ireference : _Interface
{
  static_assert(::cuda::std::is_class_v<_Interface>, "expected a class type");
  static constexpr size_t __size_      = sizeof(void*);
  static constexpr size_t __align_     = alignof(void*);
  static constexpr bool __is_const_ref = ::cuda::std::is_const_v<_Interface>;

  using interface _CCCL_NODEBUG_ALIAS = ::cuda::std::remove_const_t<_Interface>;
};

#if !_CCCL_HAS_CONCEPTS()
//!
//! A base class for __basic_any<__ireference<_Interface>> that provides a
//! conversion to __basic_any<__ireference<_Interface const>>. Only used
//! when concepts are not available.
//!
template <class _Interface>
struct __basic_any_reference_conversion_base
{
  [[nodiscard]] _CCCL_API operator __basic_any<__ireference<_Interface const>>() const noexcept
  {
    return __basic_any<__ireference<_Interface const>>(
      static_cast<__basic_any<__ireference<_Interface>> const&>(*this));
  }
};

template <class _Interface>
struct __basic_any_reference_conversion_base<_Interface const>
{};
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_DIAG_PUSH
// "operator __basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_DIAG_SUPPRESS_NVHPC(conversion_function_not_usable)
// "operator __basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_BEGIN_NV_DIAG_SUPPRESS(554)

//!
//! \c __basic_any<__ireference<_Interface>>
//!
//! A `__basic_any<__ireference<_Interface>>` is a reference to an object that
//! satisfies the given interface. It is used as the result of dereferencing
//! a `__basic_any<_Interface*>` and as the implementation of
//! `__basic_any<_Interface&>`.
//!
//! `__basic_any<__ireference<_Interface>>` is neither copyable nor movable. It is
//! not an end-user type.
template <class _Interface>
struct _CCCL_DECLSPEC_EMPTY_BASES __basic_any<__ireference<_Interface>>
    : __interface_of<__ireference<_Interface>>
#if !_CCCL_HAS_CONCEPTS()
    , __basic_any_reference_conversion_base<_Interface>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
{
  static_assert(::cuda::std::is_class_v<_Interface>, "expecting a class type");
  using interface_type                 = ::cuda::std::remove_const_t<_Interface>;
  static constexpr bool __is_const_ref = ::cuda::std::is_const_v<_Interface>;

  __basic_any(__basic_any&&)      = delete;
  __basic_any(__basic_any const&) = delete;

  auto operator=(__basic_any&&) -> __basic_any&      = delete;
  auto operator=(__basic_any const&) -> __basic_any& = delete;

#if _CCCL_HAS_CONCEPTS()
  //! \brief A non-const __basic_any reference can be implicitly converted to a
  //! const __basic_any reference.
  [[nodiscard]] _CCCL_API operator __basic_any<__ireference<_Interface const>>() const noexcept
    requires(!__is_const_ref)
  {
    return __basic_any<__ireference<_Interface const>>(*this);
  }
#endif // _CCCL_HAS_CONCEPTS()

  //! \brief Returns a const reference to the type_info for the decayed type
  //! of the type-erased object.
  [[nodiscard]] _CCCL_API auto type() const noexcept -> ::cuda::std::__type_info_ref
  {
    return *__get_rtti()->__object_info_->__object_typeid_;
  }

  //! \brief Returns a const reference to the type_info for the decayed type
  //! of the type-erased object.
  [[nodiscard]] _CCCL_API auto interface() const noexcept -> ::cuda::std::__type_info_ref
  {
    return *__get_rtti()->__interface_typeid_;
  }

  //! \brief Returns a reference to a type_info object representing the type of
  //! the dynamic interface.
  //!
  //! The dynamic interface is the interface that was used to construct the
  //! object, which may be different from the current object's interface if
  //! there was a conversion.
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto has_value() noexcept -> bool
  {
    return true;
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

  __basic_any() = default;

  //! \brief Constructs a \c __basic_any<__ireference<_Interface>> from an lvalue
  //! reference to a \c __basic_any<_Interface>.
  _CCCL_API explicit __basic_any(
    ::cuda::std::__maybe_const<__is_const_ref, __basic_any<interface_type>>& __other) noexcept
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  //! \brief Constructs a \c __basic_any<__ireference<_Interface>> from a
  //! vtable pointer and an object pointer.
  _CCCL_API __basic_any(__vptr_for<interface_type> __vptr,
                        ::cuda::std::__maybe_const<__is_const_ref, void>* __optr) noexcept
      : __vptr_(__vptr)
      , __optr_(__optr)
  {}

  //! \brief No-op.
  _CCCL_NODEBUG_API void reset() noexcept {}

  //! \brief No-op.
  _CCCL_NODEBUG_API void __release_() noexcept {}

  //! \brief Rebinds the reference with a vtable pointer and object pointer.
  _CCCL_API void __set_ref(__vptr_for<interface_type> __vptr,
                           ::cuda::std::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    __vptr_ = __vptr;
    __optr_ = __vptr_ ? __obj : nullptr;
  }

  //! \brief Rebinds the reference with a vtable pointer for a different
  //! interface and object pointer. The vtable pointer is cast to the correct
  //! type. If the cast fails, the reference is set to null.
  template <class _VTable>
  _CCCL_API void __set_ref(_VTable const* __other, ::cuda::std::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = typename _VTable::interface;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  //! \brief Rebinds the reference with a pointer to an __iset vtable, which
  //! is cast to a pointer to the vtable for `_Interface`. If \c _Interface
  //! is a specialization `__iset_<Is...>`, the cast succeeds if the
  //! \c Is... is a subset of `Interfaces...`. Otherwise, the cast succeeds
  //! if the \c _Interface is a base of one of `Interfaces...`.
  template <class... _Interfaces>
  _CCCL_API void __set_ref(__iset_vptr<_Interfaces...> __other,
                           ::cuda::std::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = __iset_<_Interfaces...>;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  template <class _SrcCvAny>
  _CCCL_API void __convert_from(_SrcCvAny&& __from)
  {
    using __src_interface_t _CCCL_NODEBUG_ALIAS = typename ::cuda::std::remove_reference_t<_SrcCvAny>::interface_type;
    if (!__from.has_value())
    {
      __throw_bad_any_cast();
    }
    auto __to_vptr = __vptr_cast<__src_interface_t, interface_type>(__from.__get_vptr());
    __set_ref(__to_vptr, __from.__get_optr());
  }

  _CCCL_NODEBUG_API auto __get_optr() const noexcept -> ::cuda::std::__maybe_const<__is_const_ref, void>*
  {
    return __optr_;
  }

  _CCCL_NODEBUG_API auto __get_vptr() const noexcept -> __vptr_for<interface_type>
  {
    return __vptr_;
  }

  _CCCL_NODEBUG_API auto __get_rtti() const noexcept -> __rtti const*
  {
    return __vptr_->__query_interface(__iunknown());
  }

  __vptr_for<interface_type> __vptr_{};
  ::cuda::std::__maybe_const<__is_const_ref, void>* __optr_{};
};

_CCCL_END_NV_DIAG_SUPPRESS()
_CCCL_DIAG_POP

//!
//! __basic_any<_Interface&>
//!
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __basic_any<_Interface&> : __basic_any<__ireference<_Interface>>
{
  static_assert(::cuda::std::is_class_v<_Interface>, "expecting a class type");
  using typename __basic_any<__ireference<_Interface>>::interface_type;
  using __basic_any<__ireference<_Interface>>::__is_const_ref;

  _CCCL_NODEBUG_API __basic_any(__basic_any&& __other) noexcept
      : __basic_any(const_cast<__basic_any const&>(__other))
  {}

  _CCCL_NODEBUG_API __basic_any(__basic_any& __other) noexcept
      : __basic_any(const_cast<__basic_any const&>(__other))
  {}

  _CCCL_API __basic_any(__basic_any const& __other) noexcept
      : __basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  _CCCL_TEMPLATE(class _Tp, class _Up = ::cuda::std::remove_const_t<_Tp>)
  _CCCL_REQUIRES((!__is_basic_any<_Up>) _CCCL_AND(__is_const_ref || !::cuda::std::is_const_v<_Tp>)
                   _CCCL_AND __satisfies<_Up, interface_type>)
  _CCCL_API __basic_any(_Tp& __obj) noexcept
      : __basic_any<__ireference<_Interface>>()
  {
    __vptr_for<interface_type> const __vptr = ::cuda::__get_vtable_ptr_for<interface_type, _Up>();
    this->__set_ref(__vptr, &__obj);
  }

  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_SrcInterface, _Interface&>) _CCCL_AND //
                 (!__is_value_v<_SrcInterface>) _CCCL_AND //
                   __any_convertible_to<__basic_any<_SrcInterface>, __basic_any>)
  _CCCL_API __basic_any(__basic_any<_SrcInterface>&& __src) noexcept
      : __basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_SrcInterface, _Interface&>) _CCCL_AND //
                   __any_convertible_to<__basic_any<_SrcInterface>&, __basic_any>)
  _CCCL_API __basic_any(__basic_any<_SrcInterface>& __src) noexcept
      : __basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_SrcInterface, _Interface&>) _CCCL_AND //
                   __any_convertible_to<__basic_any<_SrcInterface> const&, __basic_any>)
  _CCCL_API __basic_any(__basic_any<_SrcInterface> const& __src) noexcept
      : __basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  auto operator=(__basic_any&&) -> __basic_any&      = delete;
  auto operator=(__basic_any const&) -> __basic_any& = delete;

  _CCCL_NODEBUG_API auto move() & noexcept -> __basic_any<__ireference<_Interface>>&&
  {
    return ::cuda::std::move(*this);
  }

private:
  template <class>
  friend struct __basic_any;
  friend struct __basic_any_access;

  __basic_any() = default;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_REF_H
