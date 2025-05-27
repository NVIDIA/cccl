//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_REF_H
#define __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_REF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/basic_any_base.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/conversions.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/iset.cuh>
#include <cuda/experimental/__utility/basic_any/rtti.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//!
//! \c __ireference<_Interface>
//!
//! For an interface `I`, `__ireference<I>` represents a reference to an object
//! that satisfies the given interface. When you dereference a `basic_any<I*>`,
//! you get a `basic_any<__ireference<I>>`. Also, `basic_any<I&>` is implemented
//! in terms of `basic_any<__ireference<I>>`.
//!
//! Note: a `basic_any<__ireference<_Interface>> &&` is an rvalue reference,
//! whereas a `basic_any<_Interface &> &&` is an lvalue reference.
template <class _Interface>
struct __ireference : _Interface
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expected a class type");
  static constexpr size_t __size_      = sizeof(void*);
  static constexpr size_t __align_     = alignof(void*);
  static constexpr bool __is_const_ref = _CUDA_VSTD::is_const_v<_Interface>;

  using interface _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_Interface>;
};

#if defined(_CCCL_NO_CONCEPTS)
//!
//! A base class for basic_any<__ireference<_Interface>> that provides a
//! conversion to basic_any<__ireference<_Interface const>>. Only used
//! when concepts are not available.
//!
template <class _Interface>
struct __basic_any_reference_conversion_base
{
  [[nodiscard]] _CCCL_HOST_API operator basic_any<__ireference<_Interface const>>() const noexcept
  {
    return basic_any<__ireference<_Interface const>>(static_cast<basic_any<__ireference<_Interface>> const&>(*this));
  }
};

template <class _Interface>
struct __basic_any_reference_conversion_base<_Interface const>
{};
#endif // _CCCL_NO_CONCEPTS

_CCCL_DIAG_PUSH
// "operator basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_DIAG_SUPPRESS_NVHPC(conversion_function_not_usable)
// "operator basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_NV_DIAG_SUPPRESS(554)

//!
//! \c basic_any<__ireference<_Interface>>
//!
//! A `basic_any<__ireference<_Interface>>` is a reference to an object that
//! satisfies the given interface. It is used as the result of dereferencing
//! a `basic_any<_Interface*>` and as the implementation of
//! `basic_any<_Interface&>`.
//!
//! `basic_any<__ireference<_Interface>>` is neither copyable nor movable. It is
//! not an end-user type.
template <class _Interface>
struct _CCCL_DECLSPEC_EMPTY_BASES basic_any<__ireference<_Interface>>
    : __interface_of<__ireference<_Interface>>
#if defined(_CCCL_NO_CONCEPTS)
    , __basic_any_reference_conversion_base<_Interface>
#endif // _CCCL_NO_CONCEPTS
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expecting a class type");
  using interface_type                 = _CUDA_VSTD::remove_const_t<_Interface>;
  static constexpr bool __is_const_ref = _CUDA_VSTD::is_const_v<_Interface>;

  basic_any(basic_any&&)      = delete;
  basic_any(basic_any const&) = delete;

  auto operator=(basic_any&&) -> basic_any&      = delete;
  auto operator=(basic_any const&) -> basic_any& = delete;

#if !defined(_CCCL_NO_CONCEPTS)
  //! \brief A non-const basic_any reference can be implicitly converted to a
  //! const basic_any reference.
  [[nodiscard]] _CCCL_HOST_API operator basic_any<__ireference<_Interface const>>() const noexcept
    requires(!__is_const_ref)
  {
    return basic_any<__ireference<_Interface const>>(*this);
  }
#endif // !_CCCL_NO_CONCEPTS

  //! \brief Returns a const reference to the type_info for the decayed type
  //! of the type-erased object.
  [[nodiscard]] _CCCL_HOST_API auto type() const noexcept -> _CUDA_VSTD::__type_info_ref
  {
    return *__get_rtti()->__object_info_->__object_typeid_;
  }

  //! \brief Returns a const reference to the type_info for the decayed type
  //! of the type-erased object.
  [[nodiscard]] _CCCL_HOST_API auto interface() const noexcept -> _CUDA_VSTD::__type_info_ref
  {
    return *__get_rtti()->__interface_typeid_;
  }

  //! \brief Returns a reference to a type_info object representing the type of
  //! the dynamic interface.
  //!
  //! The dynamic interface is the interface that was used to construct the
  //! object, which may be different from the current object's interface if
  //! there was a conversion.
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API static constexpr auto has_value() noexcept -> bool
  {
    return true;
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

  basic_any() = default;

  //! \brief Constructs a \c basic_any<__ireference<_Interface>> from an lvalue
  //! reference to a \c basic_any<_Interface>.
  _CCCL_HOST_API explicit basic_any(
    _CUDA_VSTD::__maybe_const<__is_const_ref, basic_any<interface_type>>& __other) noexcept
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  //! \brief Constructs a \c basic_any<__ireference<_Interface>> from a
  //! vtable pointer and an object pointer.
  _CCCL_HOST_API basic_any(__vptr_for<interface_type> __vptr,
                           _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __optr) noexcept
      : __vptr_(__vptr)
      , __optr_(__optr)
  {}

  //! \brief No-op.
  _CCCL_TRIVIAL_HOST_API void reset() noexcept {}

  //! \brief No-op.
  _CCCL_TRIVIAL_HOST_API void __release() noexcept {}

  //! \brief Rebinds the reference with a vtable pointer and object pointer.
  _CCCL_HOST_API void __set_ref(__vptr_for<interface_type> __vptr,
                                _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    __vptr_ = __vptr;
    __optr_ = __vptr_ ? __obj : nullptr;
  }

  //! \brief Rebinds the reference with a vtable pointer for a different
  //! interface and object pointer. The vtable pointer is cast to the correct
  //! type. If the cast fails, the reference is set to null.
  template <class _VTable>
  _CCCL_HOST_API void __set_ref(_VTable const* __other, _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = typename _VTable::interface;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  //! \brief Rebinds the reference with a pointer to an iset vtable, which
  //! is cast to a pointer to the vtable for `_Interface`. If \c _Interface
  //! is a specialization `__iset<Is...>`, the cast succeeds if the
  //! \c Is... is a subset of `Interfaces...`. Otherwise, the cast succeeds
  //! if the \c _Interface is a base of one of `Interfaces...`.
  template <class... _Interfaces>
  _CCCL_HOST_API void
  __set_ref(__iset_vptr<_Interfaces...> __other, _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = __iset<_Interfaces...>;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  template <class _SrcCvAny>
  _CCCL_HOST_API void __convert_from(_SrcCvAny&& __from)
  {
    using __src_interface_t _CCCL_NODEBUG_ALIAS = typename _CUDA_VSTD::remove_reference_t<_SrcCvAny>::interface_type;
    if (!__from.has_value())
    {
      __throw_bad_any_cast();
    }
    auto __to_vptr = __vptr_cast<__src_interface_t, interface_type>(__from.__get_vptr());
    __set_ref(__to_vptr, __from.__get_optr());
  }

  _CCCL_TRIVIAL_HOST_API auto __get_optr() const noexcept -> _CUDA_VSTD::__maybe_const<__is_const_ref, void>*
  {
    return __optr_;
  }

  _CCCL_TRIVIAL_HOST_API auto __get_vptr() const noexcept -> __vptr_for<interface_type>
  {
    return __vptr_;
  }

  _CCCL_TRIVIAL_HOST_API auto __get_rtti() const noexcept -> __rtti const*
  {
    return __vptr_->__query_interface(iunknown());
  }

  __vptr_for<interface_type> __vptr_{};
  _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __optr_{};
};

_CCCL_NV_DIAG_DEFAULT(554)
_CCCL_DIAG_POP

//!
//! basic_any<_Interface&>
//!
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any<_Interface&> : basic_any<__ireference<_Interface>>
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expecting a class type");
  using typename basic_any<__ireference<_Interface>>::interface_type;
  using basic_any<__ireference<_Interface>>::__is_const_ref;

  _CCCL_TRIVIAL_HOST_API basic_any(basic_any&& __other) noexcept
      : basic_any(const_cast<basic_any const&>(__other))
  {}

  _CCCL_TRIVIAL_HOST_API basic_any(basic_any& __other) noexcept
      : basic_any(const_cast<basic_any const&>(__other))
  {}

  _CCCL_HOST_API basic_any(basic_any const& __other) noexcept
      : basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  _CCCL_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _CCCL_REQUIRES((!__is_basic_any<_Up>) _CCCL_AND(__is_const_ref || !_CUDA_VSTD::is_const_v<_Tp>)
                   _CCCL_AND __satisfies<_Up, interface_type>)
  _CCCL_HOST_API basic_any(_Tp& __obj) noexcept
      : basic_any<__ireference<_Interface>>()
  {
    __vptr_for<interface_type> const __vptr = experimental::__get_vtable_ptr_for<interface_type, _Up>();
    this->__set_ref(__vptr, &__obj);
  }

  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>) _CCCL_AND //
                 (!__is_value_v<_SrcInterface>) _CCCL_AND //
                   __any_convertible_to<basic_any<_SrcInterface>, basic_any>)
  _CCCL_HOST_API basic_any(basic_any<_SrcInterface>&& __src) noexcept
      : basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>) _CCCL_AND //
                   __any_convertible_to<basic_any<_SrcInterface>&, basic_any>)
  _CCCL_HOST_API basic_any(basic_any<_SrcInterface>& __src) noexcept
      : basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>) _CCCL_AND //
                   __any_convertible_to<basic_any<_SrcInterface> const&, basic_any>)
  _CCCL_HOST_API basic_any(basic_any<_SrcInterface> const& __src) noexcept
      : basic_any<__ireference<_Interface>>()
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  auto operator=(basic_any&&) -> basic_any&      = delete;
  auto operator=(basic_any const&) -> basic_any& = delete;

  _CCCL_TRIVIAL_HOST_API auto move() & noexcept -> basic_any<__ireference<_Interface>>&&
  {
    return _CUDA_VSTD::move(*this);
  }

private:
  template <class>
  friend struct basic_any;
  friend struct __basic_any_access;

  basic_any() = default;
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_REF_H
