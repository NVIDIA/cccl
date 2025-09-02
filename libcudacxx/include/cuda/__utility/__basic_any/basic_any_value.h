//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_VALUE_H
#define _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_VALUE_H

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
#include <cuda/__utility/__basic_any/basic_any_ref.h>
#include <cuda/__utility/__basic_any/conversions.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/__utility/__basic_any/rtti.h>
#include <cuda/__utility/__basic_any/semiregular.h>
#include <cuda/__utility/__basic_any/storage.h>
#include <cuda/__utility/__basic_any/virtual_tables.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// constructible_from using list initialization syntax.
// clang-format off
template <class _Tp, class... _Args>
_CCCL_CONCEPT __list_initializable_from =
  _CCCL_REQUIRES_EXPR((_Tp, variadic _Args), _Args&&... __args)
  (
    _Tp{static_cast<_Args&&>(__args)...}
  );
// clang-format on

//!
//! __basic_any
//!
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __basic_any : __basic_any_base<_Interface>
{
private:
  static_assert(::cuda::std::is_class_v<_Interface>,
                "__basic_any requires an interface type, or a pointer or reference to an interface "
                "type.");
  static_assert(!::cuda::std::is_const_v<_Interface>, "__basic_any does not support const-qualified interfaces.");

  using __basic_any_base<_Interface>::__size_;
  using __basic_any_base<_Interface>::__align_;
  using __basic_any_base<_Interface>::__vptr_;
  using __basic_any_base<_Interface>::__buffer_;

public:
  using interface_type = _Interface;

  //! \brief Constructs an empty `__basic_any` object.
  //! \post `has_value() == false`
  __basic_any() = default;

  //! \brief Constructs a `__basic_any` object that contains a copy of `__value`.
  //! \pre `__value` must be move constructible. `_Tp` must satisfy the
  //! requirements of `_Interface`.
  //! \post `has_value() == true`
  _CCCL_TEMPLATE(class _Tp, class _Up = ::cuda::std::decay_t<_Tp>)
  _CCCL_REQUIRES((!__is_basic_any<_Up>) _CCCL_AND __satisfies<_Up, _Interface>)
  _CCCL_API __basic_any(_Tp&& __value) noexcept(__is_small<_Up>(__size_, __align_))
  {
    __emplace<_Up>(static_cast<_Tp&&>(__value));
  }

  //! \brief Constructs a `__basic_any` object that contains a new object of type
  //! `_Tp` constructed as `_Tp{__args...}`.
  //! \pre `_Tp` must satisfy the requirements of `_Interface`.
  //! \post `has_value() == true`
  _CCCL_TEMPLATE(class _Tp, class _Up = ::cuda::std::decay_t<_Tp>, class... _Args)
  _CCCL_REQUIRES(__list_initializable_from<_Up, _Args...> _CCCL_AND __satisfies<_Tp, _Interface>)
  _CCCL_API explicit __basic_any(::cuda::std::in_place_type_t<_Tp>, _Args&&... __args) noexcept(
    __is_small<_Up>(__size_, __align_) && ::cuda::std::is_nothrow_constructible_v<_Up, _Args...>)
  {
    __emplace<_Up>(static_cast<_Args&&>(__args)...);
  }

  //! \brief Constructs a `__basic_any` object that contains a new object of type
  //! `_Tp` constructed as `_Tp{__il, __args...}`.
  //! \pre `_Tp` must satisfy the requirements of `_Interface`.
  //! \post `has_value() == true`
  _CCCL_TEMPLATE(class _Tp, class _Up, class _Vp = ::cuda::std::decay_t<_Tp>, class... _Args)
  _CCCL_REQUIRES(__list_initializable_from<_Vp, ::cuda::std::initializer_list<_Up>&, _Args...> _CCCL_AND
                   __satisfies<_Tp, _Interface>)
  _CCCL_API explicit __basic_any(
    ::cuda::std::in_place_type_t<_Tp>,
    ::cuda::std::initializer_list<_Up> __il,
    _Args&&... __args) noexcept(__is_small<_Vp>(__size_, __align_)
                                && ::cuda::std::
                                  is_nothrow_constructible_v<_Vp, ::cuda::std::initializer_list<_Up>&, _Args...>)
  {
    __emplace<_Vp>(__il, static_cast<_Args&&>(__args)...);
  }

#if _CCCL_HAS_CONCEPTS() || defined(_CCCL_DOXYGEN_INVOKED)
  //! \brief Move constructs a `__basic_any` object.
  //! \pre `_Interface` must extend `__imovable<>`.
  //! \post `__other.has_value() == false` and `has_value()` is `true` if and
  //! only if `__other.has_value()` was `true`.
  _CCCL_API __basic_any(__basic_any&& __other) noexcept
    requires(__extension_of<_Interface, __imovable<>>)
  {
    __convert_from(::cuda::std::move(__other));
  }

  //! \brief Copy constructs a `__basic_any` object.
  //! \pre `_Interface` must extend `__icopyable<>`.
  //! \post `has_value() == __other.has_value()`. If `_Interface` extends
  //! `__iequality_comparable<>`, then `*this == __other` is `true`.
  _CCCL_API __basic_any(__basic_any const& __other)
    requires(__extension_of<_Interface, __icopyable<>>)
  {
    __convert_from(__other);
  }
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  // Without real concepts, we use base classes to implement movability and
  // copyability. All we need here is to accept the default implementations.
  __basic_any(__basic_any&& __other)      = default;
  __basic_any(__basic_any const& __other) = default;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  //! \brief Converting constructor that move constructs from a compatible
  //! `__basic_any` object.
  //! \pre Let `I` be the decayed type of `_OtherInterface`. `I` must extend
  //! `_Interface`. If `_OtherInterface` is a reference type, then `I` must
  //! extend `__icopyable<>`. Otherwise, `I` must extend `__imovable<>`.
  //! \post `__other.has_value() == false` and `has_value()` is `true` if and
  //! only if `__other.has_value()` was `true`.
  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<__basic_any<_OtherInterface>, __basic_any>)
  _CCCL_API __basic_any(__basic_any<_OtherInterface>&& __other)
  {
    __convert_from(::cuda::std::move(__other));
  }

  //! \brief Converting constructor that copy constructs from a compatible
  //! `__basic_any` object.
  //! \pre The decayed type of `_OtherInterface` must extend `_Interface` and
  //! `__icopyable<>`.
  //! \post `has_value() == __other.has_value()`.
  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<__basic_any<_OtherInterface> const&, __basic_any>)
  _CCCL_API __basic_any(__basic_any<_OtherInterface> const& __other)
  {
    __convert_from(__other);
  }

#if _CCCL_COMPILER(CLANG, <, 12) || _CCCL_COMPILER(GCC, <, 11)
  // Older versions of clang and gcc need help disambiguating between
  // __basic_any<__ireference<I>> and __basic_any<I&>.
  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_OtherInterface&>, __basic_any>)
  _CCCL_API __basic_any(__basic_any<_OtherInterface&>&& __other)
  {
    __convert_from(__other);
  }

  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_OtherInterface&>, __basic_any>)
  _CCCL_API __basic_any(__basic_any<_OtherInterface&> const& __other)
  {
    __convert_from(__other);
  }
#endif // _CCCL_COMPILER(CLANG, <, 12) || _CCCL_COMPILER(GCC, <, 11)

  //! \brief Destroys the contained value, if any.
  _CCCL_API ~__basic_any()
  {
    reset();
  }

#if _CCCL_HAS_CONCEPTS() || defined(_CCCL_DOXYGEN_INVOKED)
  //! \brief Move assigns a `__basic_any` object.
  //! \pre `_Interface` must extend `__imovable<>`.
  //! \post `__other.has_value() == false` and `has_value()` is `true` if and
  //! only if `__other.has_value()` was `true`.
  _CCCL_API __basic_any& operator=(__basic_any&& __other) noexcept
    requires(__extension_of<_Interface, __imovable<>>)
  {
    return __assign_from(::cuda::std::move(__other));
  }

  //! \brief Copy assigns a `__basic_any` object.
  //! \pre `_Interface` must extend `__icopyable<>`.
  //! \post `has_value() == __other.has_value()`. If `_Interface` extends
  //! `__iequality_comparable<>`, then `*this == __other` is `true`.
  _CCCL_API __basic_any& operator=(__basic_any const& __other)
    requires(__extension_of<_Interface, __icopyable<>>)
  {
    return __assign_from(__other);
  }
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  // Without real concepts, we use base classes to implement movability and
  // copyability. All we need here is to accept the default implementations.
  auto operator=(__basic_any&& __other) -> __basic_any&      = default;
  auto operator=(__basic_any const& __other) -> __basic_any& = default;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  //! \brief Converting move assignment operator from a compatible `__basic_any`
  //! object.
  //!
  //! Equivalent to:
  //!
  //! \code{.cpp}
  //! __basic_any(cuda::std::move(__other)).swap(*this);
  //! return *this;
  //! \endcode
  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<__basic_any<_OtherInterface>, __basic_any>)
  _CCCL_API auto operator=(__basic_any<_OtherInterface>&& __other) -> __basic_any&
  {
    return __assign_from(::cuda::std::move(__other));
  }

  //! \brief Converting copy assignment operator from a compatible `__basic_any`
  //! object.
  //!
  //! Equivalent to:
  //!
  //! \code{.cpp}
  //! __basic_any(__other).swap(*this);
  //! return *this;
  //! \endcode
  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES((!::cuda::std::same_as<_OtherInterface, _Interface>)
                   _CCCL_AND __any_convertible_to<__basic_any<_OtherInterface> const&, __basic_any>)
  _CCCL_API auto operator=(__basic_any<_OtherInterface> const& __other) -> __basic_any&
  {
    return __assign_from(__other);
  }

#if _CCCL_COMPILER(CLANG, <, 12) || _CCCL_COMPILER(GCC, <, 11)
  // Older versions of clang and gcc need help disambiguating between
  // __basic_any<__ireference<I>> and __basic_any<I&>.
  _CCCL_TEMPLATE(class _OtherInterface)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_OtherInterface&>, __basic_any>)
  _CCCL_API auto operator=(__basic_any<_OtherInterface&> __other) -> __basic_any&
  {
    return __assign_from(__other);
  }
#endif // _CCCL_COMPILER(CLANG, <, 12) || _CCCL_COMPILER(GCC, <, 11)

  //! \brief Implicitly convert to a `__basic_any` non-const reference type:
  [[nodiscard]] _CCCL_API operator __basic_any<__ireference<_Interface>>() & noexcept
  {
    return __basic_any<__ireference<_Interface>>(*this);
  }

  //! \brief Implicitly convert to a `__basic_any` const reference type:
  [[nodiscard]] _CCCL_API operator __basic_any<__ireference<_Interface const>>() const& noexcept
  {
    return __basic_any<__ireference<_Interface const>>(*this);
  }

  //! \brief Exchanges the values of two `__basic_any` objects.
  _CCCL_API void swap(__basic_any& __other) noexcept
  {
    //! if both objects refer to heap-allocated object, we can just
    //! swap the pointers. otherwise, do it the slow(er) way.
    if (!__in_situ() && !__other.__in_situ())
    {
      ::cuda::std::swap(__vptr_, __other.__vptr_);
      __swap_ptr_ptr(__buffer_, __other.__buffer_);
    }

    __basic_any __tmp;
    __tmp.__convert_from(::cuda::std::move(*this));
    (*this).__convert_from(::cuda::std::move(__other));
    __other.__convert_from(::cuda::std::move(__tmp));
  }

  //! \brief Exchanges the values of two `__basic_any` objects.
  friend _CCCL_NODEBUG_API void swap(__basic_any& __lhs, __basic_any& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }

  //! \brief Emplaces a new object of type `_Tp` constructed as
  //! `_Tp{__args...}`.
  //! \pre `_Tp` must satisfy the requirements of `_Interface`.
  //! \post `has_value() == true`
  _CCCL_TEMPLATE(class _Tp, class _Up = ::cuda::std::decay_t<_Tp>, class... _Args)
  _CCCL_REQUIRES(__list_initializable_from<_Up, _Args...>)
  _CCCL_API auto emplace(_Args&&... __args) noexcept(
    __is_small<_Up>(__size_, __align_) && ::cuda::std::is_nothrow_constructible_v<_Up, _Args...>) -> _Up&
  {
    reset();
    return __emplace<_Up>(static_cast<_Args&&>(__args)...);
  }

  //! \brief Emplaces a new object of type `_Tp` constructed as
  //! `_Tp{__il, __args...}`.
  //! \pre `_Tp` must satisfy the requirements of `_Interface`.
  //! \post `has_value() == true`
  _CCCL_TEMPLATE(class _Tp, class _Up, class _Vp = ::cuda::std::decay_t<_Tp>, class... _Args)
  _CCCL_REQUIRES(__list_initializable_from<_Vp, ::cuda::std::initializer_list<_Up>&, _Args...>)
  _CCCL_API auto emplace(::cuda::std::initializer_list<_Up> __il, _Args&&... __args) noexcept(
    __is_small<_Vp>(__size_, __align_)
    && ::cuda::std::is_nothrow_constructible_v<_Vp, ::cuda::std::initializer_list<_Up>&, _Args...>) -> _Vp&
  {
    reset();
    return __emplace<_Vp>(__il, static_cast<_Args&&>(__args)...);
  }

  //! \brief Tests whether the `__basic_any` object contains a value.
  [[nodiscard]] _CCCL_API auto has_value() const noexcept -> bool
  {
    return __get_vptr() != nullptr;
  }

  //! \brief Resets the `__basic_any` object to an empty state.
  //! \post `has_value() == false`
  _CCCL_API void reset() noexcept
  {
    if (auto __vptr = __get_vptr())
    {
      _CCCL_ASSERT(__vptr->__query_interface(__iunknown())->__cookie_ == 0xDEADBEEF,
                   "query_interface returned a bad pointer to the __iunknown vtable");
      __vptr->__query_interface(__iunknown())->__dtor_(__buffer_, __in_situ());
      __release_();
    }
  }

  //! \brief Returns a reference to a type_info object representing the type of
  //! the contained object.
  [[nodiscard]] _CCCL_API auto type() const noexcept -> ::cuda::std::__type_info_ref
  {
    if (auto __vptr = __get_vptr())
    {
      _CCCL_ASSERT(__vptr->__query_interface(__iunknown())->__cookie_ == 0xDEADBEEF,
                   "query_interface returned a bad pointer to the __iunknown vtable");
      return *__vptr->__query_interface(__iunknown())->__object_info_->__object_typeid_;
    }
    return _CCCL_TYPEID(void);
  }

  //! \brief Returns a reference to a type_info object representing the type of
  //! the dynamic interface.
  //!
  //! The dynamic interface is the interface that was used to construct the
  //! object, which may be different from the current object's interface if
  //! there was a conversion.
  [[nodiscard]] _CCCL_API auto interface() const noexcept -> ::cuda::std::__type_info_ref
  {
    if (auto __vptr = __get_vptr())
    {
      _CCCL_ASSERT(__vptr->__query_interface(__iunknown())->__cookie_ == 0xDEADBEEF,
                   "query_interface returned a bad pointer to the __iunknown vtable");
      return *__vptr->__query_interface(__iunknown())->__interface_typeid_;
    }
    return _CCCL_TYPEID(_Interface);
  }

#if !defined(_CCCL_DOXYGEN_INVOKED) // Do not document
  [[nodiscard]] _CCCL_API auto __in_situ() const noexcept -> bool
  {
    return __vptr_.__flag();
  }
#endif // _CCCL_DOXYGEN_INVOKED

private:
  template <class>
  friend struct __basic_any;
  friend struct __basic_any_access;
  template <class, int>
  friend struct __basic_any_base;

  _CCCL_API void __release_()
  {
    __vptr_for<_Interface> __vptr = nullptr;
    __vptr_.__set(__vptr, false);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class... _Args>
  _CCCL_API auto __emplace(_Args&&... __args) noexcept(
    __is_small<_Tp>(__size_, __align_) && ::cuda::std::is_nothrow_constructible_v<_Tp, _Args...>) -> _Tp&
  {
    if constexpr (__is_small<_Tp>(__size_, __align_))
    {
      ::new (__buffer_) _Tp{static_cast<_Args&&>(__args)...};
    }
    else
    {
      ::new (__buffer_)::cuda::std::type_identity_t<_Tp*>{new _Tp{static_cast<_Args&&>(__args)...}};
    }

    __vptr_for<_Interface> __vptr = ::cuda::__get_vtable_ptr_for<_Interface, _Tp>();
    __vptr_.__set(__vptr, __is_small<_Tp>(__size_, __align_));
    return *::cuda::std::launder(static_cast<_Tp*>(__get_optr()));
  }

  // this overload handles moving from __basic_any<_SrcInterface> and
  // __basic_any<__ireference<_SrcInterface>> (but not
  // __basic_any<__ireference<_SrcInterface const>>).
  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES(__any_castable_to<__basic_any<_SrcInterface>, __basic_any>)
  _CCCL_API void
  __convert_from(__basic_any<_SrcInterface>&& __from) noexcept(::cuda::std::is_same_v<_SrcInterface, _Interface>)
  {
    _CCCL_ASSERT(!has_value(), "forgot to clear the destination object first");
    using __src_interface_t _CCCL_NODEBUG_ALIAS = __remove_ireference_t<_SrcInterface>;
    // if the source is an lvalue reference, we need to copy from it.
    if constexpr (__is_lvalue_reference_v<_SrcInterface>)
    {
      __convert_from(__from); // will copy from the source
    }
    else if (auto __to_vptr = __vptr_cast<__src_interface_t, _Interface>(__from.__get_vptr()))
    {
      if (!__from.__in_situ())
      {
        ::new (__buffer_)::cuda::std::type_identity_t<void*>(__from.__get_optr());
        __vptr_.__set(__to_vptr, false);
        __from.__release_();
      }
      else if constexpr (::cuda::std::is_same_v<_SrcInterface, _Interface>)
      {
        __from.__move_to(__buffer_);
        __vptr_.__set(__from.__get_vptr(), true);
        __from.reset();
      }
      else
      {
        bool const __small = __from.__move_to(__buffer_, __size_, __align_);
        __vptr_.__set(__to_vptr, __small);
        __from.reset();
      }
    }
  }

  // this overload handles copying from __basic_any<_Interface>,
  // __basic_any<__ireference<_Interface>>, and __basic_any<_Interface&>.
  _CCCL_TEMPLATE(class _SrcInterface)
  _CCCL_REQUIRES(__any_castable_to<__basic_any<_SrcInterface> const&, __basic_any>)
  _CCCL_API void __convert_from(__basic_any<_SrcInterface> const& __from)
  {
    _CCCL_ASSERT(!has_value(), "forgot to clear the destination object first");
    using __src_interface_t _CCCL_NODEBUG_ALIAS = __remove_ireference_t<::cuda::std::remove_reference_t<_SrcInterface>>;
    if (auto __to_vptr = __vptr_cast<__src_interface_t, _Interface>(__from.__get_vptr()))
    {
      bool const __small = __from.__copy_to(__buffer_, __size_, __align_);
      __vptr_.__set(__to_vptr, __small);
    }
  }

  // Assignment from a compatible __basic_any object handled here:
  _CCCL_TEMPLATE(class _SrcCvAny)
  _CCCL_REQUIRES(__any_castable_to<_SrcCvAny, __basic_any>)
  _CCCL_API auto __assign_from(_SrcCvAny&& __src) -> __basic_any&
  {
    if (!__ptr_eq(this, &__src))
    {
      reset();
      __convert_from(static_cast<_SrcCvAny&&>(__src));
    }
    return *this;
  }

  [[nodiscard]] _CCCL_API auto __get_vptr() const noexcept -> __vptr_for<_Interface>
  {
    return __vptr_.__get();
  }

  [[nodiscard]] _CCCL_API auto __get_optr() noexcept -> void*
  {
    void* __pv = __buffer_;
    return __in_situ() ? __pv : *static_cast<void**>(__pv);
  }

  _CCCL_DIAG_PUSH
  _CCCL_DIAG_SUPPRESS_MSVC(4702) // warning C4702: unreachable code (srsly where, msvc?)
  [[nodiscard]] _CCCL_API auto __get_optr() const noexcept -> void const*
  {
    void const* __pv = __buffer_;
    return __in_situ() ? __pv : *static_cast<void const* const*>(__pv);
  }
  _CCCL_DIAG_POP

  [[nodiscard]] _CCCL_API auto __get_rtti() const noexcept -> __rtti const*
  {
    return __get_vptr()->__query_interface(__iunknown());
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_VALUE_H
