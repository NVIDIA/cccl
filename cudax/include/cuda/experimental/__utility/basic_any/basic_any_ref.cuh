//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/__concepts/__concept_macros.h>
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

namespace cuda::experimental
{
///
/// \c __ireference<_Interface>
///
/// For an interface `I`, `__ireference<I>` represents a reference to an object
/// that satisfies the given interface. When you dereference a `basic_any<I*>`,
/// you get a `basic_any<__ireference<I>>`. Also, `basic_any<I&>` is implemented
/// in terms of `basic_any<__ireference<I>>`.
///
/// Note: a `basic_any<__ireference<_Interface>> &&` is an rvalue reference,
/// whereas a `basic_any<_Interface &> &&` is an lvalue reference.
template <class _Interface>
struct __ireference : _Interface
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expected a class type");
  static constexpr size_t __size_      = sizeof(void*);
  static constexpr size_t __align_     = alignof(void*);
  static constexpr bool __is_const_ref = _CUDA_VSTD::is_const_v<_Interface>;

  using interface _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_Interface>;
};

#if !defined(__cpp_concepts)
///
/// A base class for basic_any<__ireference<_Interface>> that provides a
/// conversion to basic_any<__ireference<_Interface const>>. Only used
/// when concepts are not available.
///
template <class _Interface>
struct __basic_any_reference_conversion_base
{
  _CCCL_NODISCARD _CUDAX_HOST_API operator basic_any<__ireference<_Interface const>>() const noexcept
  {
    return basic_any<__ireference<_Interface const>>(static_cast<basic_any<__ireference<_Interface>> const&>(*this));
  }
};

template <class _Interface>
struct __basic_any_reference_conversion_base<_Interface const>
{};
#endif

_CCCL_DIAG_PUSH
// "operator basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_DIAG_SUPPRESS_NVHPC(conversion_function_not_usable)
// "operator basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_NV_DIAG_SUPPRESS(554)

///
/// \c basic_any<__ireference<_Interface>>
///
/// A `basic_any<__ireference<_Interface>>` is a reference to an object that
/// satisfies the given interface. It is used as the result of dereferencing
/// a `basic_any<_Interface*>` and as the implementation of
/// `basic_any<_Interface&>`.
///
/// `basic_any<__ireference<_Interface>>` is neither copyable nor movable. It is
/// not an end-user type.
template <class _Interface, class Select>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES basic_any<__ireference<_Interface>, Select>
    : __interface_of<__ireference<_Interface>>
#if !defined(__cpp_concepts)
    , __basic_any_reference_conversion_base<_Interface>
#endif
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expecting a class type");
  using interface_type                 = _CUDA_VSTD::remove_const_t<_Interface>;
  static constexpr bool __is_const_ref = _CUDA_VSTD::is_const_v<_Interface>;

  basic_any(basic_any&&)      = delete;
  basic_any(basic_any const&) = delete;

  basic_any& operator=(basic_any&&)      = delete;
  basic_any& operator=(basic_any const&) = delete;

#if defined(__cpp_concepts)
  /// \brief A non-const basic_any reference can be implicitly converted to a
  /// const basic_any reference.
  _CCCL_NODISCARD _CUDAX_HOST_API operator basic_any<__ireference<_Interface const>>() const noexcept
    requires(!__is_const_ref)
  {
    return basic_any<__ireference<_Interface const>>(*this);
  }
#endif

  /// \brief Returns a const reference to the type_info for the decayed type
  /// of the type-erased object.
  _CCCL_NODISCARD _CUDAX_HOST_API _CUDA_VSTD::__type_info_ref type() const noexcept
  {
    return *__get_rtti()->__object_info_->__object_typeid_;
  }

  /// \brief Returns a const reference to the type_info for the decayed type
  /// of the type-erased object.
  _CCCL_NODISCARD _CUDAX_HOST_API _CUDA_VSTD::__type_info_ref interface() const noexcept
  {
    return *__get_rtti()->__interface_typeid_;
  }

  /// \brief Returns a reference to a type_info object representing the type of
  /// the dynamic interface.
  ///
  /// The dynamic interface is the interface that was used to construct the
  /// object, which may be different from the current object's interface if
  /// there was a conversion.
  _CCCL_NODISCARD _CUDAX_TRIVIAL_HOST_API static constexpr bool has_value() noexcept
  {
    return true;
  }

#if !defined(DOXYGEN_ACTIVE) // Do not document
  _CCCL_NODISCARD _CUDAX_TRIVIAL_HOST_API static constexpr bool __in_situ() noexcept
  {
    return true;
  }
#endif // DOXYGEN_ACTIVE

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  basic_any() = default;

  /// \brief Constructs a \c basic_any<__ireference<_Interface>> from an lvalue
  /// reference to a \c basic_any<_Interface>.
  _CUDAX_HOST_API explicit basic_any(
    _CUDA_VSTD::__maybe_const<__is_const_ref, basic_any<interface_type>>& __other) noexcept
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  /// \brief Constructs a \c basic_any<__ireference<_Interface>> from a
  /// vtable pointer and an object pointer.
  _CUDAX_HOST_API basic_any(__vptr_for<interface_type> __vptr,
                            _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __optr) noexcept
      : __vptr_(__vptr)
      , __optr_(__optr)
  {}

  /// \brief No-op.
  _CUDAX_TRIVIAL_HOST_API void reset() noexcept {}

  /// \brief No-op.
  _CUDAX_TRIVIAL_HOST_API void __release() {}

  /// \brief Rebinds the reference with a vtable pointer and object pointer.
  _CUDAX_HOST_API void __set_ref(__vptr_for<interface_type> __vptr,
                                 _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    __vptr_ = __vptr;
    __optr_ = __vptr_ ? __obj : nullptr;
  }

  /// \brief Rebinds the reference with a vtable pointer for a different
  /// interface and object pointer. The vtable pointer is cast to the correct
  /// type. If the cast fails, the reference is set to null.
  template <class _VTable>
  _CUDAX_HOST_API void __set_ref(_VTable const* __other, _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = typename _VTable::interface;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  /// \brief Rebinds the reference with a pointer to an iset vtable, which
  /// is cast to a pointer to the vtable for `_Interface`. If \c _Interface
  /// is a specialization `__iset<Is...>`, the cast succeeds if the
  /// \c Is... is a subset of `Interfaces...`. Otherwise, the cast succeeds
  /// if the \c _Interface is a base of one of `Interfaces...`.
  template <class... _Interfaces>
  _CUDAX_HOST_API void
  __set_ref(__iset_vptr<_Interfaces...> __other, _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = __iset<_Interfaces...>;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  template <class _SrcCvAny>
  _CUDAX_HOST_API void __convert_from(_SrcCvAny&& __from)
  {
    using __src_interface_t _CCCL_NODEBUG_ALIAS = typename _CUDA_VSTD::remove_reference_t<_SrcCvAny>::interface_type;
    if (!__from.has_value())
    {
      __throw_bad_any_cast();
    }
    auto __to_vptr = __vptr_cast<__src_interface_t, interface_type>(__from.__get_vptr());
    __set_ref(__to_vptr, __from.__get_optr());
  }

  _CUDAX_TRIVIAL_HOST_API _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __get_optr() const noexcept
  {
    return __optr_;
  }

  _CUDAX_TRIVIAL_HOST_API __vptr_for<interface_type> __get_vptr() const noexcept
  {
    return __vptr_;
  }

  _CUDAX_TRIVIAL_HOST_API __rtti const* __get_rtti() const noexcept
  {
    return __vptr_->__query_interface(iunknown());
  }

  __vptr_for<interface_type> __vptr_{};
  _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __optr_{};
};

_CCCL_NV_DIAG_DEFAULT(554)
_CCCL_DIAG_POP

///
/// basic_any<_Interface&>
///
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any<_Interface&> : basic_any<__ireference<_Interface>, __secondary>
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expecting a class type");
  using typename basic_any<__ireference<_Interface>, __secondary>::interface_type;
  using basic_any<__ireference<_Interface>, __secondary>::__is_const_ref;

  _CUDAX_HOST_API basic_any(basic_any const& __other) noexcept
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) /*_LIBCUDACXX_AND __satisfies<_Up, interface_type> _LIBCUDACXX_AND(
    __is_const_ref || !_CUDA_VSTD::is_const_v<_Tp>)*/)
  _CUDAX_HOST_API basic_any(_Tp& __obj) noexcept
  {
    __vptr_for<interface_type> const __vptr = &__vtable_for_v<interface_type, _Up>;
    this->__set_ref(__vptr, &__obj);
  }

#if defined(_CCCL_CUDA_COMPILER_NVCC) && _CCCL_STD_VER >= 2020
// For some reason, the constructor overloads below give nvcc fits when
// constrained with c++20 requires clauses. So we fall back to good ol'
// enable_if.
#  define _CUDAX_TEMPLATE(...) template <__VA_ARGS__,
#  define _CUDAX_REQUIRES(...) _CUDA_VSTD::enable_if_t<__VA_ARGS__, int> = 0 >
#  define _CUDAX_AND           , int > = 0, _CUDA_VSTD::enable_if_t <
#else
#  define _CUDAX_TEMPLATE _LIBCUDACXX_TEMPLATE
#  define _CUDAX_REQUIRES _LIBCUDACXX_REQUIRES
#  define _CUDAX_AND      _LIBCUDACXX_AND
#endif

  _CUDAX_TEMPLATE(class _Tp)
  _CUDAX_REQUIRES((!__is_basic_any<_Tp>) )
  basic_any(_Tp const&&) = delete;

  _CUDAX_TEMPLATE(class _SrcInterface)
  _CUDAX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>) _CUDAX_AND //
                  (!__is_value_v<_SrcInterface>) _CUDAX_AND //
                    __any_convertible_to<basic_any<_SrcInterface>, basic_any>)
  _CUDAX_HOST_API basic_any(basic_any<_SrcInterface>&& __src) noexcept
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _CUDAX_TEMPLATE(class _SrcInterface)
  _CUDAX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>) _CUDAX_AND //
                    __any_convertible_to<basic_any<_SrcInterface>&, basic_any>)
  _CUDAX_HOST_API basic_any(basic_any<_SrcInterface>& __src) noexcept
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _CUDAX_TEMPLATE(class _SrcInterface)
  _CUDAX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>) _CUDAX_AND //
                    __any_convertible_to<basic_any<_SrcInterface> const&, basic_any>)
  _CUDAX_HOST_API basic_any(basic_any<_SrcInterface> const& __src) noexcept
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  // A temporary value cannot bind to a basic_any reference.
  // TODO: find another way to support APIs that take by reference and want
  // implicit conversion from prvalues.
  _CUDAX_TEMPLATE(class _SrcInterface)
  _CUDAX_REQUIRES(__is_value_v<_SrcInterface>) //
  basic_any(basic_any<_SrcInterface> const&&) = delete;

#undef _CUDAX_AND
#undef _CUDAX_REQUIRES
#undef _CUDAX_TEMPLATE

  basic_any& operator=(basic_any&&)      = delete;
  basic_any& operator=(basic_any const&) = delete;

  _CUDAX_TRIVIAL_HOST_API basic_any<__ireference<_Interface>>&& move() & noexcept
  {
    return _CUDA_VSTD::move(*this);
  }

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  basic_any() = default;
};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_REF_H
