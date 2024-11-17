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

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_const.h>
// #include <cuda/std/__type_traits/remove_reference.h>
// #include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/basic_any_base.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_ref.cuh>
#include <cuda/experimental/__utility/basic_any/conversions.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/rtti.cuh>
#include <cuda/experimental/__utility/basic_any/virtual_tables.cuh>

namespace cuda::experimental
{
///
/// basic_any<_Interface*>
///
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any<_Interface*>
{
  using interface_type                 = _CUDA_VSTD::remove_const_t<_Interface>;
  static constexpr bool __is_const_ptr = _CUDA_VSTD::is_const_v<_Interface>;

  ///
  /// Constructors
  ///
  basic_any() = default;

  _CUDAX_TRIVIAL_API basic_any(_CUDA_VSTD::nullptr_t) {}

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) _LIBCUDACXX_AND __satisfies<_Up, interface_type> _LIBCUDACXX_AND(
    __is_const_ptr || !_CUDA_VSTD::is_const_v<_Tp>))
  _CUDAX_API basic_any(_Tp* __obj) noexcept
  {
    operator=(__obj);
  }

  _CUDAX_API basic_any(basic_any const& __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_OtherInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_OtherInterface*>, basic_any<_Interface*>>)
  _CUDAX_API basic_any(basic_any<_OtherInterface*> const& __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface>&, basic_any<_Interface&>>)
  _CUDAX_API basic_any(basic_any<_OtherInterface>* __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface> const&, basic_any<_Interface&>>)
  _CUDAX_API basic_any(basic_any<_OtherInterface> const* __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _LIBCUDACXX_REQUIRES(__is_interface<_OtherInterface<_Super>> _LIBCUDACXX_AND
                         _CUDA_VSTD::derived_from<basic_any<_Super>, _OtherInterface<_Super>> _LIBCUDACXX_AND
                           _CUDA_VSTD::same_as<__normalized_interface_of<basic_any<_Super>*>, _Interface*>)
  _CUDAX_API explicit basic_any(_OtherInterface<_Super>* __self) noexcept
  {
    __convert_from(basic_any_from(__self));
  }

  _LIBCUDACXX_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _LIBCUDACXX_REQUIRES(__is_interface<_OtherInterface<_Super>> _LIBCUDACXX_AND
                         _CUDA_VSTD::derived_from<basic_any<_Super>, _OtherInterface<_Super>> _LIBCUDACXX_AND
                           _CUDA_VSTD::same_as<__normalized_interface_of<basic_any<_Super> const*>, _Interface*>)
  _CUDAX_API explicit basic_any(_OtherInterface<_Super> const* __self) noexcept
  {
    __convert_from(basic_any_from(__self));
  }

  ///
  /// Assignment operators
  ///
  _CUDAX_API basic_any& operator=(_CUDA_VSTD::nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) _LIBCUDACXX_AND //
                         __satisfies<_Up, interface_type> _LIBCUDACXX_AND //
                       (__is_const_ptr || !_CUDA_VSTD::is_const_v<_Tp>))
  _CUDAX_API basic_any& operator=(_Tp* __obj) noexcept
  {
    __vptr_for<interface_type> __vptr = &__vtable_for_v<interface_type, _Up>;
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *this;
  }

  _CUDAX_API basic_any& operator=(basic_any const& __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_OtherInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_OtherInterface*>, basic_any<_Interface*>>)
  _CUDAX_API basic_any& operator=(basic_any<_OtherInterface*> const& __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface>&, basic_any<_Interface&>>)
  _CUDAX_API basic_any& operator=(basic_any<_OtherInterface>* __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface> const&, basic_any<_Interface&>>)
  _CUDAX_API basic_any& operator=(basic_any<_OtherInterface> const* __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  ///
  /// emplace
  ///
  _LIBCUDACXX_TEMPLATE(
    class _Tp, class _Up = _CUDA_VSTD::remove_pointer_t<_Tp>, class _Vp = _CUDA_VSTD::remove_const_t<_Up>)
  _LIBCUDACXX_REQUIRES(__satisfies<_Vp, _Interface> _LIBCUDACXX_AND(__is_const_ptr || !_CUDA_VSTD::is_const_v<_Up>))
  _CUDAX_API _CUDA_VSTD::__maybe_const<__is_const_ptr, _Vp>*& emplace(_CUDA_VSTD::type_identity_t<_Up>* __obj) noexcept
  {
    __vptr_for<interface_type> __vptr = &__vtable_for_v<interface_type, _Vp>;
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *static_cast<_CUDA_VSTD::__maybe_const<__is_const_ptr, _Vp>**>(static_cast<void*>(&__ref_.__optr_));
  }

#if defined(__cpp_three_way_comparison)
  _CCCL_NODISCARD _CUDAX_API bool operator==(basic_any const& __other) const noexcept
  {
    using __void_ptr_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__get_optr()) == *static_cast<__void_ptr_t>(__other.__get_optr());
  }
#else
  _CCCL_NODISCARD_FRIEND _CUDAX_API bool operator==(basic_any const& __lhs, basic_any const& __rhs) noexcept
  {
    using __void_ptr_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__lhs.__get_optr()) == *static_cast<__void_ptr_t>(__rhs.__get_optr());
  }

  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_API bool operator!=(basic_any const& __lhs, basic_any const& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif

  using __any_ref_t _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::__maybe_const<__is_const_ptr, basic_any<__ireference<_Interface>>>;

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API auto operator->() const noexcept -> __any_ref_t*
  {
    return &__ref_;
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API auto operator*() const noexcept -> __any_ref_t&
  {
    return __ref_;
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref type() const noexcept
  {
    return __ref_.__vptr_ != nullptr
           ? (__is_const_ptr ? *__get_rtti()->__object_info_->__const_pointer_typeid_
                             : *__get_rtti()->__object_info_->__pointer_typeid_)
           : _CCCL_TYPEID(void);
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref interface() const noexcept
  {
    return __ref_.__vptr_ != nullptr ? *__get_rtti()->__interface_typeid_ : _CCCL_TYPEID(interface_type);
  }

  _CCCL_NODISCARD _CUDAX_API bool has_value() const noexcept
  {
    return __ref_.__vptr_ != nullptr;
  }

  _CUDAX_API void reset() noexcept
  {
    __vptr_for<interface_type> __vptr = nullptr;
    __ref_.__set_ref(__vptr, nullptr);
  }

  _CCCL_NODISCARD _CUDAX_API explicit operator bool() const noexcept
  {
    return __ref_.__vptr_ != nullptr;
  }

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Do not document
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API static constexpr bool __in_situ() noexcept
  {
    return true;
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  template <class _SrcCvAny>
  _CUDAX_API void __convert_from(_SrcCvAny* __other) noexcept
  {
    __other ? __ref_.__set_ref(__other->__get_vptr(), __other->__get_optr()) : reset();
  }

  template <class _OtherInterface>
  _CUDAX_API void __convert_from(basic_any<_OtherInterface*> const& __other) noexcept
  {
    using __other_interface_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_OtherInterface>;
    auto __to_vptr = __try_vptr_cast<__other_interface_t, interface_type>(__other.__get_vptr());
    auto __to_optr = __to_vptr ? *__other.__get_optr() : nullptr;
    __ref_.__set_ref(__to_vptr, __to_optr);
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__maybe_const<__is_const_ptr, void>** __get_optr() noexcept
  {
    return &__ref_.__optr_;
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const* __get_optr() const noexcept
  {
    return &__ref_.__optr_;
  }

  _CCCL_NODISCARD _CUDAX_API __vptr_for<interface_type> __get_vptr() const noexcept
  {
    return __ref_.__vptr_;
  }

  _CCCL_NODISCARD _CUDAX_API __rtti const* __get_rtti() const noexcept
  {
    return __ref_.__vptr_ ? __ref_.__vptr_->__query_interface(iunknown()) : nullptr;
  }

  mutable basic_any<__ireference<_Interface>> __ref_;
};

_LIBCUDACXX_TEMPLATE(template <class...> class _Interface, class _Super)
_LIBCUDACXX_REQUIRES(__is_interface<_Interface<_Super>>)
_CUDAX_PUBLIC_API basic_any(_Interface<_Super>*) //
  -> basic_any<__normalized_interface_of<basic_any<_Super>*>>;

_LIBCUDACXX_TEMPLATE(template <class...> class _Interface, class _Super)
_LIBCUDACXX_REQUIRES(__is_interface<_Interface<_Super>>)
_CUDAX_PUBLIC_API basic_any(_Interface<_Super> const*) //
  -> basic_any<__normalized_interface_of<basic_any<_Super> const*>>;

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_PTR_H
