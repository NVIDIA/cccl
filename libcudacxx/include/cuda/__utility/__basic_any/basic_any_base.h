//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_BASE_H
#define _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/__utility/__basic_any/storage.h>
#include <cuda/__utility/__basic_any/tagged_ptr.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstddef> // for byte

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Interface>
_CCCL_API auto __is_basic_any_test(__basic_any<_Interface>&&) -> __basic_any<_Interface>&&;
template <class _Interface>
_CCCL_API auto __is_basic_any_test(__basic_any<_Interface>&) -> __basic_any<_Interface>&;
template <class _Interface>
_CCCL_API auto __is_basic_any_test(__basic_any<_Interface> const&) -> __basic_any<_Interface> const&;

#if _CCCL_COMPILER(CLANG, <, 12) || _CCCL_COMPILER(GCC, <, 11)
// Older versions of clang and gcc need help disambiguating between
// __basic_any<__ireference<I>> and __basic_any<I&>.
template <class _Interface>
_CCCL_API auto __is_basic_any_test(__basic_any<_Interface&>&&) -> __basic_any<_Interface&>&&;
template <class _Interface>
_CCCL_API auto __is_basic_any_test(__basic_any<_Interface&>&) -> __basic_any<_Interface&>&;
template <class _Interface>
_CCCL_API auto __is_basic_any_test(__basic_any<_Interface&> const&) -> __basic_any<_Interface&> const&;
#endif

// clang-format off
template <class _Tp>
_CCCL_CONCEPT __is_basic_any =
  _CCCL_REQUIRES_EXPR((_Tp), _Tp& __value)
  (
    (::cuda::__is_basic_any_test(__value))
  );
// clang-format on

#if _CCCL_HAS_CONCEPTS()
template <class _Interface, int = 0>
struct __basic_any_base : __interface_of<_Interface>
{
private:
  template <class>
  friend struct __basic_any;
  friend struct __basic_any_access;

  static constexpr size_t __size_  = __buffer_size(_Interface::size);
  static constexpr size_t __align_ = __buffer_align(_Interface::align);

  __tagged_ptr<__vptr_for<_Interface>> __vptr_{};
  alignas(__align_)::cuda::std::byte __buffer_[__size_];
};
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
// Without concepts, we need a base class to correctly implement movability
// and copyability.
template <class _Interface, int = __extension_of<_Interface, __imovable<>> + __extension_of<_Interface, __icopyable<>>>
struct __basic_any_base;

template <class _Interface>
struct __basic_any_base<_Interface, 2> : __interface_of<_Interface> // copyable interfaces
{
  __basic_any_base() = default;

  _CCCL_API __basic_any_base(__basic_any_base&& __other) noexcept
  {
    static_cast<__basic_any<_Interface>*>(this)->__convert_from(static_cast<__basic_any<_Interface>&&>(__other));
  }

  _CCCL_API __basic_any_base(__basic_any_base const& __other)
  {
    static_cast<__basic_any<_Interface>*>(this)->__convert_from(static_cast<__basic_any<_Interface> const&>(__other));
  }

  _CCCL_API auto operator=(__basic_any_base&& __other) noexcept -> __basic_any_base&
  {
    static_cast<__basic_any<_Interface>*>(this)->__assign_from(static_cast<__basic_any<_Interface>&&>(__other));
    return *this;
  }

  _CCCL_API auto operator=(__basic_any_base const& __other) -> __basic_any_base&
  {
    static_cast<__basic_any<_Interface>*>(this)->__assign_from(static_cast<__basic_any<_Interface> const&>(__other));
    return *this;
  }

private:
  template <class>
  friend struct __basic_any;
  friend struct __basic_any_access;

  static constexpr size_t __size_  = __buffer_size(_Interface::size);
  static constexpr size_t __align_ = __buffer_align(_Interface::align);

  __tagged_ptr<__vptr_for<_Interface>> __vptr_{};
  alignas(__align_)::cuda::std::byte __buffer_[__size_];
};

template <class _Interface>
struct __basic_any_base<_Interface, 1> : __basic_any_base<_Interface, 2> // move-only interfaces
{
  __basic_any_base()                                               = default;
  __basic_any_base(__basic_any_base&&) noexcept                    = default;
  __basic_any_base(__basic_any_base const&)                        = delete;
  auto operator=(__basic_any_base&&) noexcept -> __basic_any_base& = default;
  auto operator=(__basic_any_base const&) -> __basic_any_base&     = delete;
};

template <class _Interface>
struct __basic_any_base<_Interface, 0> : __basic_any_base<_Interface, 2> // immovable interfaces
{
  __basic_any_base()                                               = default;
  __basic_any_base(__basic_any_base&&) noexcept                    = delete;
  __basic_any_base(__basic_any_base const&)                        = delete;
  auto operator=(__basic_any_base&&) noexcept -> __basic_any_base& = delete;
  auto operator=(__basic_any_base const&) -> __basic_any_base&     = delete;
};
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_BASIC_ANY_BASE_H
