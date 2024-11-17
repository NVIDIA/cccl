//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_BASE_H
#define __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/storage.cuh>
#include <cuda/experimental/__utility/basic_any/tagged_ptr.cuh>

namespace cuda::experimental
{
template <class _Interface>
_CUDAX_API auto __is_basic_any_test(basic_any<_Interface>&&) -> basic_any<_Interface>&&;
template <class _Interface>
_CUDAX_API auto __is_basic_any_test(basic_any<_Interface>&) -> basic_any<_Interface>&;
template <class _Interface>
_CUDAX_API auto __is_basic_any_test(basic_any<_Interface> const&) -> basic_any<_Interface> const&;

// clang-format off
template <class _Tp>
_LIBCUDACXX_CONCEPT __is_basic_any =
  _LIBCUDACXX_REQUIRES_EXPR((_Tp), _Tp& __value)
  (
    __is_basic_any_test(__value)
  );
// clang-format on

#if defined(__cpp_concepts)
template <class _Interface, int = 0>
struct __basic_any_base : __interface_of<_Interface>
{
private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  static constexpr size_t __size_  = __buffer_size(_Interface::size);
  static constexpr size_t __align_ = __buffer_align(_Interface::align);

  __tagged_ptr<__vptr_for<_Interface>> __vptr_{};
  alignas(__align_) _CUDA_VSTD_NOVERSION::byte __buffer_[__size_];
};
#else
// Without concepts, we need a base class to correctly implement movability
// and copyability.
template <class _Interface, int = extension_of<_Interface, imovable<>> + extension_of<_Interface, icopyable<>>>
struct __basic_any_base;

template <class _Interface>
struct __basic_any_base<_Interface, 2> : __interface_of<_Interface> // copyable interfaces
{
  __basic_any_base() = default;

  _CUDAX_API __basic_any_base(__basic_any_base&& __other) noexcept
  {
    static_cast<basic_any<_Interface>*>(this)->__convert_from(static_cast<basic_any<_Interface>&&>(__other));
  }

  _CUDAX_API __basic_any_base(__basic_any_base const& __other)
  {
    static_cast<basic_any<_Interface>*>(this)->__convert_from(static_cast<basic_any<_Interface> const&>(__other));
  }

  _CUDAX_API __basic_any_base& operator=(__basic_any_base&& __other) noexcept
  {
    static_cast<basic_any<_Interface>*>(this)->__assign_from(static_cast<basic_any<_Interface>&&>(__other));
    return *this;
  }

  _CUDAX_API __basic_any_base& operator=(__basic_any_base const& __other)
  {
    static_cast<basic_any<_Interface>*>(this)->__assign_from(static_cast<basic_any<_Interface> const&>(__other));
    return *this;
  }

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  static constexpr size_t __size_  = __buffer_size(_Interface::size);
  static constexpr size_t __align_ = __buffer_align(_Interface::align);

  __tagged_ptr<__vptr_for<_Interface>> __vptr_{};
  alignas(__align_) _CUDA_VSTD_NOVERSION::byte __buffer_[__size_];
};

template <class _Interface>
struct __basic_any_base<_Interface, 1> : __basic_any_base<_Interface, 2> // move-only interfaces
{
  __basic_any_base()                                       = default;
  __basic_any_base(__basic_any_base&&) noexcept            = default;
  __basic_any_base(__basic_any_base const&)                = delete;
  __basic_any_base& operator=(__basic_any_base&&) noexcept = default;
  __basic_any_base& operator=(__basic_any_base const&)     = delete;
};

template <class _Interface>
struct __basic_any_base<_Interface, 0> : __basic_any_base<_Interface, 2> // immovable interfaces
{
  __basic_any_base()                                       = default;
  __basic_any_base(__basic_any_base&&) noexcept            = delete;
  __basic_any_base(__basic_any_base const&)                = delete;
  __basic_any_base& operator=(__basic_any_base&&) noexcept = delete;
  __basic_any_base& operator=(__basic_any_base const&)     = delete;
};
#endif

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_BASIC_ANY_BASE_H
