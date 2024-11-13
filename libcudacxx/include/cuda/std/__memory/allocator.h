// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ALLOCATOR_H
#define _LIBCUDACXX___MEMORY_ALLOCATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/allocate_at_least.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__new_>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>

#if defined(_CCCL_HAS_CONSTEXPR_ALLOCATION) && !defined(_CCCL_COMPILER_NVRTC)
#  include <memory>
#endif // _CCCL_HAS_CONSTEXPR_ALLOCATION && !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
class allocator;

#if _CCCL_STD_VER <= 2017
// These specializations shouldn't be marked _LIBCUDACXX_DEPRECATED_IN_CXX17.
// Specializing allocator<void> is deprecated, but not using it.
template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT allocator<void>
{
public:
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef void* pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const void* const_pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef void value_type;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED_IN_CXX17 rebind
  {
    typedef allocator<_Up> other;
  };
};

template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT allocator<const void>
{
public:
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const void* pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const void* const_pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const void value_type;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED_IN_CXX17 rebind
  {
    typedef allocator<_Up> other;
  };
};
#endif // _CCCL_STD_VER <= 2017

// This class provides a non-trivial default constructor to the class that derives from it
// if the condition is satisfied.
//
// The second template parameter exists to allow giving a unique type to __non_trivial_if,
// which makes it possible to avoid breaking the ABI when making this a base class of an
// existing class. Without that, imagine we have classes D1 and D2, both of which used to
// have no base classes, but which now derive from __non_trivial_if. The layout of a class
// that inherits from both D1 and D2 will change because the two __non_trivial_if base
// classes are not allowed to share the same address.
//
// By making those __non_trivial_if base classes unique, we work around this problem and
// it is safe to start deriving from __non_trivial_if in existing classes.
template <bool _Cond, class _Unique>
struct __non_trivial_if
{};

template <class _Unique>
struct __non_trivial_if<true, _Unique>
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __non_trivial_if() noexcept {}
};

// allocator
//
// Note: For ABI compatibility between C++20 and previous standards, we make
//       allocator<void> trivial in C++20.

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT allocator : private __non_trivial_if<!_CCCL_TRAIT(is_void, _Tp), allocator<_Tp>>
{
  static_assert(!_CCCL_TRAIT(is_volatile, _Tp), "std::allocator does not support volatile types");

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef _Tp value_type;
  typedef true_type propagate_on_container_move_assignment;
  typedef true_type is_always_equal;

  _CCCL_CONSTEXPR_CXX20 allocator() noexcept = default;

  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 allocator(const allocator<_Up>&) noexcept
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20_ALLOCATION _Tp* allocate(size_t __n)
  {
    if (__n > allocator_traits<allocator>::max_size(*this))
    {
      __throw_bad_array_new_length();
    }
#if defined(_CCCL_HAS_CONSTEXPR_ALLOCATION)
    if (__libcpp_is_constant_evaluated())
    {
      return ::std::allocator<_Tp>{}.allocate(__n);
    }
#endif // _CCCL_HAS_CONSTEXPR_ALLOCATION
    {
      return static_cast<_Tp*>(_CUDA_VSTD::__libcpp_allocate(__n * sizeof(_Tp), _LIBCUDACXX_ALIGNOF(_Tp)));
    }
  }

#if _CCCL_STD_VER >= 2023
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr allocation_result<_Tp*> allocate_at_least(size_t __n)
  {
    return {allocate(__n), __n};
  }
#endif // _CCCL_HAS_CONSTEXPR_ALLOCATION

  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20_ALLOCATION void deallocate(_Tp* __p, size_t __n) noexcept
  {
#if defined(_CCCL_HAS_CONSTEXPR_ALLOCATION)
    if (__libcpp_is_constant_evaluated())
    {
      return ::std::allocator<_Tp>{}.deallocate(__p, __n);
    }
    else
#endif // _CCCL_STD_VER >= 2020
    {
      _CUDA_VSTD::__libcpp_deallocate((void*) __p, __n * sizeof(_Tp), _LIBCUDACXX_ALIGNOF(_Tp));
    }
  }

  // C++20 Removed members
#if _CCCL_STD_VER <= 2017
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef _Tp* pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const _Tp* const_pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef _Tp& reference;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const _Tp& const_reference;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED_IN_CXX17 rebind
  {
    typedef allocator<_Up> other;
  };

  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI pointer address(reference __x) const noexcept
  {
    return _CUDA_VSTD::addressof(__x);
  }
  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI const_pointer address(const_reference __x) const noexcept
  {
    return _CUDA_VSTD::addressof(__x);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_DEPRECATED_IN_CXX17 _Tp* allocate(size_t __n, const void*)
  {
    return allocate(__n);
  }

  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI size_type max_size() const noexcept
  {
    return size_type(~0) / sizeof(_Tp);
  }

  template <class _Up, class... _Args>
  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI void construct(_Up* __p, _Args&&... __args)
  {
    ::new ((void*) __p) _Up(_CUDA_VSTD::forward<_Args>(__args)...);
  }

  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI void destroy(pointer __p) noexcept
  {
    __p->~_Tp();
  }
#endif // _CCCL_STD_VER <= 2017
};

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT allocator<const _Tp>
    : private __non_trivial_if<!_CCCL_TRAIT(is_void, _Tp), allocator<const _Tp>>
{
  static_assert(!_CCCL_TRAIT(is_volatile, _Tp), "std::allocator does not support volatile types");

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef const _Tp value_type;
  typedef true_type propagate_on_container_move_assignment;
  typedef true_type is_always_equal;

  _CCCL_CONSTEXPR_CXX20 allocator() noexcept = default;

  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 allocator(const allocator<_Up>&) noexcept
  {}

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 const _Tp* allocate(size_t __n)
  {
    if (__n > allocator_traits<allocator>::max_size(*this))
    {
      __throw_bad_array_new_length();
    }
    if (__libcpp_is_constant_evaluated())
    {
      return static_cast<const _Tp*>(::operator new(__n * sizeof(_Tp)));
    }
    else
    {
      return static_cast<const _Tp*>(_CUDA_VSTD::__libcpp_allocate(__n * sizeof(_Tp), _LIBCUDACXX_ALIGNOF(_Tp)));
    }
  }

#if _CCCL_STD_VER >= 2023
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr allocation_result<const _Tp*> allocate_at_least(size_t __n)
  {
    return {allocate(__n), __n};
  }
#endif // _CCCL_STD_VER >= 2023

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 void deallocate(const _Tp* __p, size_t __n) noexcept
  {
    if (__libcpp_is_constant_evaluated())
    {
      ::operator delete(const_cast<_Tp*>(__p));
    }
    else
    {
      _CUDA_VSTD::__libcpp_deallocate((void*) const_cast<_Tp*>(__p), __n * sizeof(_Tp), _LIBCUDACXX_ALIGNOF(_Tp));
    }
  }

  // C++20 Removed members
#if _CCCL_STD_VER <= 2017
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const _Tp* pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const _Tp* const_pointer;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const _Tp& reference;
  _LIBCUDACXX_DEPRECATED_IN_CXX17 typedef const _Tp& const_reference;

  template <class _Up>
  struct _LIBCUDACXX_DEPRECATED_IN_CXX17 rebind
  {
    typedef allocator<_Up> other;
  };

  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI const_pointer address(const_reference __x) const noexcept
  {
    return _CUDA_VSTD::addressof(__x);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_DEPRECATED_IN_CXX17 const _Tp* allocate(size_t __n, const void*)
  {
    return allocate(__n);
  }

  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI size_type max_size() const noexcept
  {
    return size_type(~0) / sizeof(_Tp);
  }

  template <class _Up, class... _Args>
  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI void construct(_Up* __p, _Args&&... __args)
  {
    ::new ((void*) __p) _Up(_CUDA_VSTD::forward<_Args>(__args)...);
  }

  _LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_HIDE_FROM_ABI void destroy(pointer __p) noexcept
  {
    __p->~_Tp();
  }
#endif // _CCCL_STD_VER <= 2017
};

template <class _Tp, class _Up>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 bool operator==(const allocator<_Tp>&, const allocator<_Up>&) noexcept
{
  return true;
}

template <class _Tp, class _Up>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 bool operator!=(const allocator<_Tp>&, const allocator<_Up>&) noexcept
{
  return false;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_ALLOCATOR_H
