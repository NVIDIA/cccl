// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_COPYABLE_BOX_H
#define _LIBCUDACXX___RANGES_COPYABLE_BOX_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/copyable.h"
#include "../__concepts/invocable.h"
#include "../__concepts/movable.h"
#include "../__memory/addressof.h"
#include "../__memory/construct_at.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__tuple_dir/sfinae_helpers.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__type_traits/is_default_constructible.h"
#include "../__type_traits/is_move_assignable.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_assignable.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_trivially_copy_assignable.h"
#include "../__type_traits/is_trivially_copy_constructible.h"
#include "../__type_traits/is_trivially_destructible.h"
#include "../__type_traits/is_trivially_move_assignable.h"
#include "../__type_traits/is_trivially_move_constructible.h"
#include "../__type_traits/is_void.h"
#include "../__utility/exception_guard.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

// __copyable_box allows turning a type that is copy-constructible (but maybe not copy-assignable) into
// a type that is both copy-constructible and copy-assignable. It does that by introducing an empty state
// and basically doing destroy-then-copy-construct in the assignment operator. The empty state is necessary
// to handle the case where the copy construction fails after destroying the object.
//
// In some cases, we can completely avoid the use of an empty state; we provide a specialization of
// __copyable_box that does this, see below for the details.
#if _LIBCUDACXX_STD_VER > 17
  template<class _Tp>
  concept __copy_constructible_object = copy_constructible<_Tp> && is_object_v<_Tp>;
#else
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __copy_constructible_object_,
    requires()(
      requires(copy_constructible<_Tp>),
      requires(is_object_v<_Tp>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __copy_constructible_object = _LIBCUDACXX_FRAGMENT(__copy_constructible_object_, _Tp);
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template<class _Tp, bool = is_trivially_destructible_v<_Tp>>
struct __cb_destruct_base {
  union
  {
      _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;
  };
  bool __engaged_ = false;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_destruct_base() noexcept {};

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_destruct_base(const bool __engaged) noexcept : __engaged_(__engaged) {}

  template<class ..._Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_destruct_base(in_place_t, _Args&& ...__args)
    noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
    : __val_(_CUDA_VSTD::forward<_Args>(__args)...), __engaged_(true)
  {}
};

template<class _Tp>
struct __cb_destruct_base<_Tp, false> {
  union
  {
      _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;
  };
  bool __engaged_ = false;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_destruct_base() noexcept {};

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_destruct_base(const bool __engaged) noexcept : __engaged_(__engaged) {}

  template<class ..._Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_destruct_base(in_place_t, _Args&& ...__args)
    noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
    : __val_(_CUDA_VSTD::forward<_Args>(__args)...), __engaged_(true)
  { }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  ~__cb_destruct_base() noexcept {
      if (__engaged_) {
          __val_.~_Tp();
      }
  }
};

template<class _Tp, bool = is_trivially_copy_constructible_v<_Tp>>
struct __cb_copy_base : public __cb_destruct_base<_Tp> {
  using __base = __cb_destruct_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __cb_copy_base() noexcept = default;

  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_copy_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp>
struct __cb_copy_base<_Tp, false> : __cb_destruct_base<_Tp> {
  using __base = __cb_destruct_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_copy_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __cb_copy_base() noexcept = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_copy_base(const __cb_copy_base& __other) noexcept(is_nothrow_copy_constructible_v<_Tp>)
    : __base(__other.__engaged_)
  {
    if (__other.__engaged_) {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
    }
  }

  constexpr __cb_copy_base(__cb_copy_base&&) = default;
  constexpr __cb_copy_base& operator=(const __cb_copy_base&) = default;
  constexpr __cb_copy_base& operator=(__cb_copy_base&&) = default;
};

template<class _Tp, bool = is_trivially_move_constructible_v<_Tp>>
struct __cb_move_base : public __cb_copy_base<_Tp> {
  using __base = __cb_copy_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __cb_move_base() noexcept = default;

  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_move_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp>
struct __cb_move_base<_Tp, false> : __cb_copy_base<_Tp> {
  using __base = __cb_copy_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_move_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __cb_move_base() noexcept = default;
  constexpr __cb_move_base(const __cb_move_base&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_move_base(__cb_move_base&& __other) noexcept(is_nothrow_move_constructible_v<_Tp>)
    : __base(__other.__engaged_)
  {
    if (__other.__engaged_) {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
    }
  }

  constexpr __cb_move_base& operator=(const __cb_move_base&) = default;
  constexpr __cb_move_base& operator=(__cb_move_base&&) = default;
};

template<class _Tp, bool = is_trivially_destructible_v<_Tp> &&
                           is_trivially_copy_constructible_v<_Tp> &&
                           is_trivially_copy_assignable_v<_Tp>,
                    bool = copyable<_Tp>>
struct __cb_copy_assign_base : public __cb_move_base<_Tp> {
  using __base = __cb_move_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __cb_copy_assign_base() noexcept = default;

  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_copy_assign_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp>
struct __cb_copy_assign_base<_Tp, false, true> : __cb_copy_base<_Tp> {
  using __base = __cb_copy_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_copy_assign_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __cb_copy_assign_base() noexcept = default;
  constexpr __cb_copy_assign_base(const __cb_copy_assign_base&) = default;
  constexpr __cb_copy_assign_base(__cb_copy_assign_base&&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_copy_assign_base& operator=(const __cb_copy_assign_base& __other) noexcept(
    is_nothrow_copy_assignable_v<_Tp> && is_nothrow_copy_constructible_v<_Tp>)
  {
    if (this == _CUDA_VSTD::addressof(__other)) {
      return *this;
    }

    if (this->__engaged_ && __other.__engaged_) {
      this->__val_ = __other.__val_;
    } else if (this->__engaged_) {
      if constexpr(!is_trivially_destructible_v<_Tp>) {
        this->__val_.~_Tp();
      }
      this->__engaged_ = false;
    } else if (__other.__engaged_) {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
      this->__engaged_ = true;
    }
    return *this;
  }

  constexpr __cb_copy_assign_base& operator=(__cb_copy_assign_base&&) = default;
};

template <class _Tp>
struct __cb_copy_assign_base<_Tp, false, false> : __cb_copy_base<_Tp> {
  using __base = __cb_copy_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_copy_assign_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __cb_copy_assign_base() noexcept = default;
  constexpr __cb_copy_assign_base(const __cb_copy_assign_base&) = default;
  constexpr __cb_copy_assign_base(__cb_copy_assign_base&&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_copy_assign_base& operator=(const __cb_copy_assign_base& __other) noexcept(is_nothrow_copy_constructible_v<_Tp>)
  {
    if (this == _CUDA_VSTD::addressof(__other)) {
      return *this;
    }

    if constexpr(!is_trivially_destructible_v<_Tp>) {
      if (this->__engaged_) {
        this->__val_.~_Tp();
      }
    }

    if (__other.__engaged_) {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
    }
    this->__engaged_ = __other.__engaged_;
    return *this;
  }

  constexpr __cb_copy_assign_base& operator=(__cb_copy_assign_base&&) = default;
};

template<class _Tp, bool = is_trivially_destructible_v<_Tp> &&
                           is_trivially_move_constructible_v<_Tp> &&
                           is_trivially_move_assignable_v<_Tp>,
                    bool = movable<_Tp>>
struct __cb_move_assign_base : public __cb_copy_assign_base<_Tp> {
  using __base = __cb_copy_assign_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __cb_move_assign_base() noexcept = default;

  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_move_assign_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp>
struct __cb_move_assign_base<_Tp, false, true> : __cb_copy_assign_base<_Tp> {
  using __base = __cb_copy_assign_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_move_assign_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC

  constexpr __cb_move_assign_base() noexcept = default;
  constexpr __cb_move_assign_base(const __cb_move_assign_base&) = default;
  constexpr __cb_move_assign_base(__cb_move_assign_base&&) = default;

  constexpr __cb_move_assign_base& operator=(const __cb_move_assign_base&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_move_assign_base& operator=(__cb_move_assign_base&& __other) noexcept(
    is_nothrow_move_assignable_v<_Tp> && is_nothrow_move_constructible_v<_Tp>)
  {
    if (this == _CUDA_VSTD::addressof(__other)) {
      return *this;
    }

    if (this->__engaged_ && __other.__engaged_) {
      this->__val_ = _CUDA_VSTD::move(__other.__val_);
    } else if (this->__engaged_) {
      if constexpr(!is_trivially_destructible_v<_Tp>) {
        this->__val_.~_Tp();
      }
      this->__engaged_ = false;
    } else if (__other.__engaged_) {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
      this->__engaged_ = true;
    }
    return *this;
  }
};

template <class _Tp>
struct __cb_move_assign_base<_Tp, false, false> : __cb_copy_assign_base<_Tp> {
  using __base = __cb_copy_assign_base<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_move_assign_base(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __cb_move_assign_base() noexcept = default;
  constexpr __cb_move_assign_base(const __cb_move_assign_base&) = default;
  constexpr __cb_move_assign_base(__cb_move_assign_base&&) = default;

  constexpr __cb_move_assign_base& operator=(const __cb_move_assign_base&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_move_assign_base& operator=(__cb_move_assign_base&& __other) noexcept(is_nothrow_move_constructible_v<_Tp>)
  {
    if (this == _CUDA_VSTD::addressof(__other)) {
      return *this;
    }

    if constexpr(!is_trivially_destructible_v<_Tp>) {
      if (this->__engaged_) {
        this->__val_.~_Tp();
      }
    }

    if (__other.__engaged_) {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
    }
    this->__engaged_ = __other.__engaged_;
    return *this;
  }
};

// This partial specialization implements an optimization for when we know we don't need to store
// an empty state to represent failure to perform an assignment. For copy-assignment, this happens:
//
// 1. If the type is copyable (which includes copy-assignment), we can use the type's own assignment operator
//    directly and avoid using _CUDA_VSTD::optional.
// 2. If the type is not copyable, but it is nothrow-copy-constructible, then we can implement assignment as
//    destroy-and-then-construct and we know it will never fail, so we don't need an empty state.
//
// The exact same reasoning can be applied for move-assignment, with copyable replaced by movable and
// nothrow-copy-constructible replaced by nothrow-move-constructible. This specialization is enabled
// whenever we can apply any of these optimizations for both the copy assignment and the move assignment
// operator.
template<class _Tp>
_LIBCUDACXX_CONCEPT __doesnt_need_empty_state_for_copy = copyable<_Tp> || is_nothrow_copy_constructible_v<_Tp>;

template<class _Tp>
_LIBCUDACXX_CONCEPT __doesnt_need_empty_state_for_move = movable<_Tp> || is_nothrow_move_constructible_v<_Tp>;

template<class _Tp, bool = copyable<_Tp>>
struct __cb_copy_assign_base_simple {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;

  constexpr __cb_copy_assign_base_simple() = default;

  _LIBCUDACXX_TEMPLATE(class ..._Args)
    (requires is_constructible_v<_Tp, _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __cb_copy_assign_base_simple(in_place_t, _Args&& ...__args)
    noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
    : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
};

template<class _Tp>
struct __cb_copy_assign_base_simple<_Tp, false> {
  /* _LIBCUDACXX_NO_UNIQUE_ADDRESS */ _Tp __val_;

  constexpr __cb_copy_assign_base_simple() = default;

  _LIBCUDACXX_TEMPLATE(class ..._Args)
    (requires is_constructible_v<_Tp, _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __cb_copy_assign_base_simple(in_place_t, _Args&& ...__args)
    noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
    : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  constexpr __cb_copy_assign_base_simple(const __cb_copy_assign_base_simple&) = default;
  constexpr __cb_copy_assign_base_simple(__cb_copy_assign_base_simple&&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_copy_assign_base_simple& operator=(const __cb_copy_assign_base_simple& __other) noexcept {
    static_assert(is_nothrow_copy_constructible_v<_Tp>);
    if (this != _CUDA_VSTD::addressof(__other)) {
      if constexpr(!is_trivially_destructible_v<_Tp>) {
        __val_.~_Tp();
      }
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__val_), __other.__val_);
    }
    return *this;
  }

  constexpr __cb_copy_assign_base_simple& operator=(__cb_copy_assign_base_simple&&) = default;
};

template<class _Tp, bool = movable<_Tp>>
struct __cb_move_assign_base_simple : public __cb_copy_assign_base_simple<_Tp> {
  using __base = __cb_copy_assign_base_simple<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  __cb_move_assign_base_simple() = default;

  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_move_assign_base_simple(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template<class _Tp>
struct __cb_move_assign_base_simple<_Tp, false> : public __cb_copy_assign_base_simple<_Tp> {
  using __base = __cb_copy_assign_base_simple<_Tp>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __cb_move_assign_base_simple(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __cb_move_assign_base_simple() = default;
  constexpr __cb_move_assign_base_simple(const __cb_move_assign_base_simple&) = default;
  constexpr __cb_move_assign_base_simple(__cb_move_assign_base_simple&&) = default;

  constexpr __cb_move_assign_base_simple& operator=(const __cb_move_assign_base_simple&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __cb_move_assign_base_simple& operator=(__cb_move_assign_base_simple&& __other) noexcept {
    static_assert(is_nothrow_copy_constructible_v<_Tp>);
    if (this != _CUDA_VSTD::addressof(__other)) {
      if constexpr(!is_trivially_destructible_v<_Tp>) {
        this->__val_.~_Tp();
      }
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
    }
    return *this;
  }
};

template <class _Tp>
using __cb_sfinae_ctor_base_t = __sfinae_ctor_base<
    is_copy_constructible<_Tp>::value,
    is_move_constructible<_Tp>::value
>;

template <class _Tp>
using __cb_sfinae_assign_base_t = __sfinae_assign_base<
    (is_copy_constructible<_Tp>::value && is_copy_assignable<_Tp>::value),
    (is_move_constructible<_Tp>::value && is_move_assignable<_Tp>::value)
>;

// Primary template - uses _CUDA_VSTD::optional and introduces an empty state in case assignment fails.
template<class _Tp, bool = __copy_constructible_object<_Tp>,
                    bool = __doesnt_need_empty_state_for_copy<_Tp>
                        && __doesnt_need_empty_state_for_move<_Tp>>
#if _LIBCUDACXX_STD_VER > 17
  requires __copy_constructible_object<_Tp>
#endif // _LIBCUDACXX_STD_VER > 17
class __copyable_box;

template<class _Tp>
#if _LIBCUDACXX_STD_VER > 17
  requires __copy_constructible_object<_Tp>
#endif // _LIBCUDACXX_STD_VER > 17
class __copyable_box<_Tp, true, false> : public __cb_move_assign_base<_Tp>
                                       , public __cb_sfinae_ctor_base_t<_Tp>
                                       , public __cb_sfinae_assign_base_t<_Tp> {
  using __base = __cb_move_assign_base<_Tp>;

public:
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    (requires default_initializable<_Tp2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __copyable_box() noexcept(is_nothrow_default_constructible_v<_Tp>)
    : __base(in_place)
  {}

  _LIBCUDACXX_TEMPLATE(class ..._Args)
    (requires is_constructible_v<_Tp, _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __copyable_box(in_place_t, _Args&& ...__args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
    : __base(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp const& operator*() const noexcept { return this->__val_; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp& operator*() noexcept { return this->__val_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp *operator->() const noexcept { return _CUDA_VSTD::addressof(this->__val_); }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp *operator->() noexcept { return _CUDA_VSTD::addressof(this->__val_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __engaged_ue() const noexcept { return this->__engaged_; }
};

template<class _Tp>
#if _LIBCUDACXX_STD_VER > 17
  requires __copy_constructible_object<_Tp>
#endif // _LIBCUDACXX_STD_VER > 17
class __copyable_box<_Tp, true, true> : public __cb_move_assign_base_simple<_Tp> {
  using __base = __cb_move_assign_base_simple<_Tp>;

public:
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    (requires default_initializable<_Tp2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __copyable_box() noexcept(is_nothrow_default_constructible_v<_Tp>)
    : __base()
  {}

  _LIBCUDACXX_TEMPLATE(class ..._Args)
    (requires is_constructible_v<_Tp, _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __copyable_box(in_place_t, _Args&& ...__args)
    noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
    : __base(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp const& operator*() const noexcept { return this->__val_; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp& operator*() noexcept { return this->__val_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp *operator->() const noexcept { return _CUDA_VSTD::addressof(this->__val_); }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp *operator->() noexcept { return _CUDA_VSTD::addressof(this->__val_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __engaged_ue() const noexcept { return true; }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_COPYABLE_BOX_H
