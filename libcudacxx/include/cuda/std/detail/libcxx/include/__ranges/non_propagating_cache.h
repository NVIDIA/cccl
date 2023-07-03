// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_NON_PROPAGATING_CACHE_H
#define _LIBCUDACXX___RANGES_NON_PROPAGATING_CACHE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/concepts.h"        // indirectly_readable
#include "../__iterator/iterator_traits.h" // iter_reference_t
#include "../__memory/addressof.h"
#include "../__memory/construct_at.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_trivially_destructible.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _LIBCUDACXX_STD_VER > 14
  struct __forward_tag { };
  // This helper class is needed to perform copy and move elision when
  // constructing the contained type from an iterator.
  template<class _Tp>
  struct __wrapper {
    template<class ..._Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    explicit __wrapper(__forward_tag, _Args&& ...__args) : __t_(_CUDA_VSTD::forward<_Args>(__args)...) { }
    _Tp __t_;
  };

  template<class _Tp, class = void>
  struct __npc_destruct_base {
    union
    {
        _LIBCUDACXX_NO_UNIQUE_ADDRESS __wrapper<_Tp> __val_;
    };
    bool __engaged_ = false;

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __npc_destruct_base() noexcept {}

    template<class ..._Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __npc_destruct_base(__forward_tag, _Args&& ...__args)
      noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(__forward_tag{}, _CUDA_VSTD::forward<_Args>(__args)...), __engaged_(true)
    { }

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    ~__npc_destruct_base() noexcept {
        if (__engaged_) {
            __val_.~_Tp();
        }
    }

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    void __reset() noexcept {
        if (__engaged_) {
            __val_.~_Tp();
            __engaged_ = false;
        }
    }
  };

  template <class _Tp>
  struct __npc_destruct_base<_Tp, enable_if_t<is_trivially_destructible_v<_Tp>>> {
    union
    {
        _LIBCUDACXX_NO_UNIQUE_ADDRESS __wrapper<_Tp> __val_;
    };
    bool __engaged_ = false;

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __npc_destruct_base() noexcept {}

    template<class ..._Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __npc_destruct_base(__forward_tag, _Args&& ...__args)
      noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(__forward_tag{}, _CUDA_VSTD::forward<_Args>(__args)...), __engaged_(true)
    {}

    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr void __reset() noexcept {
        if (__engaged_) {
            __engaged_ = false;
        }
    }
  };

  // __non_propagating_cache is a helper type that allows storing an optional value in it,
  // but which does not copy the source's value when it is copy constructed/assigned to,
  // and which resets the source's value when it is moved-from.
  //
  // This type is used as an implementation detail of some views that need to cache the
  // result of `begin()` in order to provide an amortized O(1) begin() method. Typically,
  // we don't want to propagate the value of the cache upon copy because the cached iterator
  // may refer to internal details of the source view.
  template<class _Tp, enable_if_t<is_object_v<_Tp>, int> = 0>
  class _LIBCUDACXX_TEMPLATE_VIS __non_propagating_cache : public __npc_destruct_base<_Tp> {
  public:
    using __base = __npc_destruct_base<_Tp>;
#if defined(_LIBCUDACXX_COMPILER_NVRTC) // nvbug 3961621
    template<class ..._Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __non_propagating_cache(_Args&&... __args)
      noexcept(noexcept(__base(cuda::std::declval<_Args>()...)))
      : __base(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
#else
    using __base::__base;
#endif

    __non_propagating_cache() = default;

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __non_propagating_cache(const __non_propagating_cache&) noexcept : __base() { }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __non_propagating_cache(__non_propagating_cache&& __other) noexcept : __base()
    {
      __other.__reset();
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __non_propagating_cache& operator=(const __non_propagating_cache& __other) noexcept {
      if (this != _CUDA_VSTD::addressof(__other)) {
        this->__reset();
      }
      return *this;
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __non_propagating_cache& operator=(__non_propagating_cache&& __other) noexcept {
      this->__reset();
      __other.__reset();
      return *this;
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp& operator*() { return this->__val_.__t_; }
    _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp const& operator*() const { return this->__val_.__t_; }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr bool __has_value() const noexcept { return this->__engaged_; }

    template<class ..._Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    _Tp& __emplace(_Args&& ...__args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>) {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __forward_tag{}, _CUDA_VSTD::forward<_Args>(__args)...);
      this->__engaged_ = true;
      return this->__val_.__t_;
    }
  };

  struct __empty_cache { };

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_NON_PROPAGATING_CACHE_H
