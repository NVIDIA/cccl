//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___OPTIONAL_OPTIONAL_H
#define _CUDA_STD___OPTIONAL_OPTIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__fwd/optional.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__optional/bad_optional_access.h>
#include <cuda/std/__optional/nullopt.h>
#include <cuda/std/__optional/optional_base.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/reference_constructs_from_temporary.h>
#include <cuda/std/__type_traits/reference_converts_from_temporary.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // suppress bogus unreachable code warning

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Constraints
template <class _Tp, class _Up, class _Opt = optional<_Up>>
using __opt_check_constructible_from_opt =
  _Or<is_constructible<_Tp, _Opt&>,
      is_constructible<_Tp, _Opt const&>,
      is_constructible<_Tp, _Opt&&>,
      is_constructible<_Tp, _Opt const&&>,
      is_convertible<_Opt&, _Tp>,
      is_convertible<_Opt const&, _Tp>,
      is_convertible<_Opt&&, _Tp>,
      is_convertible<_Opt const&&, _Tp>>;

template <class _Tp, class _Up, class _Opt = optional<_Up>>
using __opt_check_assignable_from_opt =
  _Or<is_assignable<_Tp&, _Opt&>,
      is_assignable<_Tp&, _Opt const&>,
      is_assignable<_Tp&, _Opt&&>,
      is_assignable<_Tp&, _Opt const&&>>;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_implictly_constructible = is_constructible_v<_Tp, _Up> && is_convertible_v<_Up, _Tp>;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_explictly_constructible = is_constructible_v<_Tp, _Up> && !is_convertible_v<_Up, _Tp>;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_constructible_from_U =
  !is_same_v<remove_cvref_t<_Up>, in_place_t> && !is_same_v<remove_cvref_t<_Up>, optional<_Tp>>;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_constructible_from_opt =
  !is_same_v<_Up, _Tp> && !__opt_check_constructible_from_opt<_Tp, _Up>::value;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_assignable = is_constructible_v<_Tp, _Up> && is_assignable_v<_Tp&, _Up>;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_assignable_from_U =
  !is_same_v<remove_cvref_t<_Up>, optional<_Tp>> && (!is_same_v<remove_cvref_t<_Up>, _Tp> || !is_scalar_v<_Tp>);

template <class _Tp, class _Up>
inline constexpr bool __opt_is_assignable_from_opt =
  !is_same_v<_Up, _Tp> && !__opt_check_constructible_from_opt<_Tp, _Up>::value
  && !__opt_check_assignable_from_opt<_Tp, _Up>::value;

template <class _Tp>
class optional : private __optional_move_assign_base<_Tp>
{
  using __base = __optional_move_assign_base<_Tp>;

  template <class>
  friend class optional;

public:
  using value_type = _Tp;

private:
  // Disable the reference extension using this static assert.
  static_assert(!is_same_v<remove_cvref_t<value_type>, in_place_t>,
                "instantiation of optional with in_place_t is ill-formed");
  static_assert(!is_same_v<remove_cvref_t<value_type>, nullopt_t>,
                "instantiation of optional with nullopt_t is ill-formed");
  static_assert(!is_reference_v<value_type>,
                "instantiation of optional with a reference type is ill-formed. Define "
                "CCCL_ENABLE_OPTIONAL_REF to enable it as a non-standard extension");
  static_assert(is_destructible_v<value_type>, "instantiation of optional with a non-destructible type is ill-formed");
  static_assert(!is_array_v<value_type>, "instantiation of optional with an array type is ill-formed");

public:
  _CCCL_API constexpr optional() noexcept {}
  _CCCL_HIDE_FROM_ABI constexpr optional(const optional&) = default;
  _CCCL_HIDE_FROM_ABI constexpr optional(optional&&)      = default;
  _CCCL_API constexpr optional(nullopt_t) noexcept {}

  _CCCL_TEMPLATE(class _In_place_t, class... _Args)
  _CCCL_REQUIRES(is_same_v<_In_place_t, in_place_t> _CCCL_AND is_constructible_v<value_type, _Args...>)
  _CCCL_API constexpr explicit optional(_In_place_t, _Args&&... __args)
      : __base(in_place, ::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_TEMPLATE(class _Up, class... _Args)
  _CCCL_REQUIRES(is_constructible_v<value_type, initializer_list<_Up>&, _Args...>)
  _CCCL_API constexpr explicit optional(in_place_t, initializer_list<_Up> __il, _Args&&... __args)
      : __base(in_place, __il, ::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_TEMPLATE(class _Up = value_type)
  _CCCL_REQUIRES(__opt_is_constructible_from_U<_Tp, _Up> _CCCL_AND __opt_is_implictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr optional(_Up&& __v)
      : __base(in_place, ::cuda::std::forward<_Up>(__v))
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_U<_Tp, _Up> _CCCL_AND __opt_is_explictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr explicit optional(_Up&& __v)
      : __base(in_place, ::cuda::std::forward<_Up>(__v))
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_implictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr optional(const optional<_Up>& __v)
  {
    this->__construct_from(__v);
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_explictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr explicit optional(const optional<_Up>& __v)
  {
    this->__construct_from(__v);
  }

#ifdef CCCL_ENABLE_OPTIONAL_REF
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((is_same_v<remove_cv_t<_Tp>, bool> || __opt_is_constructible_from_opt<_Tp, _Up>)
                   _CCCL_AND __opt_is_implictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr optional(const optional<_Up&>& __v)
  {
    this->__construct_from(__v);
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((is_same_v<remove_cv_t<_Tp>, bool> || __opt_is_constructible_from_opt<_Tp, _Up>)
                   _CCCL_AND __opt_is_explictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr explicit optional(const optional<_Up&>& __v)
  {
    this->__construct_from(__v);
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND
                   __opt_is_implictly_constructible<_Tp, _Up> _CCCL_AND(!is_reference_v<_Up>))
  _CCCL_API constexpr optional(optional<_Up>&& __v)
  {
    this->__construct_from(::cuda::std::move(__v));
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND
                   __opt_is_explictly_constructible<_Tp, _Up> _CCCL_AND(!is_reference_v<_Up>))
  _CCCL_API constexpr explicit optional(optional<_Up>&& __v)
  {
    this->__construct_from(::cuda::std::move(__v));
  }
#else // ^^^ CCCL_ENABLE_OPTIONAL_REF ^^^ / vvv !CCCL_ENABLE_OPTIONAL_REF vvv
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_implictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr optional(optional<_Up>&& __v)
  {
    this->__construct_from(::cuda::std::move(__v));
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_explictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr explicit optional(optional<_Up>&& __v)
  {
    this->__construct_from(::cuda::std::move(__v));
  }
#endif // !CCCL_ENABLE_OPTIONAL_REF

private:
  template <class _Fp, class... _Args>
  _CCCL_API constexpr explicit optional(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
      : __base(__optional_construct_from_invoke_tag{},
               ::cuda::std::forward<_Fp>(__f),
               ::cuda::std::forward<_Args>(__args)...)
  {}

public:
  _CCCL_API constexpr optional& operator=(nullopt_t) noexcept
  {
    reset();
    return *this;
  }

  constexpr optional& operator=(const optional&) = default;
  constexpr optional& operator=(optional&&)      = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up = value_type)
  _CCCL_REQUIRES(__opt_is_assignable_from_U<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, _Up>)
  _CCCL_API constexpr optional& operator=(_Up&& __v)
  {
    if (this->has_value())
    {
      this->__get() = ::cuda::std::forward<_Up>(__v);
    }
    else
    {
      this->__construct(::cuda::std::forward<_Up>(__v));
    }
    return *this;
  }

#ifdef CCCL_ENABLE_OPTIONAL_REF
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!is_reference_v<_Up>)
                   _CCCL_AND __opt_is_assignable_from_opt<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, const _Up&>)
  _CCCL_API constexpr optional& operator=(const optional<_Up>& __v)
  {
    this->__assign_from(__v);
    return *this;
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(
    is_reference_v<_Up> _CCCL_AND __opt_is_assignable_from_opt<_Tp, _Up&> _CCCL_AND __opt_is_assignable<_Tp, _Up&>)
  _CCCL_API constexpr optional& operator=(const optional<_Up>& __v)
  {
    this->__assign_from(__v);
    return *this;
  }
#else // ^^^ CCCL_ENABLE_OPTIONAL_REF ^^^ / vvv !CCCL_ENABLE_OPTIONAL_REF vvv
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_assignable_from_opt<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, const _Up&>)
  _CCCL_API constexpr optional& operator=(const optional<_Up>& __v)
  {
    this->__assign_from(__v);
    return *this;
  }
#endif // !CCCL_ENABLE_OPTIONAL_REF

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_assignable_from_opt<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, _Up>)
  _CCCL_API constexpr optional& operator=(optional<_Up>&& __v)
  {
    this->__assign_from(::cuda::std::move(__v));
    return *this;
  }

  template <class... _Args, enable_if_t<is_constructible_v<value_type, _Args...>, int> = 0>
  _CCCL_API constexpr _Tp& emplace(_Args&&... __args)
  {
    reset();
    this->__construct(::cuda::std::forward<_Args>(__args)...);
    return this->__get();
  }

  template <class _Up,
            class... _Args,
            enable_if_t<is_constructible_v<value_type, initializer_list<_Up>&, _Args...>, int> = 0>
  _CCCL_API constexpr _Tp& emplace(initializer_list<_Up> __il, _Args&&... __args)
  {
    reset();
    this->__construct(__il, ::cuda::std::forward<_Args>(__args)...);
    return this->__get();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr void
  swap(optional& __opt) noexcept(is_nothrow_move_constructible_v<value_type> && is_nothrow_swappable_v<value_type>)
  {
    if (this->has_value() == __opt.has_value())
    {
      using ::cuda::std::swap;
      if (this->has_value())
      {
        swap(this->__get(), __opt.__get());
      }
    }
    else
    {
      if (this->has_value())
      {
        __opt.__construct(::cuda::std::move(this->__get()));
        reset();
      }
      else
      {
        this->__construct(::cuda::std::move(__opt.__get()));
        __opt.reset();
      }
    }
  }

  _CCCL_API constexpr add_pointer_t<value_type const> operator->() const
  {
    _CCCL_ASSERT(this->has_value(), "optional operator-> called on a disengaged value");
    return ::cuda::std::addressof(this->__get());
  }

  _CCCL_API constexpr add_pointer_t<value_type> operator->()
  {
    _CCCL_ASSERT(this->has_value(), "optional operator-> called on a disengaged value");
    return ::cuda::std::addressof(this->__get());
  }

  _CCCL_API constexpr const value_type& operator*() const& noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  _CCCL_API constexpr value_type& operator*() & noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  _CCCL_API constexpr value_type&& operator*() && noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return ::cuda::std::move(this->__get());
  }

  _CCCL_API constexpr const value_type&& operator*() const&& noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return ::cuda::std::move(this->__get());
  }

  _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return has_value();
  }

  using __base::__get;
  using __base::has_value;

  _CCCL_API constexpr value_type const& value() const&
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return this->__get();
  }

  _CCCL_API constexpr value_type& value() &
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return this->__get();
  }

  _CCCL_API constexpr value_type&& value() &&
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return ::cuda::std::move(this->__get());
  }

  _CCCL_API constexpr value_type const&& value() const&&
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return ::cuda::std::move(this->__get());
  }

  template <class _Up>
  _CCCL_API constexpr value_type value_or(_Up&& __v) const&
  {
    static_assert(is_copy_constructible_v<value_type>, "optional<T>::value_or: T must be copy constructible");
    static_assert(is_convertible_v<_Up, value_type>, "optional<T>::value_or: U must be convertible to T");
    return this->has_value() ? this->__get() : static_cast<value_type>(::cuda::std::forward<_Up>(__v));
  }

  template <class _Up>
  _CCCL_API constexpr value_type value_or(_Up&& __v) &&
  {
    static_assert(is_move_constructible_v<value_type>, "optional<T>::value_or: T must be move constructible");
    static_assert(is_convertible_v<_Up, value_type>, "optional<T>::value_or: U must be convertible to T");
    return this->has_value() ? ::cuda::std::move(this->__get())
                             : static_cast<value_type>(::cuda::std::forward<_Up>(__v));
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) &
  {
    using _Up = invoke_result_t<_Func, value_type&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(value()) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return ::cuda::std::invoke(::cuda::std::forward<_Func>(__f), this->__get());
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) const&
  {
    using _Up = invoke_result_t<_Func, const value_type&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(value()) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return ::cuda::std::invoke(::cuda::std::forward<_Func>(__f), this->__get());
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) &&
  {
    using _Up = invoke_result_t<_Func, value_type&&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(std::move(value())) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return ::cuda::std::invoke(::cuda::std::forward<_Func>(__f), ::cuda::std::move(this->__get()));
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) const&&
  {
    using _Up = invoke_result_t<_Func, const value_type&&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(std::move(value())) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return ::cuda::std::invoke(::cuda::std::forward<_Func>(__f), ::cuda::std::move(this->__get()));
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) &
  {
    using _Up = remove_cv_t<invoke_result_t<_Func, value_type&>>;
    static_assert(!is_array_v<_Up>, "Result of f(value()) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(value()) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(value()) should not be std::nullopt_t");
    static_assert(is_object_v<_Up>, "Result of f(value()) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(__optional_construct_from_invoke_tag{}, ::cuda::std::forward<_Func>(__f), this->__get());
    }
    return optional<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) const&
  {
    using _Up = remove_cv_t<invoke_result_t<_Func, const value_type&>>;
    static_assert(!is_array_v<_Up>, "Result of f(value()) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(value()) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(value()) should not be std::nullopt_t");
    static_assert(is_object_v<_Up>, "Result of f(value()) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(__optional_construct_from_invoke_tag{}, ::cuda::std::forward<_Func>(__f), this->__get());
    }
    return optional<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) &&
  {
    using _Up = remove_cv_t<invoke_result_t<_Func, value_type&&>>;
    static_assert(!is_array_v<_Up>, "Result of f(std::move(value())) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(std::move(value())) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(std::move(value())) should not be std::nullopt_t");
    static_assert(is_object_v<_Up>, "Result of f(std::move(value())) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(
        __optional_construct_from_invoke_tag{}, ::cuda::std::forward<_Func>(__f), ::cuda::std::move(this->__get()));
    }
    return optional<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) const&&
  {
    using _Up = remove_cvref_t<invoke_result_t<_Func, const value_type&&>>;
    static_assert(!is_array_v<_Up>, "Result of f(std::move(value())) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(std::move(value())) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(std::move(value())) should not be std::nullopt_t");
    static_assert(is_object_v<_Up>, "Result of f(std::move(value())) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(
        __optional_construct_from_invoke_tag{}, ::cuda::std::forward<_Func>(__f), ::cuda::std::move(this->__get()));
    }
    return optional<_Up>();
  }

  _CCCL_TEMPLATE(class _Func, class _Tp2 = _Tp)
  _CCCL_REQUIRES(invocable<_Func> _CCCL_AND is_copy_constructible_v<_Tp2>)
  _CCCL_API constexpr optional or_else(_Func&& __f) const&
  {
    static_assert(is_same_v<remove_cvref_t<invoke_result_t<_Func>>, optional>,
                  "Result of f() should be the same type as this optional");
    if (this->__engaged_)
    {
      return *this;
    }
    return ::cuda::std::forward<_Func>(__f)();
  }

  _CCCL_TEMPLATE(class _Func, class _Tp2 = _Tp)
  _CCCL_REQUIRES(invocable<_Func> _CCCL_AND is_move_constructible_v<_Tp2>)
  _CCCL_API constexpr optional or_else(_Func&& __f) &&
  {
    static_assert(is_same_v<remove_cvref_t<invoke_result_t<_Func>>, optional>,
                  "Result of f() should be the same type as this optional");
    if (this->__engaged_)
    {
      return ::cuda::std::move(*this);
    }
    return ::cuda::std::forward<_Func>(__f)();
  }

  using __base::reset;
};

template <class _Tp>
_CCCL_HOST_DEVICE optional(_Tp) -> optional<_Tp>;

// Comparisons between optionals
_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() == declval<const _Up&>()), bool>, bool>
operator==(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (static_cast<bool>(__x) != static_cast<bool>(__y))
  {
    return false;
  }
  if (!static_cast<bool>(__x))
  {
    return true;
  }
  return *__x == *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() != declval<const _Up&>()), bool>, bool>
operator!=(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (static_cast<bool>(__x) != static_cast<bool>(__y))
  {
    return true;
  }
  if (!static_cast<bool>(__x))
  {
    return false;
  }
  return *__x != *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() < declval<const _Up&>()), bool>, bool>
operator<(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__y))
  {
    return false;
  }
  if (!static_cast<bool>(__x))
  {
    return true;
  }
  return *__x < *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() > declval<const _Up&>()), bool>, bool>
operator>(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__x))
  {
    return false;
  }
  if (!static_cast<bool>(__y))
  {
    return true;
  }
  return *__x > *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() <= declval<const _Up&>()), bool>, bool>
operator<=(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__x))
  {
    return true;
  }
  if (!static_cast<bool>(__y))
  {
    return false;
  }
  return *__x <= *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() >= declval<const _Up&>()), bool>, bool>
operator>=(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__y))
  {
    return true;
  }
  if (!static_cast<bool>(__x))
  {
    return false;
  }
  return *__x >= *__y;
}

// Comparisons with nullopt
template <class _Tp>
_CCCL_API constexpr bool operator==(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return !static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator==(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return !static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator!=(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator!=(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator<(const optional<_Tp>&, nullopt_t) noexcept
{
  return false;
}

template <class _Tp>
_CCCL_API constexpr bool operator<(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator<=(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return !static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator<=(nullopt_t, const optional<_Tp>&) noexcept
{
  return true;
}

template <class _Tp>
_CCCL_API constexpr bool operator>(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator>(nullopt_t, const optional<_Tp>&) noexcept
{
  return false;
}

template <class _Tp>
_CCCL_API constexpr bool operator>=(const optional<_Tp>&, nullopt_t) noexcept
{
  return true;
}

template <class _Tp>
_CCCL_API constexpr bool operator>=(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return !static_cast<bool>(__x);
}

// Comparisons with T
_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() == declval<const _Up&>()), bool>, bool>
operator==(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x == __v : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() == declval<const _Up&>()), bool>, bool>
operator==(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v == *__x : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() != declval<const _Up&>()), bool>, bool>
operator!=(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x != __v : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() != declval<const _Up&>()), bool>, bool>
operator!=(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v != *__x : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() < declval<const _Up&>()), bool>, bool>
operator<(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x < __v : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() < declval<const _Up&>()), bool>, bool>
operator<(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v < *__x : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() <= declval<const _Up&>()), bool>, bool>
operator<=(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x <= __v : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() <= declval<const _Up&>()), bool>, bool>
operator<=(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v <= *__x : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() > declval<const _Up&>()), bool>, bool>
operator>(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x > __v : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() > declval<const _Up&>()), bool>, bool>
operator>(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v > *__x : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() >= declval<const _Up&>()), bool>, bool>
operator>=(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x >= __v : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<is_convertible_v<decltype(declval<const _Tp&>() >= declval<const _Up&>()), bool>, bool>
operator>=(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v >= *__x : true;
}

template <class _Tp>
_CCCL_API constexpr enable_if_t<
#ifdef CCCL_ENABLE_OPTIONAL_REF
  is_reference_v<_Tp> ||
#endif // CCCL_ENABLE_OPTIONAL_REF
    (is_move_constructible_v<_Tp> && is_swappable_v<_Tp>),
  void>
swap(optional<_Tp>& __x, optional<_Tp>& __y) noexcept(noexcept(__x.swap(__y)))
{
  __x.swap(__y);
}

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___OPTIONAL_OPTIONAL_H
