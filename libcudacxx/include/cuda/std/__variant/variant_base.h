//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VARIANT_BASE_H
#define _CUDA_STD___VARIANT_VARIANT_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/variant.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/__variant/variant_access.h>
#include <cuda/std/__variant/variant_traits.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __variant_detail
{
// NOTE: we need to define all special member functions, because NVCC has issues with host only types
template <size_t _Index, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __alt
{
  using __value_type = _Tp;

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API explicit constexpr __alt(in_place_t, _Args&&... __args)
      : __value(::cuda::std::forward<_Args>(__args)...)
  {}
  _CCCL_EXEC_CHECK_DISABLE
  constexpr __alt(const __alt&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  constexpr __alt(__alt&&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  constexpr __alt& operator=(const __alt&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  constexpr __alt& operator=(__alt&&) = default;

  _CCCL_EXEC_CHECK_DISABLE
  ~__alt() = default;

  __value_type __value;
};

template <_Trait _DestructibleTrait, size_t _Index, class... _Types>
union _CCCL_TYPE_VISIBILITY_DEFAULT __union;

template <_Trait _DestructibleTrait, size_t _Index>
union _CCCL_TYPE_VISIBILITY_DEFAULT __union<_DestructibleTrait, _Index>
{};

#define _LIBCUDACXX_VARIANT_UNION_BODY(destructible_trait)                       \
                                                                                 \
private:                                                                         \
  char __dummy;                                                                  \
  __alt<_Index, _Tp> __head_;                                                    \
  __union<destructible_trait, _Index + 1, _Types...> __tail_;                    \
                                                                                 \
  friend struct __access::__union;                                               \
                                                                                 \
public:                                                                          \
  _CCCL_API explicit constexpr __union(__valueless_t) noexcept                   \
      : __dummy{}                                                                \
  {}                                                                             \
                                                                                 \
  template <class... _Args>                                                      \
  _CCCL_API explicit constexpr __union(in_place_index_t<0>, _Args&&... __args)   \
      : __head_(in_place, ::cuda::std::forward<_Args>(__args)...)                \
  {}                                                                             \
                                                                                 \
  template <size_t _Ip, class... _Args>                                          \
  _CCCL_API explicit constexpr __union(in_place_index_t<_Ip>, _Args&&... __args) \
      : __tail_(in_place_index<_Ip - 1>, ::cuda::std::forward<_Args>(__args)...) \
  {}                                                                             \
                                                                                 \
  _CCCL_HIDE_FROM_ABI __union(const __union&)            = default;              \
  _CCCL_HIDE_FROM_ABI __union(__union&&)                 = default;              \
  _CCCL_HIDE_FROM_ABI __union& operator=(const __union&) = default;              \
  _CCCL_HIDE_FROM_ABI __union& operator=(__union&&)      = default;

template <size_t _Index, class _Tp, class... _Types>
union _CCCL_TYPE_VISIBILITY_DEFAULT __union<_Trait::_TriviallyAvailable, _Index, _Tp, _Types...>
{
  _LIBCUDACXX_VARIANT_UNION_BODY(_Trait::_TriviallyAvailable)
  _CCCL_HIDE_FROM_ABI ~__union() = default;
};

template <size_t _Index, class _Tp, class... _Types>
union _CCCL_TYPE_VISIBILITY_DEFAULT __union<_Trait::_Available, _Index, _Tp, _Types...>
{
  _LIBCUDACXX_VARIANT_UNION_BODY(_Trait::_Available)
  _CCCL_API ~__union() {}
};

template <size_t _Index, class _Tp, class... _Types>
union _CCCL_TYPE_VISIBILITY_DEFAULT __union<_Trait::_Unavailable, _Index, _Tp, _Types...>
{
  _LIBCUDACXX_VARIANT_UNION_BODY(_Trait::_Unavailable)
  _CCCL_API ~__union() = delete;
};

#undef _LIBCUDACXX_VARIANT_UNION_BODY

template <_Trait _DestructibleTrait, class... _Types>
class _CCCL_TYPE_VISIBILITY_DEFAULT __base
{
public:
  using __index_t = __variant_index_t<sizeof...(_Types)>;

  _CCCL_API explicit constexpr __base(__valueless_t __tag) noexcept
      : __data_(__tag)
      , __index_(__variant_npos<__index_t>)
  {}

  template <size_t _Ip, class... _Args>
  _CCCL_API explicit constexpr __base(in_place_index_t<_Ip>, _Args&&... __args)
      : __data_(in_place_index<_Ip>, ::cuda::std::forward<_Args>(__args)...)
      , __index_(_Ip)
  {}

  [[nodiscard]] _CCCL_API constexpr bool valueless_by_exception() const noexcept
  {
    return index() == variant_npos;
  }

  [[nodiscard]] _CCCL_API constexpr size_t index() const noexcept
  {
    return __index_ == __variant_npos<__index_t> ? variant_npos : __index_;
  }

protected:
  [[nodiscard]] _CCCL_API constexpr auto&& __as_base() &
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr auto&& __as_base() &&
  {
    return ::cuda::std::move(*this);
  }

  [[nodiscard]] _CCCL_API constexpr auto&& __as_base() const&
  {
    return *this;
  }

  _CCCL_API constexpr auto&& __as_base() const&&
  {
    return ::cuda::std::move(*this);
  }

  [[nodiscard]] _CCCL_API static constexpr size_t __size() noexcept
  {
    return sizeof...(_Types);
  }

  __union<_DestructibleTrait, 0, _Types...> __data_;
  __index_t __index_;

  friend struct __access::__base;
};

template <class _Traits, _Trait = _Traits::__destructible_trait>
class _CCCL_TYPE_VISIBILITY_DEFAULT __dtor;

#define _LIBCUDACXX_VARIANT_DESTRUCTOR_BODY(destructible_trait)   \
  using __base_type = __base<destructible_trait, _Types...>;      \
  using __index_t   = typename __base_type::__index_t;            \
                                                                  \
public:                                                           \
  using __base_type::__base_type;                                 \
  using __base_type::operator=;                                   \
                                                                  \
  _CCCL_HIDE_FROM_ABI __dtor(const __dtor&)            = default; \
  _CCCL_HIDE_FROM_ABI __dtor(__dtor&&)                 = default; \
  _CCCL_HIDE_FROM_ABI __dtor& operator=(const __dtor&) = default; \
  _CCCL_HIDE_FROM_ABI __dtor& operator=(__dtor&&)      = default;

template <class... _Types>
class _CCCL_TYPE_VISIBILITY_DEFAULT
__dtor<__traits<_Types...>, _Trait::_TriviallyAvailable> : public __base<_Trait::_TriviallyAvailable, _Types...>
{
  _LIBCUDACXX_VARIANT_DESTRUCTOR_BODY(_Trait::_TriviallyAvailable)
  _CCCL_HIDE_FROM_ABI ~__dtor() = default;

protected:
  _CCCL_API void __destroy() noexcept
  {
    this->__index_ = __variant_npos<__index_t>;
  }
};

template <class... _Types>
class _CCCL_TYPE_VISIBILITY_DEFAULT
__dtor<__traits<_Types...>, _Trait::_Available> : public __base<_Trait::_Available, _Types...>
{
  struct __visitor
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class _Alt>
    _CCCL_API void operator()(_Alt& __alt) const noexcept
    {
      using __alt_type = remove_cvref_t<decltype(__alt)>;
      __alt.~__alt_type();
    }
  };

  _LIBCUDACXX_VARIANT_DESTRUCTOR_BODY(_Trait::_Available)

  _CCCL_API ~__dtor() noexcept
  {
    __destroy();
  }

protected:
  _CCCL_API void __destroy() noexcept
  {
    if (!this->valueless_by_exception())
    {
      constexpr size_t __np = remove_cvref_t<__dtor>::__size();
      __destroy(integral_constant<size_t, __np - 1>{}, this->__index_);
    }
    this->__index_ = __variant_npos<__index_t>;
  }

private:
  template <size_t _CurrentIndex>
  _CCCL_API void __destroy(integral_constant<size_t, _CurrentIndex>, const size_t __index_) noexcept
  {
    if (__index_ == _CurrentIndex)
    {
      using __alt_type = remove_cvref_t<decltype(__access::__base::__get_alt<_CurrentIndex>(this->__as_base()))>;
      __access::__base::__get_alt<_CurrentIndex>(this->__as_base()).~__alt_type();
      return;
    }
    __destroy(integral_constant<size_t, _CurrentIndex - 1>{}, __index_);
  }
  _CCCL_API void __destroy(integral_constant<size_t, 0>, const size_t __index_) noexcept
  {
    if (__index_ == 0)
    {
      using __alt_type = remove_cvref_t<decltype(__access::__base::__get_alt<0>(this->__as_base()))>;
      __access::__base::__get_alt<0>(this->__as_base()).~__alt_type();
      return;
    }
    // We already checked that every variant has a value, so we should never reach this line
    _CCCL_UNREACHABLE();
  }
};

template <class... _Types>
class _CCCL_TYPE_VISIBILITY_DEFAULT
__dtor<__traits<_Types...>, _Trait::_Unavailable> : public __base<_Trait::_Unavailable, _Types...>
{
  _LIBCUDACXX_VARIANT_DESTRUCTOR_BODY(_Trait::_Unavailable)
  _CCCL_API ~__dtor() = delete;

protected:
  _CCCL_API void __destroy() noexcept = delete;
};

#undef _LIBCUDACXX_VARIANT_DESTRUCTOR_BODY

template <class _Traits>
class _CCCL_TYPE_VISIBILITY_DEFAULT __ctor : public __dtor<_Traits>
{
  using __base_type = __dtor<_Traits>;

  template <size_t _CurrentIndex, class _Rhs>
  _CCCL_API static constexpr void
  __generic_construct_impl(integral_constant<size_t, _CurrentIndex>, const size_t __index_, __ctor& __lhs, _Rhs&& __rhs)
  {
    if (__index_ == _CurrentIndex)
    {
      ::cuda::std::__construct_at(
        ::cuda::std::addressof(__access::__base::__get_alt<_CurrentIndex>(__lhs.__as_base())),
        in_place,
        __access::__base::__get_alt<_CurrentIndex>(::cuda::std::forward<_Rhs>(__rhs).__as_base()).__value);
      return;
    }
    __generic_construct_impl(
      integral_constant<size_t, _CurrentIndex - 1>{}, __index_, __lhs, ::cuda::std::forward<_Rhs>(__rhs));
  }

  template <class _Rhs>
  _CCCL_API static constexpr void
  __generic_construct_impl(integral_constant<size_t, 0>, const size_t __index_, __ctor& __lhs, _Rhs&& __rhs)
  {
    if (__index_ == 0)
    {
      ::cuda::std::__construct_at(
        ::cuda::std::addressof(__access::__base::__get_alt<0>(__lhs.__as_base())),
        in_place,
        __access::__base::__get_alt<0>(::cuda::std::forward<_Rhs>(__rhs).__as_base()).__value);
      return;
    }
    // We already checked that every variant has a value, so we should never reach this line
    _CCCL_UNREACHABLE();
  }

public:
  using __base_type::__base_type;
  using __base_type::operator=;

protected:
  template <size_t _Ip, class _Tp, class... _Args>
  _CCCL_API static _Tp& __construct_alt(__alt<_Ip, _Tp>& __a, _Args&&... __args)
  {
    ::cuda::std::__construct_at(::cuda::std::addressof(__a), in_place, ::cuda::std::forward<_Args>(__args)...);
    return __a.__value;
  }

  template <class _Rhs>
  _CCCL_API static void __generic_construct(__ctor& __lhs, _Rhs&& __rhs)
  {
    __lhs.__destroy();
    if (!__rhs.valueless_by_exception())
    {
      constexpr size_t __np = remove_cvref_t<__ctor>::__size();
      __generic_construct_impl(
        integral_constant<size_t, __np - 1>{}, __rhs.index(), __lhs, ::cuda::std::forward<_Rhs>(__rhs));
      __lhs.__index_ = static_cast<decltype(__lhs.__index_)>(__rhs.index());
    }
  }
};

template <class _Traits, _Trait = _Traits::__move_constructible_trait>
class _CCCL_TYPE_VISIBILITY_DEFAULT __move_constructor;

#define _LIBCUDACXX_VARIANT_MOVE_CONSTRUCTOR(move_constructible_trait, move_constructor)                 \
  template <class... _Types>                                                                             \
  class _CCCL_TYPE_VISIBILITY_DEFAULT                                                                    \
  __move_constructor<__traits<_Types...>, move_constructible_trait> : public __ctor<__traits<_Types...>> \
  {                                                                                                      \
    using __base_type = __ctor<__traits<_Types...>>;                                                     \
                                                                                                         \
  public:                                                                                                \
    using __base_type::__base_type;                                                                      \
    using __base_type::operator=;                                                                        \
                                                                                                         \
    _CCCL_HIDE_FROM_ABI __move_constructor(const __move_constructor&)            = default;              \
    _CCCL_HIDE_FROM_ABI ~__move_constructor()                                    = default;              \
    _CCCL_HIDE_FROM_ABI __move_constructor& operator=(const __move_constructor&) = default;              \
    _CCCL_HIDE_FROM_ABI __move_constructor& operator=(__move_constructor&&)      = default;              \
    move_constructor                                                                                     \
  }

_LIBCUDACXX_VARIANT_MOVE_CONSTRUCTOR(_Trait::_TriviallyAvailable,
                                     _CCCL_HIDE_FROM_ABI __move_constructor(__move_constructor&& __that) = default;);

_LIBCUDACXX_VARIANT_MOVE_CONSTRUCTOR(
  _Trait::_Available,
  _CCCL_API __move_constructor(__move_constructor&& __that) noexcept(
    __all<is_nothrow_move_constructible_v<_Types>...>::value) : __move_constructor(__valueless_t{}) {
    this->__generic_construct(*this, ::cuda::std::move(__that));
  });

_LIBCUDACXX_VARIANT_MOVE_CONSTRUCTOR(_Trait::_Unavailable, __move_constructor(__move_constructor&&) = delete;);

#undef _LIBCUDACXX_VARIANT_MOVE_CONSTRUCTOR

template <class _Traits, _Trait = _Traits::__copy_constructible_trait>
class _CCCL_TYPE_VISIBILITY_DEFAULT __copy_constructor;

#define _LIBCUDACXX_VARIANT_COPY_CONSTRUCTOR(copy_constructible_trait, copy_constructor)                             \
  template <class... _Types>                                                                                         \
  class _CCCL_TYPE_VISIBILITY_DEFAULT                                                                                \
  __copy_constructor<__traits<_Types...>, copy_constructible_trait> : public __move_constructor<__traits<_Types...>> \
  {                                                                                                                  \
    using __base_type = __move_constructor<__traits<_Types...>>;                                                     \
                                                                                                                     \
  public:                                                                                                            \
    using __base_type::__base_type;                                                                                  \
    using __base_type::operator=;                                                                                    \
                                                                                                                     \
    _CCCL_HIDE_FROM_ABI __copy_constructor(__copy_constructor&&)                 = default;                          \
    _CCCL_HIDE_FROM_ABI ~__copy_constructor()                                    = default;                          \
    _CCCL_HIDE_FROM_ABI __copy_constructor& operator=(const __copy_constructor&) = default;                          \
    _CCCL_HIDE_FROM_ABI __copy_constructor& operator=(__copy_constructor&&)      = default;                          \
    copy_constructor                                                                                                 \
  }

_LIBCUDACXX_VARIANT_COPY_CONSTRUCTOR(
  _Trait::_TriviallyAvailable, _CCCL_HIDE_FROM_ABI __copy_constructor(const __copy_constructor& __that) = default;);

_LIBCUDACXX_VARIANT_COPY_CONSTRUCTOR(
  _Trait::_Available,
  _CCCL_API __copy_constructor(const __copy_constructor& __that) : __copy_constructor(__valueless_t{}) {
    this->__generic_construct(*this, __that);
  });

_LIBCUDACXX_VARIANT_COPY_CONSTRUCTOR(_Trait::_Unavailable, __copy_constructor(const __copy_constructor&) = delete;);

#undef _LIBCUDACXX_VARIANT_COPY_CONSTRUCTOR

template <class _Traits>
class _CCCL_TYPE_VISIBILITY_DEFAULT __assignment : public __copy_constructor<_Traits>
{
  using __base_type = __copy_constructor<_Traits>;

  template <size_t _CurrentIndex, class _Other>
  _CCCL_API constexpr void
  __generic_assign(integral_constant<size_t, _CurrentIndex>, const size_t __index_, _Other&& __rhs)
  {
    if (__index_ == _CurrentIndex)
    {
      this->__assign_alt(
        __access::__base::__get_alt<_CurrentIndex>(this->__as_base()),
        __access::__base::__get_alt<_CurrentIndex>(::cuda::std::forward<_Other>(__rhs).__as_base()).__value);
      return;
    }
    this->__generic_assign(
      integral_constant<size_t, _CurrentIndex - 1>{}, __index_, ::cuda::std::forward<_Other>(__rhs));
  }

  template <class _Other>
  _CCCL_API constexpr void __generic_assign(integral_constant<size_t, 0>, const size_t __index_, _Other&& __rhs)
  {
    if (__index_ == 0)
    {
      this->__assign_alt(__access::__base::__get_alt<0>(this->__as_base()),
                         __access::__base::__get_alt<0>(::cuda::std::forward<_Other>(__rhs).__as_base()).__value);
      return;
    }
    // We already checked that every variant has a value, so we should never reach this line
    _CCCL_UNREACHABLE();
  }

public:
  using __base_type::__base_type;
  using __base_type::operator=;

  template <size_t _Ip, class... _Args>
  _CCCL_API auto& __emplace(_Args&&... __args)
  {
    this->__destroy();
    auto& __res =
      this->__construct_alt(__access::__base::__get_alt<_Ip>(*this), ::cuda::std::forward<_Args>(__args)...);
    this->__index_ = _Ip;
    return __res;
  }

protected:
  _CCCL_EXEC_CHECK_DISABLE
  template <size_t _Ip,
            class _Tp,
            class _Arg,
            enable_if_t<is_nothrow_constructible_v<_Tp, _Arg> || !is_nothrow_move_constructible_v<_Tp>, int> = 0>
  _CCCL_API void __assign_alt(__alt<_Ip, _Tp>& __a, _Arg&& __arg)
  {
    if (this->index() == _Ip)
    {
      __a.__value = ::cuda::std::forward<_Arg>(__arg);
    }
    else
    {
      this->__emplace<_Ip>(::cuda::std::forward<_Arg>(__arg));
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t _Ip,
            class _Tp,
            class _Arg,
            enable_if_t<!is_nothrow_constructible_v<_Tp, _Arg> && is_nothrow_move_constructible_v<_Tp>, int> = 0>
  _CCCL_API void __assign_alt(__alt<_Ip, _Tp>& __a, _Arg&& __arg)
  {
    if (this->index() == _Ip)
    {
      __a.__value = ::cuda::std::forward<_Arg>(__arg);
    }
    else
    {
      this->__emplace<_Ip>(_Tp(::cuda::std::forward<_Arg>(__arg)));
    }
  }

  template <class _That>
  _CCCL_API void __generic_assign(_That&& __that)
  {
    if (this->valueless_by_exception() && __that.valueless_by_exception())
    {
      // do nothing.
    }
    else if (__that.valueless_by_exception())
    {
      this->__destroy();
    }
    else
    {
      constexpr size_t __np = remove_cvref_t<__assignment>::__size();
      this->__generic_assign(integral_constant<size_t, __np - 1>{}, __that.index(), ::cuda::std::forward<_That>(__that));
    }
  }
};

template <class _Traits, _Trait = _Traits::__move_assignable_trait>
class _CCCL_TYPE_VISIBILITY_DEFAULT __move_assignment;

#define _LIBCUDACXX_VARIANT_MOVE_ASSIGNMENT(move_assignable_trait, move_assignment)                        \
  template <class... _Types>                                                                               \
  class _CCCL_TYPE_VISIBILITY_DEFAULT                                                                      \
  __move_assignment<__traits<_Types...>, move_assignable_trait> : public __assignment<__traits<_Types...>> \
  {                                                                                                        \
    using __base_type = __assignment<__traits<_Types...>>;                                                 \
                                                                                                           \
  public:                                                                                                  \
    using __base_type::__base_type;                                                                        \
    using __base_type::operator=;                                                                          \
                                                                                                           \
    _CCCL_HIDE_FROM_ABI __move_assignment(const __move_assignment&)            = default;                  \
    _CCCL_HIDE_FROM_ABI __move_assignment(__move_assignment&&)                 = default;                  \
    _CCCL_HIDE_FROM_ABI ~__move_assignment()                                   = default;                  \
    _CCCL_HIDE_FROM_ABI __move_assignment& operator=(const __move_assignment&) = default;                  \
    move_assignment                                                                                        \
  }

_LIBCUDACXX_VARIANT_MOVE_ASSIGNMENT(
  _Trait::_TriviallyAvailable, _CCCL_HIDE_FROM_ABI __move_assignment& operator=(__move_assignment&& __that) = default;);

_LIBCUDACXX_VARIANT_MOVE_ASSIGNMENT(
  _Trait::_Available,
  _CCCL_API __move_assignment&
  operator=(__move_assignment&& __that) noexcept(
    __all<(is_nothrow_move_constructible_v<_Types> && is_nothrow_move_assignable_v<_Types>) ...>::value) {
    this->__generic_assign(::cuda::std::move(__that));
    return *this;
  });

_LIBCUDACXX_VARIANT_MOVE_ASSIGNMENT(_Trait::_Unavailable, __move_assignment& operator=(__move_assignment&&) = delete;);

#undef _LIBCUDACXX_VARIANT_MOVE_ASSIGNMENT

template <class _Traits, _Trait = _Traits::__copy_assignable_trait>
class _CCCL_TYPE_VISIBILITY_DEFAULT __copy_assignment;

#define _LIBCUDACXX_VARIANT_COPY_ASSIGNMENT(copy_assignable_trait, copy_assignment)                             \
  template <class... _Types>                                                                                    \
  class _CCCL_TYPE_VISIBILITY_DEFAULT                                                                           \
  __copy_assignment<__traits<_Types...>, copy_assignable_trait> : public __move_assignment<__traits<_Types...>> \
  {                                                                                                             \
    using __base_type = __move_assignment<__traits<_Types...>>;                                                 \
                                                                                                                \
  public:                                                                                                       \
    using __base_type::__base_type;                                                                             \
    using __base_type::operator=;                                                                               \
                                                                                                                \
    _CCCL_HIDE_FROM_ABI __copy_assignment(const __copy_assignment&)       = default;                            \
    _CCCL_HIDE_FROM_ABI __copy_assignment(__copy_assignment&&)            = default;                            \
    _CCCL_HIDE_FROM_ABI ~__copy_assignment()                              = default;                            \
    _CCCL_HIDE_FROM_ABI __copy_assignment& operator=(__copy_assignment&&) = default;                            \
    copy_assignment                                                                                             \
  }

_LIBCUDACXX_VARIANT_COPY_ASSIGNMENT(
  _Trait::_TriviallyAvailable,
  _CCCL_HIDE_FROM_ABI __copy_assignment& operator=(const __copy_assignment& __that) = default;);

_LIBCUDACXX_VARIANT_COPY_ASSIGNMENT(
  _Trait::_Available, _CCCL_API __copy_assignment& operator=(const __copy_assignment& __that) {
    this->__generic_assign(__that);
    return *this;
  });

_LIBCUDACXX_VARIANT_COPY_ASSIGNMENT(_Trait::_Unavailable,
                                    __copy_assignment& operator=(const __copy_assignment&) = delete;);

#undef _LIBCUDACXX_VARIANT_COPY_ASSIGNMENT

template <class... _Types>
class _CCCL_TYPE_VISIBILITY_DEFAULT __impl : public __copy_assignment<__traits<_Types...>>
{
  using __base_type = __copy_assignment<__traits<_Types...>>;

  template <size_t _CurrentIndex>
  _CCCL_API static constexpr void
  __swap_value(integral_constant<size_t, _CurrentIndex>, const size_t __index_, __impl& __lhs, __impl& __rhs)
  {
    if (__index_ == _CurrentIndex)
    {
      using ::cuda::std::swap;
      swap(__access::__base::__get_alt<_CurrentIndex>(__lhs.__as_base()).__value,
           __access::__base::__get_alt<_CurrentIndex>(__rhs.__as_base()).__value);
      return;
    }
    __swap_value(integral_constant<size_t, _CurrentIndex - 1>{}, __index_, __lhs, __rhs);
  }

  _CCCL_API static constexpr void
  __swap_value(integral_constant<size_t, 0>, const size_t __index_, __impl& __lhs, __impl& __rhs)
  {
    if (__index_ == 0)
    {
      using ::cuda::std::swap;
      swap(__access::__base::__get_alt<0>(__lhs.__as_base()).__value,
           __access::__base::__get_alt<0>(__rhs.__as_base()).__value);
      return;
    }
    // We already checked that every variant has a value, so we should never reach this line
    _CCCL_UNREACHABLE();
  }

public:
  using __base_type::__base_type; // get in_place_index_t constructor & friends
  _CCCL_HIDE_FROM_ABI __impl(__impl const&)            = default;
  _CCCL_HIDE_FROM_ABI __impl(__impl&&)                 = default;
  _CCCL_HIDE_FROM_ABI __impl& operator=(__impl const&) = default;
  _CCCL_HIDE_FROM_ABI __impl& operator=(__impl&&)      = default;

  template <size_t _Ip, class _Arg>
  _CCCL_API void __assign(_Arg&& __arg)
  {
    this->__assign_alt(__access::__base::__get_alt<_Ip>(*this), ::cuda::std::forward<_Arg>(__arg));
  }

  _CCCL_API void __swap(__impl& __that)
  {
    if (this->valueless_by_exception() && __that.valueless_by_exception())
    {
      // do nothing.
    }
    else if (this->index() == __that.index())
    {
      constexpr size_t __np = remove_cvref_t<__impl>::__size();
      __swap_value(integral_constant<size_t, __np - 1>{}, this->index(), *this, __that);
    }
    else
    {
      __impl* __lhs = this;
      __impl* __rhs = ::cuda::std::addressof(__that);
      if (__lhs->__move_nothrow() && !__rhs->__move_nothrow())
      {
        ::cuda::std::swap(__lhs, __rhs);
      }
      else
      {
        __impl __tmp(::cuda::std::move(*__rhs));
        this->__generic_construct(*__rhs, ::cuda::std::move(*__lhs));
        this->__generic_construct(*__lhs, ::cuda::std::move(__tmp));
      }
    }
  }

private:
  [[nodiscard]] _CCCL_API constexpr bool __move_nothrow() const
  {
    constexpr bool __results[] = {is_nothrow_move_constructible_v<_Types>...};
    return this->valueless_by_exception() || __results[this->index()];
  }
};
} // namespace __variant_detail

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VARIANT_BASE_H
