//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_JUST
#define __CUDAX_ASYNC_DETAIL_JUST

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/sender/utility.cuh>
#include <cuda/experimental/__async/sender/visit.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
// Forward declarations of the just* tag types:
struct just_t;
struct just_error_t;
struct just_stopped_t;

// Map from a disposition to the corresponding tag types:
namespace __detail
{
template <__disposition_t, class _Void = void>
extern __undefined<_Void> __just_tag;
template <class _Void>
extern __fn_t<just_t>* __just_tag<__value, _Void>;
template <class _Void>
extern __fn_t<just_error_t>* __just_tag<__error, _Void>;
template <class _Void>
extern __fn_t<just_stopped_t>* __just_tag<__stopped, _Void>;
} // namespace __detail

template <__disposition_t _Disposition>
struct __just_t
{
private:
  using _JustTag = decltype(__detail::__just_tag<_Disposition>());
  using _SetTag  = decltype(__detail::__set_tag<_Disposition>());

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __signatures_fn
  {
    template <class... _Ts>
    _CUDAX_API auto operator()(const _Ts&...) const noexcept -> __async::completion_signatures<_SetTag(_Ts...)>;
  };

  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __complete_fn
  {
    template <class... _Ts>
    _CUDAX_TRIVIAL_API void operator()(_Ts&&... __ts) const noexcept
    {
      _SetTag{}(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Ts&&>(__ts)...);
    }

    _Rcvr& __rcvr_;
  };

  template <class _Rcvr, class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _CUDAX_API void start() & noexcept
    {
      __values_(__complete_fn<_Rcvr>{__rcvr_});
    }

    _Rcvr __rcvr_;
    _Values __values_;
  };

  template <class... _Ts>
  static _CUDAX_TRIVIAL_API auto __mk_values(_Ts&... __ts)
  {
#if _CCCL_STD_VER >= 2020 && _CCCL_COMPILER(CLANG)
    // In C++20 we can directly move-capture a variadic pack of arguments:
    return [... __ts = static_cast<_Ts&&>(__ts)](auto fn) mutable noexcept {
      using _Fn = decltype(fn);
      static_assert(__nothrow_callable<_Fn, _Ts...>);
      return static_cast<_Fn&&>(fn)(static_cast<_Ts&&>(__ts)...);
    };
#else
    // In C++17 we need to use a tuple to move-capture a variadic pack of arguments:
    return [__ts = __tupl{static_cast<_Ts&&>(__ts)...}](auto fn) mutable noexcept {
      using _Fn = decltype(fn);
      static_assert(__nothrow_callable<_Fn, _Ts...>);
      return __ts.__apply(static_cast<_Fn&&>(fn), static_cast<__tuple<_Ts...>&&>(__ts));
    };
#endif
  }

public:
  template <class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class... _Ts>
  _CUDAX_TRIVIAL_API auto operator()(_Ts... __ts) const;
};

template <__disposition_t _Disposition>
template <class _Values>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __just_t<_Disposition>::__sndr_t
{
  using sender_concept = sender_t;

  _CCCL_NO_UNIQUE_ADDRESS _JustTag __tag_;
  _Values __values_;

  template <class _Self, class... _Env>
  _CUDAX_API static constexpr auto get_completion_signatures() noexcept
  {
    using __signatures_t = __call_result_t<_Values&, __signatures_fn>;
    return __signatures_t();
  }

  template <class _Rcvr>
  _CUDAX_API __opstate_t<_Rcvr, _Values> connect(_Rcvr __rcvr) && //
    noexcept(__nothrow_decay_copyable<_Rcvr, _Values>)
  {
    return __opstate_t<_Rcvr, _Values>{static_cast<_Rcvr&&>(__rcvr), static_cast<_Values&&>(__values_)};
  }

  template <class _Rcvr>
  _CUDAX_API __opstate_t<_Rcvr, _Values> connect(_Rcvr __rcvr) const& //
    noexcept(__nothrow_decay_copyable<_Rcvr, _Values const&>)
  {
    return __opstate_t<_Rcvr, _Values>{static_cast<_Rcvr&&>(__rcvr), __values_};
  }
};

template <__disposition_t _Disposition>
template <class... _Ts>
_CUDAX_TRIVIAL_API auto __just_t<_Disposition>::operator()(_Ts... __ts) const
{
  using _Values = decltype(__mk_values(__ts...));
  return __sndr_t<_Values>{{}, __mk_values(__ts...)};
}

template <class _Fn>
inline constexpr size_t structured_binding_size<__just_t<__value>::__sndr_t<_Fn>> = 2;
template <class _Fn>
inline constexpr size_t structured_binding_size<__just_t<__error>::__sndr_t<_Fn>> = 2;
template <class _Fn>
inline constexpr size_t structured_binding_size<__just_t<__stopped>::__sndr_t<_Fn>> = 2;

_CCCL_GLOBAL_CONSTANT struct just_t : __just_t<__value>
{
} just{};

_CCCL_GLOBAL_CONSTANT struct just_error_t : __just_t<__error>
{
} just_error{};

_CCCL_GLOBAL_CONSTANT struct just_stopped_t : __just_t<__stopped>
{
} just_stopped{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
