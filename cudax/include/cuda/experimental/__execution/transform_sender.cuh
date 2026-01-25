//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_TRANSFORM_SENDER
#define __CUDAX_EXECUTION_TRANSFORM_SENDER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __detail
{
template <class _Env>
using __starting_domain = __domain_of_t<const _Env&>;

template <class _Sndr, class _Env>
using __completing_domain = __call_result_t<get_completion_domain_t<set_value_t>, env_of_t<_Sndr>, const _Env&>;

template <class _Domain, class _OpTag>
struct __transform_sender_t
{
  template <class _Sndr, class _Env>
  using __domain_for_t = ::cuda::std::_If< //
    __has_transform_sender<_Domain, _OpTag, _Sndr, _Env>,
    _Domain,
    default_domain>;

  template <class _Sndr, class _Env, bool _Nothrow = true>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto __get_declfn() noexcept
  {
    using __domain_t = __domain_for_t<_Sndr, _Env>;
    using __result_t = __transform_sender_result_t<__domain_t, _OpTag, _Sndr, _Env>;

    constexpr bool __is_nothrow = __nothrow_transform_sender<__domain_t, _OpTag, _Sndr, _Env>;

    if constexpr (__same_as<__result_t, _Sndr>)
    {
      return __declfn<__result_t, __is_nothrow>;
    }
    else if constexpr (__same_as<_OpTag, start_t>)
    {
      return __get_declfn<__result_t, const _Env&, (_Nothrow && __is_nothrow)>();
    }
    else
    {
      using __transform_recurse_t = __transform_sender_t<__completing_domain<__result_t, _Env>, set_value_t>;
      return __transform_recurse_t::template __get_declfn<__result_t, _Env, (_Nothrow && __is_nothrow)>();
    }
  }

  template <class _Sndr, class _Env, auto _DeclFn = __get_declfn<_Sndr, _Env>()>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr&& __sndr, const _Env& __env) const
    noexcept(noexcept(_DeclFn())) -> decltype(_DeclFn())
  {
    using __domain_t = __domain_for_t<_Sndr, _Env>;
    using __result_t = __transform_sender_result_t<__domain_t, _OpTag, _Sndr, _Env>;

    if constexpr (__same_as<__result_t, _Sndr>)
    {
      return __domain_t().transform_sender(_OpTag(), static_cast<_Sndr&&>(__sndr), __env);
    }
    else if constexpr (__same_as<_OpTag, start_t>)
    {
      return (*this)(__domain_t().transform_sender(_OpTag(), static_cast<_Sndr&&>(__sndr), __env), __env);
    }
    else
    {
      using __transform_recurse_t = __transform_sender_t<__completing_domain<__result_t, _Env>, set_value_t>;
      return __transform_recurse_t()(
        __domain_t().transform_sender(_OpTag(), static_cast<_Sndr&&>(__sndr), __env), __env);
    }
  }
};
} // namespace __detail

struct _CCCL_TYPE_VISIBILITY_DEFAULT transform_sender_t
{
private:
  template <class _Fn1, class _Fn2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __compose
  {
    template <class _Sndr, class _Env>
    _CCCL_API constexpr auto operator()(_Sndr&& __sndr, const _Env& __env) const
      noexcept(noexcept(_Fn1()(_Fn2()(static_cast<_Sndr&&>(__sndr), __env), __env)))
        -> decltype(_Fn1()(_Fn2()(static_cast<_Sndr&&>(__sndr), __env), __env))
    {
      return _Fn1()(_Fn2()(static_cast<_Sndr&&>(__sndr), __env), __env);
    }
  };

  template <class _Sndr, class _Env>
  using __impl_fn_t =
    __compose<__detail::__transform_sender_t<__detail::__starting_domain<_Env>, start_t>,
              __detail::__transform_sender_t<__detail::__completing_domain<_Sndr, _Env>, set_value_t>>;

public:
  template <class _Sndr, class _Env, class _ImplFn = __impl_fn_t<_Sndr, _Env>>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr&& __sndr, const _Env& __env) const
    noexcept(noexcept(_ImplFn()(static_cast<_Sndr&&>(__sndr), __env)))
      -> decltype(_ImplFn()(static_cast<_Sndr&&>(__sndr), __env))
  {
    return _ImplFn()(static_cast<_Sndr&&>(__sndr), __env);
  }
};

_CCCL_GLOBAL_CONSTANT transform_sender_t transform_sender{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TRANSFORM_SENDER
