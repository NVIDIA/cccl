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

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT transform_sender_t
{
  template <class _Domain, class _Sndr, class... _Env>
  using __transform_domain_t = ::cuda::std::
    _If<__is_instantiable_with<__transform_sender_result_t, _Domain, _Sndr, _Env...>, _Domain, default_domain>;

  enum class __strategy
  {
    __passthru,
    __transform,
    __transform_recurse
  };

  struct __transform_strategy_t
  {
    bool __nothrow_;
    __strategy __strategy_;
  };

  template <class _Self, class _Domain, class _Sndr, class... _Env>
  _CCCL_NODEBUG_API static constexpr auto __get_transform_strategy() noexcept -> __transform_strategy_t
  {
    using __dom_t _CCCL_NODEBUG_ALIAS    = __transform_domain_t<_Domain, _Sndr, _Env...>;
    using __result_t _CCCL_NODEBUG_ALIAS = __transform_sender_result_t<__dom_t, _Sndr, _Env...>;

    if constexpr (::cuda::std::_IsSame<__result_t&&, _Sndr&&>::value)
    {
      return __transform_strategy_t{true, __strategy::__passthru};
    }
    else
    {
      using __dom2_t _CCCL_NODEBUG_ALIAS =
        __transform_domain_t<__domain_of_t<__result_t, _Env...>, __result_t, _Env...>;
      using __result2_t _CCCL_NODEBUG_ALIAS = __transform_sender_result_t<__dom2_t, __result_t, _Env...>;

      if constexpr (::cuda::std::_IsSame<__result2_t&&, __result_t&&>::value)
      {
        constexpr bool __nothrow_ = noexcept(__dom_t{}.transform_sender(declval<_Sndr>(), declval<const _Env&>()...));
        return __transform_strategy_t{__nothrow_, __strategy::__transform};
      }
      else
      {
        constexpr bool __nothrow_ = noexcept(
          _Self{}(__dom2_t{},
                  __dom_t{}.transform_sender(declval<_Sndr>(), declval<const _Env&>()...),
                  declval<const _Env&>()...));
        return __transform_strategy_t{__nothrow_, __strategy::__transform_recurse};
      }
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Self = transform_sender_t, class _Domain, class _Sndr, class... _Env)
  _CCCL_REQUIRES((__get_transform_strategy<_Self, _Domain, _Sndr, _Env...>().__strategy_ == __strategy::__passthru))
  _CCCL_NODEBUG_API constexpr auto operator()(_Domain, _Sndr&& __sndr, const _Env&...) const
    noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    return static_cast<_Sndr&&>(__sndr);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Self = transform_sender_t, class _Domain, class _Sndr, class... _Env)
  _CCCL_REQUIRES((__get_transform_strategy<_Self, _Domain, _Sndr, _Env...>().__strategy_ == __strategy::__transform))
  _CCCL_NODEBUG_API constexpr auto operator()(_Domain, _Sndr&& __sndr, const _Env&... __env) const
    noexcept(__get_transform_strategy<_Self, _Domain, _Sndr, _Env...>().__nothrow_) -> decltype(auto)
  {
    using __dom_t _CCCL_NODEBUG_ALIAS = __transform_domain_t<_Domain, _Sndr, _Env...>;
    return __dom_t{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env...);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Self = transform_sender_t, class _Domain, class _Sndr, class... _Env)
  _CCCL_REQUIRES((__get_transform_strategy<_Self, _Domain, _Sndr, _Env...>().__strategy_
                  == __strategy::__transform_recurse))
  _CCCL_NODEBUG_API constexpr auto operator()(_Domain, _Sndr&& __sndr, const _Env&... __env) const
    noexcept(__get_transform_strategy<_Self, _Domain, _Sndr, _Env...>().__nothrow_) -> decltype(auto)
  {
    using __dom1_t _CCCL_NODEBUG_ALIAS    = __transform_domain_t<_Domain, _Sndr, _Env...>;
    using __result1_t _CCCL_NODEBUG_ALIAS = __transform_sender_result_t<__dom1_t, _Sndr, _Env...>;
    using __dom2_t _CCCL_NODEBUG_ALIAS = __transform_domain_t<__early_domain_of_t<__result1_t>, __result1_t, _Env...>;
    return (*this)(__dom2_t{}, __dom1_t{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env...), __env...);
  }
};

_CCCL_GLOBAL_CONSTANT transform_sender_t transform_sender{};

template <class _Sndr, class... _Env>
_CCCL_CONCEPT __has_sender_transform =
  transform_sender_t::__get_transform_strategy<transform_sender_t, __domain_of_t<_Sndr, _Env...>, _Sndr, _Env...>()
    .__strategy_
  != transform_sender_t::__strategy::__passthru;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TRANSFORM_SENDER
