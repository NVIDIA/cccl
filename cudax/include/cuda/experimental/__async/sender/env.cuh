//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_ENV
#define __CUDAX_ASYNC_DETAIL_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/reference_wrapper.h>

#include <cuda/experimental/__async/sender/fwd.cuh>
#include <cuda/experimental/__async/sender/meta.cuh>
#include <cuda/experimental/__async/sender/queries.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/sender/type_traits.cuh>
#include <cuda/experimental/__async/sender/utility.cuh>

#include <functional>

#include <cuda/experimental/__async/sender/prologue.cuh>

_CCCL_DIAG_PUSH

// Suppress the warning: "definition of implicit copy constructor for 'env<>' is
// deprecated because it has a user-declared copy assignment operator". We need to
// suppress this warning rather than fix the code because defaulting or defining
// the copy constructor would prevent aggregate initialization, which these types
// depend on.
_CCCL_DIAG_SUPPRESS_CLANG("-Wunknown-warning-option")
_CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-copy")

namespace cuda::experimental::__async
{
template <class _Ty>
extern _Ty __unwrap_ref;

template <class _Ty>
extern _Ty& __unwrap_ref<::std::reference_wrapper<_Ty>>;

template <class _Ty>
extern _Ty& __unwrap_ref<_CUDA_VSTD::reference_wrapper<_Ty>>;

template <class _Ty>
using __unwrap_reference_t _CCCL_NODEBUG_ALIAS = decltype(__unwrap_ref<_Ty>);

template <class _Query, class _Value>
struct _CCCL_TYPE_VISIBILITY_DEFAULT prop
{
  _CCCL_NO_UNIQUE_ADDRESS _Query __query;
  _CCCL_NO_UNIQUE_ADDRESS _Value __value;

  _CUDAX_TRIVIAL_API constexpr auto query(_Query) const noexcept -> const _Value&
  {
    return __value;
  }

  auto operator=(const prop&) -> prop& = delete;
};

template <class _Query, class _Value>
prop(_Query, _Value) -> prop<_Query, _Value>;

template <class... _Envs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT env
{
  __tuple<_Envs...> __envs_;

  template <class _Query>
  _CUDAX_TRIVIAL_API static constexpr decltype(auto) __get_1st(const env& __self) noexcept
  {
    // NOLINTNEXTLINE (modernize-avoid-c-arrays)
    constexpr bool __flags[] = {__queryable_with<_Envs, _Query>..., false};
    constexpr size_t __idx   = __async::__find_pos(__flags, __flags + sizeof...(_Envs));
    if constexpr (__idx != __npos)
    {
      return __async::__cget<__idx>(__self.__envs_);
    }
  }

  template <class _Query>
  using __1st_env_t _CCCL_NODEBUG_ALIAS = decltype(env::__get_1st<_Query>(declval<const env&>()));

  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__queryable_with<__1st_env_t<_Query>, _Query>)
  _CUDAX_TRIVIAL_API constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable_with<__1st_env_t<_Query>, _Query>) -> __query_result_t<__1st_env_t<_Query>, _Query>
  {
    return env::__get_1st<_Query>(*this).query(__query);
  }

  auto operator=(const env&) -> env& = delete;
};

// partial specialization for two environments
template <class _Env0, class _Env1>
struct _CCCL_TYPE_VISIBILITY_DEFAULT env<_Env0, _Env1>
{
  _CCCL_NO_UNIQUE_ADDRESS _Env0 __env0_;
  _CCCL_NO_UNIQUE_ADDRESS _Env1 __env1_;

  template <class _Query>
  _CUDAX_TRIVIAL_API static constexpr decltype(auto) __get_1st(const env& __self) noexcept
  {
    if constexpr (__queryable_with<_Env0, _Query>)
    {
      return (__self.__env0_);
    }
    else
    {
      return (__self.__env1_);
    }
  }

  template <class _Query>
  using __1st_env_t _CCCL_NODEBUG_ALIAS = decltype(env::__get_1st<_Query>(declval<const env&>()));

  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__queryable_with<__1st_env_t<_Query>, _Query>)
  _CUDAX_TRIVIAL_API constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable_with<__1st_env_t<_Query>, _Query>) -> __query_result_t<__1st_env_t<_Query>, _Query>
  {
    return env::__get_1st<_Query>(*this).query(__query);
  }

  auto operator=(const env&) -> env& = delete;
};

template <class... _Envs>
env(_Envs...) -> env<__unwrap_reference_t<_Envs>...>;

using empty_env CCCL_DEPRECATED_BECAUSE("please use env<> instead of empty_env") = env<>;

struct get_env_t
{
  template <class _Ty>
  using __env_of _CCCL_NODEBUG_ALIAS = decltype(declval<_Ty>().get_env());

  template <class _Ty>
  _CUDAX_TRIVIAL_API auto operator()(_Ty&& __ty) const noexcept -> __env_of<_Ty&>
  {
    static_assert(noexcept(__ty.get_env()));
    return __ty.get_env();
  }

  _CUDAX_TRIVIAL_API auto operator()(__ignore) const noexcept -> env<>
  {
    return {};
  }
};

_CCCL_GLOBAL_CONSTANT get_env_t get_env{};

template <class _Ty>
using env_of_t _CCCL_NODEBUG_ALIAS = decltype(__async::get_env(declval<_Ty>()));

struct __not_a_scheduler
{
  using scheduler_concept _CCCL_NODEBUG_ALIAS = scheduler_t;
};

using __no_completion_scheduler_t _CCCL_NODEBUG_ALIAS =
  prop<get_completion_scheduler_t<set_value_t>, __not_a_scheduler>;
using __no_scheduler_t = prop<get_scheduler_t, __not_a_scheduler>;

// First look in the sender's environment for a domain. If none is found, look
// in the sender's (value) completion scheduler, if any.
template <class _Sndr>
using __early_domain_env _CCCL_NODEBUG_ALIAS =
  env<env_of_t<_Sndr>, __completion_scheduler_of_t<env<env_of_t<_Sndr>, __no_completion_scheduler_t>>>;

template <class _Sndr>
using early_domain_of_t _CCCL_NODEBUG_ALIAS = __domain_of_t<__early_domain_env<_Sndr>>;

// First look in the sender's environment for a domain. If none is found, look
// in the sender's (value) completion scheduler, if any. Then look in _Env for a
// domain. If none is found, look in the environment's scheduler, if any.
template <class _Sndr, class _Env>
using __late_domain_env _CCCL_NODEBUG_ALIAS =
  env<__early_domain_env<_Sndr>, env<_Env, __scheduler_of_t<env<_Env, __no_scheduler_t>>>>;

template <class _Sndr, class _Env>
using late_domain_of_t _CCCL_NODEBUG_ALIAS = __domain_of_t<__late_domain_env<_Sndr, _Env>>;

} // namespace cuda::experimental::__async

_CCCL_DIAG_POP

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
