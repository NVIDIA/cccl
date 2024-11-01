//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <functional>

#include <cuda/experimental/__async/prologue.cuh>

_CCCL_DIAG_PUSH

// Suppress the warning: "definition of implicit copy constructor for 'env<>' is
// deprecated because it has a user-declared copy assignment operator". We need to
// suppress this warning rather than fix the code because defaulting or defining
// the copy constructor would prevent aggregate initialization, which these types
// depend on.
_CCCL_DIAG_SUPPRESS_CLANG("-Wunknown-warning-option")
_CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-copy")

// warning #20012-D: __device__ annotation is ignored on a
// function("inplace_stop_source") that is explicitly defaulted on its first
// declaration
_CCCL_NV_DIAG_SUPPRESS(20012)

namespace cuda::experimental::__async
{
template <class _Ty>
extern _Ty __unwrap_ref;

template <class _Ty>
extern _Ty& __unwrap_ref<::std::reference_wrapper<_Ty>>;

template <class _Ty>
extern _Ty& __unwrap_ref<_CUDA_VSTD::reference_wrapper<_Ty>>;

template <class _Ty>
using __unwrap_reference_t = decltype(__unwrap_ref<_Ty>);

template <class _Query, class _Value>
struct prop
{
  _CCCL_NO_UNIQUE_ADDRESS _Query __query;
  _CCCL_NO_UNIQUE_ADDRESS _Value __value;

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(_Query) const noexcept -> const _Value&
  {
    return __value;
  }

  prop& operator=(const prop&) = delete;
};

template <class... _Envs>
struct env
{
  __tuple<_Envs...> __envs_;

  template <class _Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr decltype(auto) __get_1st(_Query) const noexcept
  {
    constexpr bool __flags[] = {__queryable<_Envs, _Query>..., false};
    constexpr size_t __idx   = __async::__find_pos(__flags, __flags + sizeof...(_Envs));
    if constexpr (__idx != __npos)
    {
      return __async::__cget<__idx>(__envs_);
    }
  }

  template <class _Query, class _Env = env>
  using __1st_env_t = decltype(__declval<const _Env&>().__get_1st(_Query{}));

  template <class _Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable<__1st_env_t<_Query>, _Query>) //
    -> __query_result_t<__1st_env_t<_Query>, _Query>
  {
    return __get_1st(__query).__query(__query);
  }

  env& operator=(const env&) = delete;
};

// partial specialization for two environments
template <class _Env0, class _Env1>
struct env<_Env0, _Env1>
{
  _CCCL_NO_UNIQUE_ADDRESS _Env0 __env0_;
  _CCCL_NO_UNIQUE_ADDRESS _Env1 __env1_;

  template <class _Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr decltype(auto) __get_1st(_Query) const noexcept
  {
    if constexpr (__queryable<_Env0, _Query>)
    {
      return (__env0_);
    }
    else if constexpr (__queryable<_Env1, _Query>)
    {
      return (__env1_);
    }
  }

  template <class _Query, class _Env = env>
  using __1st_env_t = decltype(__declval<const _Env&>().__get_1st(_Query{}));

  template <class _Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable<__1st_env_t<_Query>, _Query>) //
    -> __query_result_t<__1st_env_t<_Query>, _Query>
  {
    return __get_1st(__query).__query(__query);
  }

  env& operator=(const env&) = delete;
};

template <class... _Envs>
_CCCL_HOST_DEVICE env(_Envs...) -> env<__unwrap_reference_t<_Envs>...>;

using empty_env = env<>;

namespace __adl
{
template <class _Ty>
_CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto get_env(_Ty* __ty) noexcept //
  -> decltype(__ty->get_env())
{
  static_assert(noexcept(__ty->get_env()));
  return __ty->get_env();
}

struct __get_env_t
{
  template <class _Ty>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(_Ty* __ty) const noexcept //
    -> decltype(get_env(__ty))
  {
    static_assert(noexcept(get_env(__ty)));
    return get_env(__ty);
  }
};
} // namespace __adl

struct get_env_t
{
  template <class _Ty>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(_Ty&& __ty) const noexcept //
    -> decltype(__ty.get_env())
  {
    static_assert(noexcept(__ty.get_env()));
    return __ty.get_env();
  }

  template <class _Ty>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(_Ty* __ty) const noexcept //
    -> __call_result_t<__adl::__get_env_t, _Ty*>
  {
    return __adl::__get_env_t()(__ty);
  }

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE empty_env operator()(__ignore) const noexcept
  {
    return {};
  }
};

namespace __region
{
_CCCL_GLOBAL_CONSTANT get_env_t get_env{};
} // namespace __region

using namespace __region;

template <class _Ty>
using env_of_t = decltype(get_env(__declval<_Ty>()));
} // namespace cuda::experimental::__async

_CCCL_NV_DIAG_DEFAULT(20012)

_CCCL_DIAG_POP

#include <cuda/experimental/__async/epilogue.cuh>

#endif
