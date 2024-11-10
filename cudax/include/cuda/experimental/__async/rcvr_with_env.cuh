//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the _Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: _Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_RCVR_WITH_ENV
#define __CUDAX_ASYNC_DETAIL_RCVR_WITH_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/env.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
template <class _Rcvr, class _Env>
struct __rcvr_with_env_t : _Rcvr
{
  using __env_t = __rcvr_with_env_t const&;

  _CUDAX_TRIVIAL_API auto __rcvr() noexcept -> _Rcvr&
  {
    return *this;
  }

  _CUDAX_TRIVIAL_API auto __rcvr() const noexcept -> const _Rcvr&
  {
    return *this;
  }

  _CUDAX_TRIVIAL_API auto get_env() const noexcept -> __env_t
  {
    return __env_t{*this};
  }

  template <class _Query>
  _CUDAX_TRIVIAL_API constexpr decltype(auto) __get_1st(_Query) const noexcept
  {
    if constexpr (__queryable<_Env, _Query>)
    {
      return (__env_);
    }
    else if constexpr (__queryable<env_of_t<_Rcvr>, _Query>)
    {
      return __async::get_env(static_cast<const _Rcvr&>(*this));
    }
  }

  template <class _Query, class _Self = __rcvr_with_env_t>
  using _1st_env_t = decltype(__declval<const _Self&>().__get_1st(_Query{}));

  template <class _Query>
  _CUDAX_TRIVIAL_API constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable<_1st_env_t<_Query>, _Query>) //
    -> __query_result_t<_1st_env_t<_Query>, _Query>
  {
    return __get_1st(__query).query(__query);
  }

  _Env __env_;
};

template <class _Rcvr, class _Env>
struct __rcvr_with_env_t<_Rcvr*, _Env>
{
  using __env_t = __rcvr_with_env_t const&;

  _CUDAX_TRIVIAL_API auto __rcvr() const noexcept -> _Rcvr*
  {
    return __rcvr_;
  }

  template <class... _As>
  _CUDAX_TRIVIAL_API void set_value(_As&&... __as) && noexcept
  {
    __async::set_value(__rcvr_, static_cast<_As&&>(__as)...);
  }

  template <class _Error>
  _CUDAX_TRIVIAL_API void set_error(_Error&& __error) && noexcept
  {
    __async::set_error(__rcvr_, static_cast<_Error&&>(__error));
  }

  _CUDAX_TRIVIAL_API void set_stopped() && noexcept
  {
    __async::set_stopped(__rcvr_);
  }

  _CUDAX_TRIVIAL_API auto get_env() const noexcept -> __env_t
  {
    return __env_t{*this};
  }

  template <class _Query>
  _CUDAX_TRIVIAL_API constexpr decltype(auto) __get_1st(_Query) const noexcept
  {
    if constexpr (__queryable<_Env, _Query>)
    {
      return (__env_);
    }
    else if constexpr (__queryable<env_of_t<_Rcvr>, _Query>)
    {
      return __async::get_env(__rcvr_);
    }
  }

  template <class _Query, class _Self = __rcvr_with_env_t>
  using _1st_env_t = decltype(__declval<const _Self&>().__get_1st(_Query{}));

  template <class _Query>
  _CUDAX_TRIVIAL_API constexpr auto query(_Query __query) const
    noexcept(__nothrow_queryable<_1st_env_t<_Query>, _Query>) //
    -> __query_result_t<_1st_env_t<_Query>, _Query>
  {
    return __get_1st(__query).query(__query);
  }

  _Rcvr* __rcvr_;
  _Env __env_;
};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
