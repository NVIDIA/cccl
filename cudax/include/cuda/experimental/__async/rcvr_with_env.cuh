//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_RCVR_WITH_ENV_H
#define __CUDAX_ASYNC_DETAIL_RCVR_WITH_ENV_H

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
template <class Rcvr, class Env>
struct _rcvr_with_env_t : Rcvr
{
  using _env_t = _rcvr_with_env_t const&;

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto rcvr() noexcept -> Rcvr&
  {
    return *this;
  }

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto rcvr() const noexcept -> const Rcvr&
  {
    return *this;
  }

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto get_env() const noexcept -> _env_t
  {
    return _env_t{*this};
  }

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr decltype(auto) _get_1st(Query) const noexcept
  {
    if constexpr (_queryable<Env, Query>)
    {
      return (_env);
    }
    else if constexpr (_queryable<env_of_t<Rcvr>, Query>)
    {
      return __async::get_env(static_cast<const Rcvr&>(*this));
    }
  }

  template <class Query, class Self = _rcvr_with_env_t>
  using _1st_env_t = decltype(DECLVAL(const Self&)._get_1st(Query{}));

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(Query query) const
    noexcept(_nothrow_queryable<_1st_env_t<Query>, Query>) //
    -> _query_result_t<_1st_env_t<Query>, Query>
  {
    return _get_1st(query).query(query);
  }

  Env _env;
};

template <class Rcvr, class Env>
struct _rcvr_with_env_t<Rcvr*, Env>
{
  using _env_t = _rcvr_with_env_t const&;

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto rcvr() const noexcept -> Rcvr*
  {
    return _rcvr;
  }

  template <class... As>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE void set_value(As&&... as) && noexcept
  {
    __async::set_value(_rcvr, static_cast<As&&>(as)...);
  }

  template <class Error>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE void set_error(Error&& error) && noexcept
  {
    __async::set_error(_rcvr, static_cast<Error&&>(error));
  }

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE void set_stopped() && noexcept
  {
    __async::set_stopped(_rcvr);
  }

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto get_env() const noexcept -> _env_t
  {
    return _env_t{*this};
  }

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr decltype(auto) _get_1st(Query) const noexcept
  {
    if constexpr (_queryable<Env, Query>)
    {
      return (_env);
    }
    else if constexpr (_queryable<env_of_t<Rcvr>, Query>)
    {
      return __async::get_env(_rcvr);
    }
  }

  template <class Query, class Self = _rcvr_with_env_t>
  using _1st_env_t = decltype(DECLVAL(const Self&)._get_1st(Query{}));

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(Query query) const
    noexcept(_nothrow_queryable<_1st_env_t<Query>, Query>) //
    -> _query_result_t<_1st_env_t<Query>, Query>
  {
    return _get_1st(query).query(query);
  }

  Rcvr* _rcvr;
  Env _env;
};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
