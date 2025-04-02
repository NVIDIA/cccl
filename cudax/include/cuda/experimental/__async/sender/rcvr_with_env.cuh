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

#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/env.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
template <class _Rcvr, class _Env>
struct __rcvr_with_env_t : _Rcvr
{
  struct __env_t
  {
    template <class _Query>
    _CUDAX_TRIVIAL_API static constexpr decltype(auto) __get_1st(const __env_t& __self) noexcept
    {
      if constexpr (__queryable_with<_Env, _Query>)
      {
        return (__self.__rcvr_->__env_);
      }
      else if constexpr (__queryable_with<env_of_t<_Rcvr>, _Query>)
      {
        return __async::get_env(static_cast<const _Rcvr&>(*__self.__rcvr_));
      }
    }

    template <class _Query>
    using __1st_env_t = decltype(__env_t::__get_1st<_Query>(declval<const __env_t&>()));

    template <class _Query>
    _CUDAX_TRIVIAL_API constexpr auto query(_Query) const
      noexcept(__nothrow_queryable_with<__1st_env_t<_Query>, _Query>) //
      -> __query_result_t<__1st_env_t<_Query>, _Query>
    {
      return __env_t::__get_1st<_Query>(*this);
    }

    __rcvr_with_env_t const* __rcvr_;
  };

  _CUDAX_TRIVIAL_API auto __base() && noexcept -> _Rcvr&&
  {
    return static_cast<_Rcvr&&>(*this);
  }

  _CUDAX_TRIVIAL_API auto __base() & noexcept -> _Rcvr&
  {
    return *this;
  }

  _CUDAX_TRIVIAL_API auto __base() const& noexcept -> _Rcvr const&
  {
    return *this;
  }

  _CUDAX_TRIVIAL_API auto get_env() const noexcept -> __env_t
  {
    return __env_t{this};
  }

  _Env __env_;
};

template <class _Rcvr, class _Env>
__rcvr_with_env_t(_Rcvr, _Env) -> __rcvr_with_env_t<_Rcvr, _Env>;

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
