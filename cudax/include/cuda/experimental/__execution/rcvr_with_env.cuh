//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the _Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: _Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_RCVR_WITH_ENV
#define __CUDAX_EXECUTION_RCVR_WITH_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Rcvr, class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_with_env_t : _Rcvr
{
  // If _Env has a value for the `get_scheduler` query, then we must ensure that we report
  // the domain correctly. Under no circumstances should we forward the `get_domain` query
  // to the receiver's environment. That environment may have a domain that does not
  // conform to the scheduler in _Env.
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    // Prefer to query _Env
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__queryable_with<_Env, _Query>)
    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<_Env, _Query>)
      -> __query_result_t<_Env, _Query>
    {
      return __rcvr_->__env_.query(_Query{});
    }

    // Fallback to querying the inner receiver's environment, but only for forwarding
    // queries.
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES((!__queryable_with<_Env, _Query>)
                     _CCCL_AND __forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Rcvr>, _Query>)
    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(_Query) const
      noexcept(__nothrow_queryable_with<env_of_t<_Rcvr>, _Query>) -> __query_result_t<env_of_t<_Rcvr>, _Query>
    {
      // If _Env has a value for the `get_scheduler` query, then we should not be
      // forwarding a get_domain query to the parent receiver's environment.
      static_assert(!_CUDA_VSTD::is_same_v<_Query, get_domain_t> || !__queryable_with<_Env, get_scheduler_t>,
                    "_Env specifies a scheduler but not a domain.");
      return __rcvr_->__base().query(_Query{});
      // return execution::get_env(__rcvr_->__base()).query(_Query{});
    }

    __rcvr_with_env_t const* __rcvr_;
  };

  [[nodiscard]] _CCCL_TRIVIAL_API auto __base() && noexcept -> _Rcvr&&
  {
    return static_cast<_Rcvr&&>(*this);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API auto __base() & noexcept -> _Rcvr&
  {
    return *this;
  }

  [[nodiscard]] _CCCL_TRIVIAL_API auto __base() const& noexcept -> _Rcvr const&
  {
    return *this;
  }

  [[nodiscard]] _CCCL_TRIVIAL_API auto get_env() const noexcept -> __env_t
  {
    return __env_t{this};
  }

  _Env __env_;
};

template <class _Rcvr, class _Env>
_CCCL_HOST_DEVICE __rcvr_with_env_t(_Rcvr, _Env) -> __rcvr_with_env_t<_Rcvr, _Env>;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_RCVR_WITH_ENV
