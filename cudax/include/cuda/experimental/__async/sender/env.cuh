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

#include <cuda/std/__execution/env.h>

#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/queries.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
using _CUDA_VSTDEXEC::env;
using _CUDA_VSTDEXEC::env_of_t;
using _CUDA_VSTDEXEC::get_env;
using _CUDA_VSTDEXEC::prop;

using _CUDA_VSTDEXEC::__nothrow_queryable_with;
using _CUDA_VSTDEXEC::__query_result_t;
using _CUDA_VSTDEXEC::__queryable_with;

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

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_ENV
