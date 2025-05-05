//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_START_DETACHED
#define __CUDAX_ASYNC_DETAIL_START_DETACHED

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/env.cuh>
#include <cuda/experimental/__async/sender/utility.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
struct start_detached_t
{
private:
  struct __opstate_base_t : private __immovable
  {};

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept _CCCL_NODEBUG_ALIAS = receiver_t;

    __opstate_base_t* __opstate_;
    void (*__destroy)(__opstate_base_t*) noexcept;

    template <class... _As>
    void set_value(_As&&...) && noexcept
    {
      __destroy(__opstate_);
    }

    template <class _Error>
    void set_error(_Error&&) && noexcept
    {
      _CUDA_VSTD_NOVERSION::terminate();
    }

    void set_stopped() && noexcept
    {
      __destroy(__opstate_);
    }
  };

  template <class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __opstate_base_t
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;
    connect_result_t<_Sndr, __rcvr_t> __opstate_;

    static void __destroy(__opstate_base_t* __ptr) noexcept
    {
      delete static_cast<__opstate_t*>(__ptr);
    }

    _CUDAX_API explicit __opstate_t(_Sndr&& __sndr)
        : __opstate_(__async::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{this, &__destroy}))
    {}

    _CUDAX_API void start() noexcept
    {
      __async::start(__opstate_);
    }
  };

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __fn
  {
    template <class _Sndr>
    _CUDAX_API auto operator()(_Sndr __sndr) const
    {
      __async::start(*new __opstate_t<_Sndr>{static_cast<_Sndr&&>(__sndr)});
    }
  };

public:
  _CUDAX_API static constexpr auto __apply() noexcept
  {
    return __fn{};
  }

  /// run detached.
  template <class _Sndr>
  _CUDAX_TRIVIAL_API void operator()(_Sndr __sndr) const
  {
    using __dom_t _CCCL_NODEBUG_ALIAS = early_domain_of_t<_Sndr>;
    __dom_t::__apply (*this)(static_cast<_Sndr&&>(__sndr));
  }
};

_CCCL_GLOBAL_CONSTANT start_detached_t start_detached{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
