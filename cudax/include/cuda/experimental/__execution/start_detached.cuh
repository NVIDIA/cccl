//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_START_DETACHED
#define __CUDAX_EXECUTION_START_DETACHED

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__exception/terminate.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/apply_sender.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct start_detached_t
{
private:
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_base_t
  {};

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    __opstate_base_t* __opstate_;
    void (*__destroy)(__opstate_base_t*) noexcept;

    template <class... _As>
    constexpr void set_value(_As&&...) noexcept
    {
      __destroy(__opstate_);
    }

    template <class _Error>
    constexpr void set_error(_Error&&) noexcept
    {
      ::cuda::std::terminate();
    }

    constexpr void set_stopped() noexcept
    {
      __destroy(__opstate_);
    }
  };

  template <class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __opstate_base_t
  {
    using operation_state_concept = operation_state_t;
    connect_result_t<_Sndr, __rcvr_t> __opstate_;

    static void __destroy(__opstate_base_t* __ptr) noexcept
    {
      delete static_cast<__opstate_t*>(__ptr);
    }

    _CCCL_API constexpr explicit __opstate_t(_Sndr&& __sndr)
        : __opstate_(execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{this, &__destroy}))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate_);
    }
  };

public:
  template <class _Sndr>
  _CCCL_API static auto apply_sender(_Sndr __sndr)
  {
    execution::start(*new __opstate_t<_Sndr>{static_cast<_Sndr&&>(__sndr)});
  }

  /// run detached.
  template <class _Sndr>
  _CCCL_NODEBUG_API void operator()(_Sndr __sndr) const
  {
    using __dom_t _CCCL_NODEBUG_ALIAS = __late_domain_of_t<_Sndr, env<>, __early_domain_of_t<_Sndr>>;
    execution::apply_sender(__dom_t{}, *this, static_cast<_Sndr&&>(__sndr));
  }
};

_CCCL_GLOBAL_CONSTANT start_detached_t start_detached{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_START_DETACHED
