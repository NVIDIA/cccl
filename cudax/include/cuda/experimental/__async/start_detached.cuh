//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
struct start_detached_t
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC
  struct __opstate_base_t : __immovable
  {};

  struct __rcvr_t
  {
    using receiver_concept = receiver_t;

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
      ::cuda::std::terminate();
    }

    void set_stopped() && noexcept
    {
      __destroy(__opstate_);
    }
  };

  template <class _Sndr>
  struct __opstate_t : __opstate_base_t
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = __async::completion_signatures_of_t<_Sndr, __rcvr_t>;
    connect_result_t<_Sndr, __rcvr_t> __opstate_;

    static void __destroy(__opstate_base_t* __ptr) noexcept
    {
      delete static_cast<__opstate_t*>(__ptr);
    }

    _CUDAX_API explicit __opstate_t(_Sndr&& __sndr)
        : __opstate_(__async::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{this, &__destroy}))
    {}

    _CUDAX_API void start() & noexcept
    {
      __async::start(__opstate_);
    }
  };

public:
  /// @brief Eagerly connects and starts a sender and lets it
  /// run detached.
  template <class _Sndr>
  _CUDAX_TRIVIAL_API void operator()(_Sndr __sndr) const
  {
    __async::start(*new __opstate_t<_Sndr>{static_cast<_Sndr&&>(__sndr)});
  }
};

_CCCL_GLOBAL_CONSTANT start_detached_t start_detached{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
