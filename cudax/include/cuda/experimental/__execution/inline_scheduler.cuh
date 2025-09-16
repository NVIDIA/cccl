//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_INLINE_SCHEDULER
#define __CUDAX_EXECUTION_INLINE_SCHEDULER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>

#include <cuda/experimental/__execution/completion_behavior.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//! Scheduler that returns a sender that always completes inline (successfully).
struct _CCCL_TYPE_VISIBILITY_DEFAULT inline_scheduler : __inln_attrs_t<set_value_t>
{
private:
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t : __inln_attrs_t<set_value_t>
  {};

  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __immovable
  {
    using operation_state_concept = operation_state_t;

    _CCCL_API constexpr void start() noexcept
    {
      set_value(static_cast<_Rcvr&&>(__rcvr));
    }

    _Rcvr __rcvr;
  };

public:
  using scheduler_concept = scheduler_t;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    template <class Self>
    [[nodiscard]] _CCCL_API static constexpr auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t()>{};
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr>
    {
      return {{}, static_cast<_Rcvr&&>(__rcvr)};
    }

    [[nodiscard]] _CCCL_API static constexpr auto get_env() noexcept -> __attrs_t
    {
      return {};
    }
  };

  [[nodiscard]] _CCCL_API constexpr auto schedule() const noexcept -> __sndr_t
  {
    return {};
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(inline_scheduler, inline_scheduler) noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(inline_scheduler, inline_scheduler) noexcept
  {
    return false;
  }
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_INLINE_SCHEDULER
