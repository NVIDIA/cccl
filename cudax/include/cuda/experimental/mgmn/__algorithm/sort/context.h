// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_CONTEXT_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_CONTEXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

#include <cuda/experimental/mgmn/__algorithm/sort/device.h>

#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__utility/move.h>

#  include <memory>
#  include <mutex>

#  include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

struct __sort_context_query
{
  template <class _Env>
  [[nodiscard]] _CCCL_HOST_API constexpr auto operator()(const _Env& __env) const noexcept
    -> decltype(__env.query(*this))
  {
    return __env.query(*this);
  }
};

//! @brief Retain NCCL device sort resources across collective sort calls.
//!
//! Use one context per local rank and bind it to that rank's execution environment.
//! The first nonempty multi-rank sort lazily creates a device communicator and a
//! symmetric workspace. Later calls reuse them, growing the workspace collectively
//! when needed. Generic and single-rank sorts do not allocate context resources.
//!
//! The context becomes associated with the first NCCL communicator and CUDA device
//! that use it. Other associations are rejected collectively until reset(). The
//! parent communicator and the last-used stream must outlive the context or reset().
//! Calls sharing a context are serialized locally; callers must still issue the
//! same collective order on all ranks. Destruction/reset must not race a sort.
//!
//! @code{.cpp}
//! cuda::experimental::mgmn::sort_context context;
//! auto env = context.bind(cuda::stream_ref{stream});
//! for (auto input : batches) {
//!   cuda::experimental::mgmn::sort(cuda::experimental::distributed, comm, env,
//!                                  input.data(), input.size());
//! }
//! context.reset(); // Before destroying comm or stream.
//! @endcode
class sort_context
{
  ::std::unique_ptr<__detail::__sort::__device_window> __window_;
  ::std::mutex __mutex_;

public:
  sort_context()                               = default;
  sort_context(const sort_context&)            = delete;
  sort_context& operator=(const sort_context&) = delete;

  //! @brief Attach this context while preserving the environment's other queries.
  //! @param[in] __env The stream and optional resource/tuning environment.
  //! @return An environment referencing this context; it must not outlive the context.
  template <class _Env>
  [[nodiscard]] _CCCL_HOST_API auto bind(
    _Env __env) & -> ::cuda::std::execution::env<::cuda::std::execution::prop<__sort_context_query, sort_context*>, _Env>
  {
    return {{__sort_context_query{}, this}, ::cuda::std::move(__env)};
  }

  //! @brief Release retained resources before the parent communicator or stream.
  _CCCL_HOST_API void reset()
  {
    const ::std::lock_guard<::std::mutex> __lock{__mutex_};
    __window_.reset();
  }

  [[nodiscard]] _CCCL_HOST_API auto __lock() -> ::std::unique_lock<::std::mutex>
  {
    return ::std::unique_lock<::std::mutex>{__mutex_};
  }

  [[nodiscard]] _CCCL_HOST_API auto __window() noexcept -> ::std::unique_ptr<__detail::__sort::__device_window>&
  {
    return __window_;
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)
#  include <cuda/std/__cccl/epilogue.h>
#endif // _CUDAX_SORT_HAS_NCCL_DEVICE
#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_CONTEXT_H
