//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_PARALLEL_SCHEDULER
#define __CUDAX_EXECUTION_PARALLEL_SCHEDULER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/__utility/basic_any.h>
#include <cuda/std/__memory/allocator.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/typeid.h>
#include <cuda/std/cstddef>
#include <cuda/std/optional>
#include <cuda/std/span>

#include <cuda/experimental/__execution/any_allocator.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/stop_token.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __detail
{
struct __try_queryable
{
  _CCCL_API virtual void operator()(const get_stop_token_t&,
                                    ::cuda::std::optional<inplace_stop_token>&) const noexcept = 0;

  _CCCL_API virtual void operator()(const get_allocator_t&,
                                    ::cuda::std::optional<any_allocator<::cuda::std::byte>>&) const noexcept = 0;

  _CCCL_HOST_API virtual void
  operator()(const get_scheduler_t&, ::cuda::std::optional<task_scheduler>&) const noexcept = 0;
};

struct __any : ::cuda::__basic_any<::cuda::__icopyable<>>
{
  using __any::__basic_any::__basic_any;
};
} // namespace __detail

class receiver_proxy : __detail::__try_queryable
{
public:
  _CCCL_HIDE_FROM_ABI virtual ~receiver_proxy() = default;

  _CCCL_API virtual void set_value() noexcept                = 0;
  _CCCL_API virtual void set_error(exception_ptr&&) noexcept = 0;
  _CCCL_API virtual void set_stopped() noexcept              = 0;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Value, class Query)
  _CCCL_REQUIRES(__callable<const __detail::__try_queryable&, Query, ::cuda::std::optional<_Value>&>)
  [[nodiscard]] _CCCL_API auto try_query(const Query& __query) const noexcept -> ::cuda::std::optional<_Value>
  {
    const __detail::__try_queryable& __queryable = *this;
    ::cuda::std::optional<_Value> __value;
    __queryable(__query, __value);
    return __value;
  }
};

struct bulk_item_receiver_proxy : receiver_proxy
{
  _CCCL_API virtual void execute(size_t, size_t) noexcept = 0;
};

struct parallel_scheduler_backend
{
  _CCCL_HIDE_FROM_ABI virtual ~parallel_scheduler_backend() = default;

  _CCCL_API virtual void schedule(receiver_proxy&, ::cuda::std::span<::cuda::std::byte>) noexcept = 0;

  _CCCL_API virtual void
  schedule_bulk_chunked(size_t, bulk_item_receiver_proxy&, ::cuda::std::span<::cuda::std::byte>) noexcept = 0;

  _CCCL_API virtual void
  schedule_bulk_unchunked(size_t, bulk_item_receiver_proxy&, ::cuda::std::span<::cuda::std::byte>) noexcept = 0;
};

namespace __detail
{
template <class _Rcvr>
struct __proxy_receiver_impl : receiver_proxy
{
public:
  using receiver_concept = receiver_t;

  explicit __proxy_receiver_impl(_Rcvr rcvr)
      : __rcvr_(static_cast<_Rcvr&&>(rcvr))
  {}

  _CCCL_API void set_value() noexcept final override
  {
    execution::set_value(_CCCL_MOVE(__rcvr_));
  }

  _CCCL_API void set_error(exception_ptr&& eptr) noexcept final override
  {
    execution::set_error(_CCCL_MOVE(__rcvr_), _CCCL_MOVE(eptr));
  }

  _CCCL_API void set_stopped() noexcept final override
  {
    execution::set_stopped(_CCCL_MOVE(__rcvr_));
  }

private:
  _CCCL_API void operator()(const get_stop_token_t& __query,
                            ::cuda::std::optional<inplace_stop_token>& out) const noexcept final override
  {
    if constexpr (__callable<const get_stop_token_t&, env_of_t<_Rcvr>>)
    {
      if constexpr (__same_as<__call_result_t<const get_stop_token_t&, env_of_t<_Rcvr>>, inplace_stop_token>)
      {
        out.emplace(__query(get_env(__rcvr_)));
      }
      else
      {
        out.emplace();
      }
    }
  }

  _CCCL_API void operator()(const get_allocator_t& __query,
                            ::cuda::std::optional<any_allocator<::cuda::std::byte>>& out) const noexcept final override
  {
    if constexpr (__callable<const get_allocator_t&, env_of_t<_Rcvr>>)
    {
      out.emplace(any_allocator{__query(get_env(__rcvr_))});
    }
  }

  // defined in task_scheduler.cuh:
  _CCCL_HOST_API void operator()(const get_scheduler_t& __query,
                                 ::cuda::std::optional<task_scheduler>& out) const noexcept final override;

  _Rcvr __rcvr_;
};
} // namespace __detail
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_PARALLEL_SCHEDULER
