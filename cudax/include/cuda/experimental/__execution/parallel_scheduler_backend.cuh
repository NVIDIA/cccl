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
#include <cuda/__utility/immovable.h>
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

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // warning C4702: unreachable code

namespace cuda::experimental::execution
{
namespace __detail
{
struct __env_proxy : __immovable
{
  _CCCL_API virtual auto query(const get_stop_token_t&) const noexcept -> inplace_stop_token              = 0;
  _CCCL_API virtual auto query(const get_allocator_t&) const noexcept -> any_allocator<::cuda::std::byte> = 0;
  _CCCL_API virtual auto query(const get_scheduler_t&) const noexcept -> task_scheduler                   = 0;
};
} // namespace __detail

class receiver_proxy : __detail::__env_proxy
{
public:
  _CCCL_API virtual ~receiver_proxy() = 0;

  _CCCL_API virtual void set_value() noexcept                = 0;
  _CCCL_API virtual void set_error(exception_ptr&&) noexcept = 0;
  _CCCL_API virtual void set_stopped() noexcept              = 0;

  [[nodiscard]]
  _CCCL_API auto get_env() const noexcept -> const __detail::__env_proxy&
  {
    return *this;
  }

  // _CCCL_EXEC_CHECK_DISABLE
  // _CCCL_TEMPLATE(class _Value, class Query)
  // _CCCL_REQUIRES(__callable<const __detail::__try_queryable&, Query, ::cuda::std::optional<_Value>&>)
  // [[nodiscard]] _CCCL_API auto try_query(const Query& __query) const noexcept -> ::cuda::std::optional<_Value>
  // {
  //   const __detail::__try_queryable& __queryable = *this;
  //   ::cuda::std::optional<_Value> __value;
  //   __queryable(__query, __value);
  //   return __value;
  // }
};

inline receiver_proxy::~receiver_proxy() = default;

struct bulk_item_receiver_proxy : receiver_proxy
{
  _CCCL_API virtual void execute(size_t, size_t) noexcept = 0;
};

struct parallel_scheduler_backend
{
  _CCCL_API virtual ~parallel_scheduler_backend() = 0;

  _CCCL_API virtual void schedule(receiver_proxy&, ::cuda::std::span<::cuda::std::byte>) noexcept = 0;

  _CCCL_API virtual void
  schedule_bulk_chunked(size_t, bulk_item_receiver_proxy&, ::cuda::std::span<::cuda::std::byte>) noexcept = 0;

  _CCCL_API virtual void
  schedule_bulk_unchunked(size_t, bulk_item_receiver_proxy&, ::cuda::std::span<::cuda::std::byte>) noexcept = 0;
};

inline parallel_scheduler_backend::~parallel_scheduler_backend() = default;

namespace __detail
{
// Partially implements the _RcvrProxy interface (either receiver_proxy or
// bulk_item_receiver_proxy) in terms of a concrete receiver type _Rcvr.
template <class _Rcvr, class _RcvrProxy>
struct __receiver_proxy_base : _RcvrProxy
{
public:
  using receiver_concept = receiver_t;

  _CCCL_API explicit __receiver_proxy_base(_Rcvr rcvr) noexcept
      : __rcvr_(static_cast<_Rcvr&&>(rcvr))
  {}

  _CCCL_API void set_error(exception_ptr&& eptr) noexcept final override
  {
    execution::set_error(_CCCL_MOVE(__rcvr_), _CCCL_MOVE(eptr));
  }

  _CCCL_API void set_stopped() noexcept final override
  {
    execution::set_stopped(_CCCL_MOVE(__rcvr_));
  }

protected:
  _CCCL_API auto query(const get_stop_token_t&) const noexcept -> inplace_stop_token final override
  {
    if constexpr (__callable<const get_stop_token_t&, env_of_t<_Rcvr>>)
    {
      if constexpr (__same_as<stop_token_of_t<env_of_t<_Rcvr>>, inplace_stop_token>)
      {
        return get_stop_token(get_env(__rcvr_));
      }
    }
    return inplace_stop_token{}; // MSVC thinks this is unreachable. :-?
  }

  _CCCL_API auto query(const get_allocator_t&) const noexcept -> any_allocator<::cuda::std::byte> final override
  {
    return any_allocator{get_allocator(get_env(__rcvr_))};
  }

  // defined in task_scheduler.cuh:
  _CCCL_API auto query(const get_scheduler_t& __query) const noexcept -> task_scheduler final override;

  _Rcvr __rcvr_;
};

template <class _Rcvr>
struct __receiver_proxy : __receiver_proxy_base<_Rcvr, receiver_proxy>
{
  using __receiver_proxy_base<_Rcvr, receiver_proxy>::__receiver_proxy_base;

  _CCCL_API void set_value() noexcept final override
  {
    execution::set_value(_CCCL_MOVE(this->__rcvr_));
  }
};

// A receiver type that forwards its completion operations to a _RcvrProxy member held by
// reference (where _RcvrProxy is one of receiver_proxy or bulk_item_receiver_proxy). It
// is also responsible to destroying and, if necessary, deallocating the operation state.
template <class _RcvrProxy>
struct __proxy_receiver
{
  using receiver_concept = receiver_t;
  using __delete_fn_t    = void(void*) noexcept;

  _CCCL_API void set_value() noexcept
  {
    auto& __proxy = __rcvr_proxy_;
    __delete_fn_(__opstate_storage_);
    __proxy.set_value();
  }

  _CCCL_API void set_error(exception_ptr eptr) noexcept
  {
    auto& __proxy = __rcvr_proxy_;
    __delete_fn_(__opstate_storage_);
    __proxy.set_error(_CCCL_MOVE(eptr));
  }

  _CCCL_API void set_stopped() noexcept
  {
    auto& __proxy = __rcvr_proxy_;
    __delete_fn_(__opstate_storage_);
    __proxy.set_stopped();
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> env_of_t<_RcvrProxy>
  {
    return execution::get_env(__rcvr_proxy_);
  }

  _RcvrProxy& __rcvr_proxy_;
  void* __opstate_storage_;
  __delete_fn_t* __delete_fn_;
};
} // namespace __detail
} // namespace cuda::experimental::execution

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_PARALLEL_SCHEDULER
