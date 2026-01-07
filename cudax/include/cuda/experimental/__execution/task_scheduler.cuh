//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_TASK_SCHEDULER
#define __CUDAX_EXECUTION_TASK_SCHEDULER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/allocator.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/bulk.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/diagnostics.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/inline_scheduler.cuh>
#include <cuda/experimental/__execution/parallel_scheduler_backend.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__utility/shared_ptr.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct task_scheduler;

struct task_scheduler_domain;

namespace __detail
{
// The concrete type-erased sender returned by task_scheduler::schedule()
struct __task_sender;

template <class _Sndr>
struct __task_bulk_sender;

template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
class __task_bulk_state;

template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
struct __task_bulk_receiver;

struct __task_scheduler_backend : parallel_scheduler_backend
{
  _CCCL_API virtual auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee = 0;
  _CCCL_API virtual auto __equal_to(const void* __other, ::cuda::std::__type_info_ref __type) -> bool         = 0;
};

using __backend_ptr_t = __shared_ptr<__task_scheduler_backend>;

template <class _Sch>
_CCCL_CONCEPT __non_task_scheduler = _CCCL_REQUIRES_EXPR((_Sch))( //
  requires(__not_same_as<task_scheduler, _Sch>), //
  requires(scheduler<_Sch>));
} // namespace __detail

struct _CANNOT_DISPATCH_BULK_ALGORITHM_TO_TASK_SCHEDULER_BECAUSE_THERE_IS_NO_TASK_SCHEDULER_IN_THE_ENVIRONMENT;
struct _ADD_A_CONTINUES_ON_TRANSITION_TO_THE_TASK_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

struct task_scheduler_domain : default_domain
{
  _CCCL_TEMPLATE(class _Sndr, class _Env, class _BulkTag = tag_of_t<_Sndr>)
  _CCCL_REQUIRES(__one_of<_BulkTag, bulk_chunked_t, bulk_unchunked_t>)
  [[nodiscard]] _CCCL_API static constexpr auto transform_sender(set_value_t, _Sndr&& __sndr, const _Env& __env)
  {
    using __sched_t =
      __call_result_or_t<get_completion_scheduler_t<set_value_t>, __not_a_scheduler<>, env_of_t<_Sndr>, const _Env&>;
    if constexpr (!__same_as<__sched_t, task_scheduler>)
    {
      return __not_a_sender<
        _WHERE(_IN_ALGORITHM, _BulkTag),
        _WHAT(_CANNOT_DISPATCH_BULK_ALGORITHM_TO_TASK_SCHEDULER_BECAUSE_THERE_IS_NO_TASK_SCHEDULER_IN_THE_ENVIRONMENT),
        _TO_FIX_THIS_ERROR(_ADD_A_CONTINUES_ON_TRANSITION_TO_THE_TASK_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
        _WITH_SENDER(_Sndr),
        _WITH_ENVIRONMENT(_Env)>{};
    }
    else
    {
      auto __sch = get_completion_scheduler<set_value_t>(get_env(__sndr), __env);
      return __detail::__task_bulk_sender<_Sndr>{static_cast<_Sndr&&>(__sndr), _CCCL_MOVE(__sch)};
    }
  }
};

//! @brief A type-erased scheduler.
//!
//! The `task_scheduler` struct is implemented in terms of a backend type derived from
//! @c parallel_scheduler_backend, providing a type-erased interface for scheduling tasks.
//! It exposes query functions to retrieve the completion scheduler and domain.
//!
//! @note This scheduler is designed for use with CUDA experimental execution APIs.
//!
//! @see parallel_scheduler_backend
class _CCCL_TYPE_VISIBILITY_DEFAULT task_scheduler
{
  template <class _Sch, class _Alloc>
  class _CCCL_TYPE_VISIBILITY_DEFAULT __backend_for;

public:
  using scheduler_concept = scheduler_t;

  _CCCL_TEMPLATE(class _Sch, class _Alloc = ::cuda::std::allocator<::cuda::std::byte>)
  _CCCL_REQUIRES(__detail::__non_task_scheduler<_Sch>)
  _CCCL_API explicit task_scheduler(_Sch __sch, _Alloc __alloc = {})
      : __backend_(experimental::__allocate_shared<__backend_for<_Sch, _Alloc>>(__alloc, _CCCL_MOVE(__sch), __alloc))
  {}

  [[nodiscard]] _CCCL_API auto schedule() const noexcept -> __detail::__task_sender;

  [[nodiscard]] _CCCL_API friend bool operator==(const task_scheduler& __lhs, const task_scheduler& __rhs) noexcept
  {
    return __lhs.__backend_ == __rhs.__backend_;
  }

  [[nodiscard]] _CCCL_API friend bool operator!=(const task_scheduler& __lhs, const task_scheduler& __rhs) noexcept
  {
    return !(__lhs.__backend_ == __rhs.__backend_);
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_API friend auto operator==(const task_scheduler& __lhs, const _Sch& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return __lhs.__backend_->__equal_to(::cuda::std::addressof(__rhs), _CCCL_TYPEID(_Sch));
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_API friend auto operator!=(const task_scheduler& __lhs, const _Sch& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return !(__lhs == __rhs);
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_API friend auto operator==(const _Sch& __lhs, const task_scheduler& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return __rhs == __lhs;
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_API friend auto operator!=(const _Sch& __lhs, const task_scheduler& __rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return !(__rhs == __lhs);
  }

  [[nodiscard]] _CCCL_API auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee
  {
    return __backend_->query(get_forward_progress_guarantee_t{});
  }

  [[nodiscard]] _CCCL_API auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> const task_scheduler&
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept
  {
    return task_scheduler_domain{};
  }

private:
  template <class>
  friend struct __detail::__task_bulk_sender;
  friend struct __detail::__task_sender;

  __detail::__backend_ptr_t __backend_;
};

namespace __detail
{
//! @brief A type-erased opstate returned when connecting the result of
//! task_scheduler::schedule() to a receiver.
template <class _Rcvr>
class __task_opstate_t
{
public:
  using operation_state_concept = operation_state_t;

  _CCCL_API __task_opstate_t(__backend_ptr_t __backend, _Rcvr __rcvr)
      : __rcvr_proxy_(_CCCL_MOVE(__rcvr))
      , __backend_(_CCCL_MOVE(__backend))
  {}

  _CCCL_API void start() noexcept
  {
    _CCCL_TRY
    {
      __backend_->schedule(__rcvr_proxy_, ::cuda::std::span{__storage_});
    }
    _CCCL_CATCH_ALL
    {
      __rcvr_proxy_.set_error(execution::current_exception());
    }
  }

private:
  __detail::__receiver_proxy<_Rcvr> __rcvr_proxy_;
  __backend_ptr_t __backend_;
  ::cuda::std::byte __storage_[8 * sizeof(void*)];
};

//! @brief A type-erased sender returned by task_scheduler::schedule().
struct __task_sender
{
  using sender_concept = sender_t;
  using __completions_t =
    completion_signatures<set_value_t(), //
                          set_error_t(exception_ptr),
                          set_error_t(cudaError_t),
                          set_stopped_t()>;

  _CCCL_API explicit __task_sender(task_scheduler __sch)
      : __attrs_{_CCCL_MOVE(__sch)}
  {}

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) const noexcept -> __task_opstate_t<_Rcvr>
  {
    return __task_opstate_t<_Rcvr>(get_completion_scheduler<set_value_t>(__attrs_).__backend_, _CCCL_MOVE(__rcvr));
  }

  template <class _Self>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept -> __completions_t
  {
    return {};
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> const __sch_attrs_t<task_scheduler>&
  {
    return __attrs_;
  }

private:
  __sch_attrs_t<task_scheduler> __attrs_;
};

//! @brief A receiver used to connect the predecessor of a bulk operation launched by a
//! task_scheduler. Its set_value member stores the predecessor's values in the bulk
//! operation state and then starts the bulk operation.
template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
struct __task_bulk_receiver
{
  using receiver_concept = receiver_t;

  template <class... _As>
  _CCCL_API void set_value(_As&&... __as) noexcept
  {
    _CCCL_TRY
    {
      // Store the predecessor's values in the bulk operation state.
      using __values_t = ::cuda::std::__decayed_tuple<_As...>;
      __state_->__values_.template __emplace<__values_t>(static_cast<_As&&>(__as)...);

      // Start the bulk operation.
      if constexpr (__same_as<_BulkTag, bulk_chunked_t>)
      {
        __state_->__backend_->schedule_bulk_chunked(
          __state_->__shape_, *__state_, ::cuda::std::span{__state_->__storage_});
      }
      else
      {
        __state_->__backend_->schedule_bulk_unchunked(
          __state_->__shape_, *__state_, ::cuda::std::span{__state_->__storage_});
      }
    }
    _CCCL_CATCH_ALL
    {
      execution::set_error(_CCCL_MOVE(__state_->__rcvr_), execution::current_exception());
    }
  }

  template <class _Error>
  _CCCL_API void set_error(_Error&& __err) noexcept
  {
    execution::set_error(_CCCL_MOVE(__state_->__rcvr_), static_cast<_Error&&>(__err));
  }

  _CCCL_API void set_stopped() noexcept
  {
    execution::set_stopped(_CCCL_MOVE(__state_->__rcvr_));
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> env_of_t<_Rcvr>
  {
    return execution::get_env(__state_->__rcvr_);
  }

  __task_bulk_state<_BulkTag, _Policy, _Fn, _Rcvr, _Values>* __state_;
};

//! Returns a visitor (callable) used to invoke the bulk (unchunked) function with the
//! predecessor's values, which are stored in a variant in the bulk operation state.
template <bool _Parallelize, class _Fn>
[[nodiscard]] _CCCL_API constexpr auto
__get_execute_bulk_fn(bulk_unchunked_t, _Fn& __fn, size_t __shape, size_t __begin, size_t) noexcept
{
  return [=, &__fn](auto& __args) {
    constexpr bool __valid_args = !__same_as<decltype(__args), ::cuda::std::monostate&>;
    // runtime assert that we never take this path without valid args from the predecessor:
    _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

    if constexpr (__valid_args)
    {
      // If we are not parallelizing, we need to run all the iterations sequentially.
      const size_t __increments = _Parallelize ? 1 : __shape;
      // Precompose the function with the arguments so we don't have to do it every iteration.
      auto __precomposed_fn = ::cuda::std::__apply(
        [&](auto&... __as) {
          return [&](size_t __i) -> void {
            __fn(__i, __as...);
          };
        },
        __args);
      for (size_t __i = __begin; __i < __begin + __increments; ++__i)
      {
        __precomposed_fn(__i);
      }
    }
  };
}

template <bool _Parallelize, class _Fn>
struct __apply_bulk_execute
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class... _As>
  _CCCL_API void operator()(_As&... __as) const noexcept(__nothrow_callable<_Fn&, size_t, _As&...>)
  {
    if constexpr (_Parallelize)
    {
      __fn_(__begin_, __end_, __as...);
    }
    else
    {
      // If we are not parallelizing, we need to pass the entire range to the functor.
      __fn_(size_t(0), __shape_, __as...);
    }
  }

  size_t __begin_, __end_, __shape_;
  _Fn& __fn_;
};

//! Returns a visitor (callable) used to invoke the bulk (chunked) function with the
//! predecessor's values, which are stored in a variant in the bulk operation state.
template <bool _Parallelize, class _Fn>
[[nodiscard]] _CCCL_API constexpr auto
__get_execute_bulk_fn(bulk_chunked_t, _Fn& __fn, size_t __shape, size_t __begin, size_t __end) noexcept
{
  return [=, &__fn](auto& __args) {
    constexpr bool __valid_args = !__same_as<decltype(__args), ::cuda::std::monostate&>;
    _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

    if constexpr (__valid_args)
    {
      ::cuda::std::__apply(__apply_bulk_execute<_Parallelize, _Fn>{__begin, __end, __shape, __fn}, __args);
    }
  };
}

//! Stores the state for a bulk operation launched by a task_scheduler. A type-erased
//! reference to this object is passed to either the task_scheduler's
//! schedule_bulk_chunked or schedule_bulk_unchunked methods, which is expected to call
//! execute(begin, end) on it to run the bulk operation. After the bulk operation is
//! complete, set_value is called, which forwards the predecessor's values to the
//! downstream receiver.
template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
class __task_bulk_state : public __detail::__receiver_proxy_base<_Rcvr, bulk_item_receiver_proxy>
{
public:
  _CCCL_API explicit __task_bulk_state(_Rcvr __rcvr, size_t __shape, _Fn __fn, __backend_ptr_t __backend)
      : __task_bulk_state::__receiver_proxy_base(_CCCL_MOVE(__rcvr))
      , __fn_(_CCCL_MOVE(__fn))
      , __shape_(__shape)
      , __backend_(_CCCL_MOVE(__backend))
  {}

  _CCCL_API void set_value() noexcept final override
  {
    // Send the stored values to the downstream receiver.
    __visit(
      [this](auto& __tupl) {
        constexpr bool __valid_args = __not_same_as<decltype(__tupl), ::cuda::std::monostate&>;
        // runtime assert that we never take this path without valid args from the predecessor:
        _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

        if constexpr (__valid_args)
        {
          ::cuda::std::__apply(execution::set_value, _CCCL_MOVE(__tupl), _CCCL_MOVE(this->__rcvr_));
        }
      },
      __values_);
  }

  //! Actually runs the bulk operation over the specified range.
  _CCCL_API void execute(size_t __begin, size_t __end) noexcept final override
  {
    _CCCL_TRY
    {
      constexpr bool __parallelize = _Policy() == par || _Policy() == par_unseq;
      __visit(__detail::__get_execute_bulk_fn<__parallelize>(_BulkTag(), __fn_, __shape_, __begin, __end), __values_);
    }
    _CCCL_CATCH_ALL
    {
      execution::set_error(_CCCL_MOVE(this->__rcvr_), execution::current_exception());
    }
  }

private:
  template <class, class, class, class, class>
  friend struct __task_bulk_receiver;

  _Fn __fn_;
  size_t __shape_;
  _Values __values_{};
  __backend_ptr_t __backend_;
  ::cuda::std::byte __storage_[8 * sizeof(void*)];
};

////////////////////////////////////////////////////////////////////////////////////
// Operation state for task scheduler bulk operations
template <class _BulkTag, class _Policy, class _Sndr, class _Fn, class _Rcvr>
struct __task_bulk_opstate
{
  using operation_state_concept = operation_state_t;

  _CCCL_API explicit __task_bulk_opstate(
    _Sndr&& __sndr, size_t __shape, _Fn __fn, _Rcvr __rcvr, __backend_ptr_t __backend)
      : __state_{_CCCL_MOVE(__rcvr), __shape, _CCCL_MOVE(__fn), _CCCL_MOVE(__backend)}
      , __opstate1_(execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state_}))
  {}

  _CCCL_API void start() noexcept
  {
    execution::start(__opstate1_);
  }

private:
  using __values_t =
    value_types_of_t<_Sndr, __fwd_env_t<env_of_t<_Rcvr>>, ::cuda::std::__decayed_tuple, __nullable_variant>;
  using __rcvr_t     = __task_bulk_receiver<_BulkTag, _Policy, _Fn, _Rcvr, __values_t>;
  using __opstate1_t = connect_result_t<_Sndr, __rcvr_t>;

  __task_bulk_state<_BulkTag, _Policy, _Fn, _Rcvr, __values_t> __state_;
  __opstate1_t __opstate1_;
};

template <class _Sndr>
struct __task_bulk_sender
{
  using sender_concept = sender_t;

  _CCCL_API explicit __task_bulk_sender(_Sndr __sndr, task_scheduler __sch)
      : __sndr_(_CCCL_MOVE(__sndr))
      , __attrs_{_CCCL_MOVE(__sch)}
  {}

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) &&
  {
    auto& [__tag, __data, __child] = __sndr_;
    auto& [__pol, __shape, __fn]   = __data;
    return __task_bulk_opstate<decltype(__tag), decltype(__pol), decltype(__child), decltype(__fn), _Rcvr>{
      _CCCL_MOVE(__child),
      static_cast<size_t>(__shape),
      _CCCL_MOVE(__fn),
      _CCCL_MOVE(__rcvr),
      _CCCL_MOVE(__attrs_.__sch_.__backend_)};
  }

  _CCCL_TEMPLATE(class _Self, class _Env)
  _CCCL_REQUIRES(__same_as<_Self, __task_bulk_sender>) // accept only rvalues.
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    // This calls get_completion_signatures on the wrapped bulk_[un]chunked sender. We
    // call it directly instead of using execution::get_completion_signatures to avoid
    // another trip through transform_sender, which would lead to infinite recursion.
    auto __completions = decay_t<_Sndr>::template get_completion_signatures<_Sndr, _Env>();
    return transform_completion_signatures(__completions, __decay_transform<set_value_t>(), {}, {}, __eptr_completion());
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> const __sch_attrs_t<task_scheduler>&
  {
    return __attrs_;
  }

private:
  _Sndr __sndr_;
  __sch_attrs_t<task_scheduler> __attrs_;
};

//! Function called by the `bulk_chunked` operation; calls `execute` on the bulk_item_receiver_proxy.
struct __bulk_chunked_fn
{
  _CCCL_API void operator()(size_t __begin, size_t __end) noexcept
  {
    __rcvr_.execute(__begin, __end);
  }

  bulk_item_receiver_proxy& __rcvr_;
};

//! Function called by the `bulk_unchunked` operation; calls `execute` on the bulk_item_receiver_proxy.
struct __bulk_unchunked_fn
{
  _CCCL_API void operator()(size_t __idx) noexcept
  {
    __rcvr_.execute(__idx, __idx + 1);
  }

  bulk_item_receiver_proxy& __rcvr_;
};

template <class _Ty, class _Alloc, class... _Args>
_CCCL_API auto __emplace_into(::cuda::std::span<::cuda::std::byte> __storage, _Alloc& __alloc, _Args&&... __args)
  -> _Ty&
{
  using __traits_t = ::cuda::std::allocator_traits<__rebind_alloc_t<_Alloc, _Ty>>;
  __rebind_alloc_t<_Alloc, _Ty> __alloc_copy{__alloc};

  const bool __in_situ = __storage.size() >= sizeof(_Ty);
  auto* __ty_ptr       = __in_situ ? reinterpret_cast<_Ty*>(__storage.data()) : __traits_t::allocate(__alloc_copy, 1);
  __traits_t::construct(__alloc_copy, __ty_ptr, static_cast<_Args&&>(__args)...);
  return *::cuda::std::launder(__ty_ptr);
}

template <class _Alloc, class _Sndr>
class __opstate_t : _Alloc
{
public:
  using allocator_type = _Alloc;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API __opstate_t(_Alloc __alloc, _Sndr __sndr, receiver_proxy& __rcvr_proxy, bool __in_situ)
      : _Alloc(_CCCL_MOVE(__alloc))
      , __opstate_(execution::connect(
          _CCCL_MOVE(__sndr),
          __detail::__proxy_receiver<receiver_proxy>{
            __rcvr_proxy, this, __in_situ ? __delete_opstate<true> : __delete_opstate<false>}))
  {}
  __opstate_t(__opstate_t&&) = delete;

  _CCCL_API void start() noexcept
  {
    execution::start(__opstate_);
  }

  [[nodiscard]] _CCCL_API auto query(get_allocator_t) const noexcept -> const _Alloc&
  {
    return *this;
  }

private:
  template <bool _InSitu>
  _CCCL_API static void __delete_opstate(void* __ptr) noexcept
  {
    using __traits_t = ::cuda::std::allocator_traits<__rebind_alloc_t<_Alloc, __opstate_t>>;
    auto* __opstate  = static_cast<__opstate_t*>(__ptr);
    __rebind_alloc_t<_Alloc, __opstate_t> __alloc_copy{get_allocator(*__opstate)};

    __traits_t::destroy(__alloc_copy, __opstate);
    if constexpr (!_InSitu)
    {
      __traits_t::deallocate(__alloc_copy, __opstate, 1);
    }
  }

  using __child_opstate_t = connect_result_t<_Sndr, __detail::__proxy_receiver<receiver_proxy>>;
  __child_opstate_t __opstate_;
};
} // namespace __detail

[[nodiscard]] _CCCL_API inline auto task_scheduler::schedule() const noexcept -> __detail::__task_sender
{
  return __detail::__task_sender{*this};
}

template <class _Sch, class _Alloc>
class _CCCL_DECLSPEC_EMPTY_BASES task_scheduler::__backend_for
    : public __detail::__task_scheduler_backend
    , _Alloc
{
  template <class _RcvrProxy>
  friend struct __detail::__proxy_receiver;

  template <class _RcvrProxy, class _Sndr>
  _CCCL_API void
  __schedule(_RcvrProxy& __rcvr_proxy, _Sndr&& __sndr, ::cuda::std::span<::cuda::std::byte> __storage) noexcept
  {
    _CCCL_TRY
    {
      using __opstate_t    = connect_result_t<_Sndr, __detail::__proxy_receiver<_RcvrProxy>>;
      const bool __in_situ = __storage.size() >= sizeof(__opstate_t);
      _Alloc& __alloc      = *this;
      auto& __opstate      = __detail::__emplace_into<__detail::__opstate_t<_Alloc, _Sndr>>(
        __storage, __alloc, __alloc, static_cast<_Sndr&&>(__sndr), __rcvr_proxy, __in_situ);
      execution::start(__opstate);
    }
    _CCCL_CATCH_ALL
    {
      __rcvr_proxy.set_error(execution::current_exception());
    }
  }

public:
  _CCCL_API explicit __backend_for(_Sch __sch, _Alloc __alloc)
      : _Alloc(_CCCL_MOVE(__alloc))
      , __sch_(_CCCL_MOVE(__sch))
  {}

  _CCCL_API void schedule(receiver_proxy& __rcvr_proxy,
                          ::cuda::std::span<::cuda::std::byte> __storage) noexcept final override
  {
    __schedule(__rcvr_proxy, execution::schedule(__sch_), __storage);
  }

  _CCCL_API void schedule_bulk_chunked(size_t __size,
                                       bulk_item_receiver_proxy& __rcvr_proxy,
                                       ::cuda::std::span<::cuda::std::byte> __storage) noexcept final override
  {
    auto __sndr =
      execution::bulk_chunked(execution::schedule(__sch_), par, __size, __detail::__bulk_chunked_fn{__rcvr_proxy});
    __schedule(__rcvr_proxy, _CCCL_MOVE(__sndr), __storage);
  }

  _CCCL_API void schedule_bulk_unchunked(size_t __size,
                                         bulk_item_receiver_proxy& __rcvr_proxy,
                                         ::cuda::std::span<::cuda::std::byte> __storage) noexcept override
  {
    auto __sndr =
      execution::bulk_unchunked(execution::schedule(__sch_), par, __size, __detail::__bulk_unchunked_fn{__rcvr_proxy});
    __schedule(__rcvr_proxy, _CCCL_MOVE(__sndr), __storage);
  }

  [[nodiscard]]
  _CCCL_API auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee final override
  {
    return get_forward_progress_guarantee(__sch_);
  }

  [[nodiscard]]
  _CCCL_API bool __equal_to(const void* __other, ::cuda::std::__type_info_ref __type) final override
  {
    if (__type == _CCCL_TYPEID(_Sch))
    {
      const _Sch& __other_sch = *static_cast<const _Sch*>(__other);
      return __sch_ == __other_sch;
    }
    return false;
  }

private:
  _Sch __sch_;
};

namespace __detail
{
// Implementation of the get_scheduler_t query for __proxy_receiver_impl from
// parallel_scheduler_backend.cuh
template <class _Rcvr, class _Proxy>
_CCCL_API auto __receiver_proxy_base<_Rcvr, _Proxy>::query(const get_scheduler_t&) const noexcept -> task_scheduler
{
  if constexpr (__callable<const get_scheduler_t&, env_of_t<_Rcvr>>)
  {
    return task_scheduler{get_scheduler(get_env(__rcvr_))};
  }
  else
  {
    return task_scheduler{inline_scheduler{}};
  }
}
} // namespace __detail
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TASK_SCHEDULER
