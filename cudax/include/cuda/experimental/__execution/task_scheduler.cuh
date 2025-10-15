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

#include <cuda/__utility/basic_any.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__memory/allocator.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/bulk.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/variant.cuh>

#if !_CCCL_COMPILER(NVRTC)
#  include <exception> // IWYU pragma: keep
#  include <system_error> // IWYU pragma: keep
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct task_scheduler;

struct task_scheduler_domain;

namespace __detail
{
//! Pointers to this function are used as entries in the vtable for
//! the environment queries of __basic_any<__ireceiver<>&>.
template <class _Rcvr, class _Value, class _Query, bool _Nothrow>
[[nodiscard]] _CCCL_PUBLIC_API auto __try_query_vfn_impl(const _Rcvr& __rcvr) noexcept(_Nothrow)
  -> cuda::std::optional<_Value>
{
  if constexpr (__callable<_Query, env_of_t<const _Rcvr&>>)
  {
    static_assert(__nothrow_callable<_Query, env_of_t<const _Rcvr&>> || !_Nothrow,
                  "The noexcept specification of the try_query_vfn must match that of the query.");
    if constexpr (cuda::std::convertible_to<__call_result_t<_Query, env_of_t<const _Rcvr&>>, _Value>)
    {
      return cuda::std::optional<_Value>{_Query()(get_env(__rcvr))};
    }
  }
  return cuda::std::nullopt;
}

template <class _Rcvr, class _QuerySignature>
extern ::cuda::std::__undefined<_QuerySignature> __try_query_vfn;

template <class _Rcvr, class _Value, class _Query>
constexpr auto __try_query_vfn<_Rcvr, _Value(_Query)> = &__try_query_vfn_impl<_Rcvr, _Value, _Query, false>;

template <class _Rcvr, class _Value, class _Query>
constexpr auto __try_query_vfn<_Rcvr, _Value(_Query) noexcept> = &__try_query_vfn_impl<_Rcvr, _Value, _Query, true>;

//! Pointers to this function are used as entries in the vtable for
//! the completion functions of __basic_any<__ireceiver<>&>.
_CCCL_TEMPLATE(class _Rcvr, class _Tag, class... _Args)
_CCCL_REQUIRES(__callable<_Tag, _Rcvr, _Args...>)
_CCCL_PUBLIC_API void __complete_vfn(_Rcvr& __rcvr, _Args... __args) noexcept
{
  _Tag{}(static_cast<_Rcvr&&>(__rcvr), static_cast<_Args&&>(__args)...);
}

//! Backing implementation for the __ireceiver interface, a type-erased receiver interface
//! supporting a specified set of environment query signatures.
template <class... _QuerySignatures>
struct __ircvr
{
  template <class _Query>
  static constexpr bool __has_query = __one_of<_Query, _QuerySignatures...>;

  template <class...>
  struct __iface : __basic_interface<__iface>
  {
    using receiver_concept = receiver_t;

    template <class _Rcvr = __iface<>>
    using __set_value_t = void(_Rcvr&) noexcept;

    template <class _Rcvr = __iface<>>
    using __set_error_t = void(_Rcvr&, ::std::exception_ptr&&) noexcept;

    template <class _Rcvr = __iface<>>
    using __set_stopped_t = void(_Rcvr&) noexcept;

    _CCCL_API void set_value() noexcept
    {
      constexpr __set_value_t<>* __vfn = &__complete_vfn<__iface<>, set_value_t>;
      ::cuda::__virtcall<__vfn>(this);
    }

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API void set_error(::std::exception_ptr&& __eptr) noexcept
    {
      constexpr __set_error_t<>* __vfn = &__complete_vfn<__iface<>, set_error_t, ::std::exception_ptr&&>;
      ::cuda::__virtcall<__vfn>(this, _CCCL_MOVE(__eptr));
    }

    _CCCL_API void set_stopped() noexcept
    {
      constexpr __set_stopped_t<>* __vfn = &__complete_vfn<__iface<>, set_stopped_t>;
      ::cuda::__virtcall<__vfn>(this);
    }

    template <class _Value, class _Query>
    _CCCL_API auto try_query(_Query) const
      noexcept(__has_query<_Value(_Query) noexcept> || !__has_query<_Value(_Query)>) -> cuda::std::optional<_Value>
    {
      if constexpr (__has_query<_Value(_Query) noexcept>)
      {
        return ::cuda::__virtcall<__try_query_vfn<__iface<>, _Value(_Query) noexcept>>(this);
      }
      else if constexpr (__has_query<_Value(_Query)>)
      {
        return ::cuda::__virtcall<__try_query_vfn<__iface<>, _Value(_Query)>>(this);
      }
      return cuda::std::nullopt;
    }

    // GCC needs the static_casts below to resolve the overloads correctly.
    template <class _Rcvr>
    using overrides =
      __overrides_for<_Rcvr,
                      static_cast<__set_value_t<_Rcvr>*>(&__complete_vfn<_Rcvr, set_value_t>),
                      static_cast<__set_error_t<_Rcvr>*>(&__complete_vfn<_Rcvr, set_error_t, ::std::exception_ptr&&>),
                      static_cast<__set_stopped_t<_Rcvr>*>(&__complete_vfn<_Rcvr, set_stopped_t>),
                      __try_query_vfn<_Rcvr, _QuerySignatures>...>;
  };
};

//! A type-erased receiver interface supporting the specified set of query signatures.
template <class... _QuerySignatures>
using __ireceiver = typename __ircvr<_QuerySignatures...>::template __iface<>;

//! Similar to std::execution::receiver_proxy. See https://eel.is/c++draft/exec#sysctxrepl.query-4
using __ireceiver_default = __ireceiver<inplace_stop_token(get_stop_token_t) noexcept>;

struct __default_task_receiver_ref : __basic_any<__ireceiver_default&>
{
  using __basic_any<__ireceiver_default&>::__basic_any;
};

//! The "virtual function" for `start` on a type-erased operation state.
_CCCL_TEMPLATE(class _OpState)
_CCCL_REQUIRES(__callable<start_t, _OpState&>)
_CCCL_PUBLIC_API void __start_vfn(_OpState& __opstate) noexcept
{
  start(__opstate);
}

//! @brief Represents an interface for operation state objects in the CUDA experimental task scheduler.
//!
//! This struct inherits from `__basic_interface<__iopstate>` and provides a virtualized `start()` method
//! for initiating the operation state. It also defines a type alias for the operation state concept and
//! a template alias for overrides.
//!
//! @tparam ... Variadic template parameters for customization.
//!
//! Members:
//! - operation_state_concept: Alias for the operation state type.
//! - start(): Initiates the operation state; uses virtual dispatch.
//! - overrides: Template alias for specifying overrides for operation state types.
template <class...>
struct __iopstate : __basic_interface<__iopstate>
{
  using operation_state_concept = operation_state_t;

  _CCCL_API void start() noexcept
  {
    ::cuda::__virtcall<&__start_vfn<__iopstate>>(this);
  }

  template <class _OpState>
  using overrides = __overrides_for<_OpState, &__start_vfn<_OpState>>;
};

// The "virtual function" for `connect` on a __task_sender.
template <class _Sndr>
_CCCL_PUBLIC_API void __connect_vfn(_Sndr& __sndr, __default_task_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op)
{
  using __opstate_t = connect_result_t<_Sndr, __default_task_receiver_ref>;
  __op.emplace_from<__opstate_t>(execution::connect, static_cast<_Sndr&&>(__sndr), _CCCL_MOVE(__rcvr));
}

template <class _Rcvr>
struct __any_opstate
{
  template <class _ISndr>
  _CCCL_API __any_opstate(_ISndr* __isndr, _Rcvr __rcvr)
      : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
  {
    ::cuda::__virtcall<&__connect_vfn<_ISndr>>(__isndr, __default_task_receiver_ref(__rcvr_), __opstate_);
  }

  _CCCL_API void start() noexcept
  {
    __opstate_.start();
  }

private:
  _Rcvr __rcvr_;
  __basic_any<__iopstate<>> __opstate_;
};

// The interface for a type-erased sender with the completion signatures needed for a
// scheduler sender.
template <class...>
struct __itask_sender : ::cuda::__basic_interface<__itask_sender, __extends<__imovable<>>>
{
  using sender_concept  = sender_t;
  using __completions_t = completion_signatures<set_value_t(), set_error_t(::std::exception_ptr), set_stopped_t()>;

  _CCCL_TEMPLATE(class _Rcvr)
  _CCCL_REQUIRES(receiver_of<_Rcvr, __completions_t>)
  [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) && -> __any_opstate<_Rcvr>
  {
    return __any_opstate<_Rcvr>(this, static_cast<_Rcvr&&>(__rcvr));
  }

  template <class _Self>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    return __completions_t{};
  }

  template <class _Sndr>
  using overrides = __overrides_for<_Sndr, &__connect_vfn<_Sndr>>;
};

template <class...>
struct __ibulk_item_receiver : __basic_interface<__ibulk_item_receiver, __extends<__ireceiver_default>>
{
  _CCCL_API void execute(size_t __begin, size_t __end) noexcept
  {
    ::cuda::__virtcall<&__ibulk_item_receiver::execute>(this, __begin, __end);
  }

  template <class _Rcvr>
  using overrides = __overrides_for<_Rcvr, &_Rcvr::execute>;
};

using __bulk_item_receiver_ref = __basic_any<__ibulk_item_receiver<>&>;

// The concrete type-erased sender returned by task_scheduler::schedule()
struct __task_sender;

template <class _Sndr>
struct __task_bulk_sender;

// The "virtual function" for task_scheduler::schedule()
template <class _Sch>
[[nodiscard]] _CCCL_PUBLIC_API auto __schedule_vfn(_Sch&, task_scheduler) noexcept -> __task_sender;

// The "virtual function" for task_scheduler::__schedule_bulk(bulk_chunked_t, ...)
template <class _Sch>
_CCCL_PUBLIC_API void __schedule_bulk_chunked_vfn(
  _Sch& __sch, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept;

// The "virtual function" for task_scheduler::__schedule_bulk(bulk_unchunked_t, ...)
template <class _Sch>
_CCCL_PUBLIC_API void __schedule_bulk_unchunked_vfn(
  _Sch& __sch, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept;

// The interface for a type-erased task_scheduler.
template <class...>
struct __itask_scheduler : __basic_interface<__itask_scheduler, __extends<__icopyable<>, __iequality_comparable<>>>
{
  using scheduler_concept = scheduler_t;

  _CCCL_API auto schedule() noexcept -> __task_sender;

  _CCCL_API void __schedule_bulk(
    bulk_chunked_t, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept
  {
    return ::cuda::__virtcall<&__schedule_bulk_chunked_vfn<__itask_scheduler>>(this, __size, __rcvr, __op);
  }

  _CCCL_API void __schedule_bulk(
    bulk_unchunked_t, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept
  {
    return ::cuda::__virtcall<&__schedule_bulk_unchunked_vfn<__itask_scheduler>>(this, __size, __rcvr, __op);
  }

  template <class _Sndr, class _Env>
  _CCCL_API auto __bulk_transform(_Sndr&& __sndr, const _Env&) -> __task_bulk_sender<_Sndr>
  {
    static_assert(__one_of<tag_of_t<_Sndr>, bulk_chunked_t, bulk_unchunked_t>);
    const auto& __base = ::cuda::__basic_any_from(*this);
    const auto& __self = static_cast<const task_scheduler&>(__base);
    return __task_bulk_sender<_Sndr>{static_cast<_Sndr&&>(__sndr), __self};
  }

  template <class _Sch>
  using overrides =
    __overrides_for<_Sch, //
                    &__schedule_vfn<_Sch>,
                    &__schedule_bulk_chunked_vfn<_Sch>,
                    &__schedule_bulk_unchunked_vfn<_Sch>>;
};

} // namespace __detail

struct task_scheduler_domain : default_domain
{
  _CCCL_TEMPLATE(class _Sndr, class _Env)
  _CCCL_REQUIRES(__one_of<tag_of_t<_Sndr>, bulk_chunked_t, bulk_unchunked_t> _CCCL_AND
                   __same_as<__call_result_t<get_completion_scheduler_t<set_value_t>, env_of_t<_Sndr>>, task_scheduler>)
  [[nodiscard]] _CCCL_API static constexpr auto transform_sender(set_value_t, _Sndr&& __sndr, const _Env& __env)
    -> __detail::__task_bulk_sender<_Sndr>
  {
    auto&& __sch = get_completion_scheduler<set_value_t>(get_env(__sndr));
    return __sch.__bulk_transform(static_cast<_Sndr&&>(__sndr), __env);
  }
};

//! @brief A type-erased scheduler.
//!
//! The `task_scheduler` struct inherits from `__basic_any<__detail::__itask_scheduler<>>`,
//! providing a type-erased interface for scheduling tasks. It exposes query functions
//! to retrieve the completion scheduler and domain.
//!
//! @note This scheduler is designed for use with CUDA experimental execution APIs.
//!
//! @see __basic_any
//! @see __detail::__itask_scheduler
struct task_scheduler : __basic_any<__detail::__itask_scheduler<>>
{
  using __basic_any<__detail::__itask_scheduler<>>::__basic_any;

  [[nodiscard]] _CCCL_API auto query(get_completion_scheduler_t<set_value_t>) const noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept
  {
    return task_scheduler_domain{};
  }
};

namespace __detail
{
//! @brief A type-erased sender returned by task_scheduler::schedule().
struct __task_sender : __basic_any<__itask_sender<>>
{
  _CCCL_TEMPLATE(class _Sndr)
  _CCCL_REQUIRES((!__same_as<_Sndr, __task_sender>) _CCCL_AND sender<_Sndr>)
  _CCCL_API explicit __task_sender(_Sndr __sndr, task_scheduler __sch)
      : __basic_any<__itask_sender<>>{static_cast<_Sndr&&>(__sndr)}
      , __env_{{}, _CCCL_MOVE(__sch)}
  {}

  _CCCL_API explicit __task_sender(__task_sender&& __other, task_scheduler __sch) noexcept
      : __basic_any<__itask_sender<>>{static_cast<__basic_any<__itask_sender<>>&&>(__other)}
      , __env_{{}, _CCCL_MOVE(__sch)}
  {}

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> decltype(auto)
  {
    return (__env_); // load-bearing parentheses, do not remove!
  }

private:
  using __env_t = prop<get_completion_scheduler_t<set_value_t>, task_scheduler>;
  __env_t __env_;
};

template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
class __task_bulk_state;

//! @brief A receiver used to connect the predecessor of a bulk operation launched by a
//! task_scheduler. It's set_value member stores the predecessor's values in the bulk
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
      using __tupl_t = cuda::std::__decayed_tuple<_As...>;
      __state_->__values_.template __emplace<__tupl_t>(static_cast<_As&&>(__as)...);
      __state_->__sch_.__schedule_bulk(
        _BulkTag{}, __state_->__shape_, __bulk_item_receiver_ref{*__state_}, __state_->__opstate2_);
      __state_->__opstate2_.start();
    }
    _CCCL_CATCH_ALL
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), ::std::current_exception());
    }
  }

  template <class _Error>
  _CCCL_API void set_error(_Error&& __err) noexcept
  {
    execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__err));
  }

  _CCCL_API void set_stopped() noexcept
  {
    execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
  {
    return __fwd_env(execution::get_env(__state_->__rcvr_));
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
    constexpr bool __valid_args = !__same_as<decltype(__args), cuda::std::monostate&>;
    // runtime assert that we never take this path without valid args from the predecessor:
    _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

    if constexpr (__valid_args)
    {
      // If we are not parallelizing, we need to run all the iterations sequentially.
      const size_t __increments = _Parallelize ? 1 : __shape;
      // Precompose the function with the arguments so we don't have to do it every iteration.
      auto __precomposed_fn = cuda::std::__apply(
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

//! Returns a visitor (callable) used to invoke the bulk (chunked) function with the
//! predecessor's values, which are stored in a variant in the bulk operation state.
template <bool _Parallelize, class _Fn>
[[nodiscard]] _CCCL_API constexpr auto
__get_execute_bulk_fn(bulk_chunked_t, _Fn& __fn, size_t __shape, size_t __begin, size_t __end) noexcept
{
  return [=, &__fn](auto& __args) {
    constexpr bool __valid_args = !__same_as<decltype(__args), cuda::std::monostate&>;
    _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

    if constexpr (__valid_args)
    {
      cuda::std::__apply(
        [&](auto&... __as) -> void {
          // If we are not parallelizing, we need to pass the entire range to the functor.
          _Parallelize ? __fn(__begin, __end, __as...) : __fn(0, __shape, __as...);
        },
        __args);
    }
  };
}

//! Stores the state for a bulk operation launched by a task_scheduler. A type-erased
//! reference to this object is passed to the task_scheduler's __schedule_bulk method,
//! which is expected to call execute(begin, end) on it to run the bulk operation. After
//! the bulk operation is complete, set_value is called, which forwards the predecessor's
//! values to the downstream receiver.
template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
class __task_bulk_state
{
public:
  _CCCL_API explicit __task_bulk_state(size_t __shape, _Fn __fn, _Rcvr __rcvr, task_scheduler __sch)
      : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
      , __shape_(__shape)
      , __fn_(static_cast<_Fn&&>(__fn))
      , __sch_(_CCCL_MOVE(__sch))
  {}

  _CCCL_API void set_value() noexcept
  {
    // Send the stored values to the downstream receiver.
    __values_.__visit(
      [this](auto& __tupl) {
        constexpr bool __valid_args = !__same_as<decltype(__tupl), cuda::std::monostate&>;
        // runtime assert that we never take this path without valid args from the predecessor:
        _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

        if constexpr (__valid_args)
        {
          cuda::std::__apply(execution::set_value, _CCCL_MOVE(__tupl), static_cast<_Rcvr&&>(__rcvr_));
        }
      },
      __values_);
  }

  _CCCL_API void set_error(::std::exception_ptr&& __eptr) noexcept
  {
    execution::set_error(static_cast<_Rcvr&&>(__rcvr_), _CCCL_MOVE(__eptr));
  }

  _CCCL_API void set_stopped() noexcept
  {
    execution::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
  {
    return __fwd_env(execution::get_env(__rcvr_));
  }

  //! Actually runs the bulk operation over the specified range.
  _CCCL_API void execute(size_t __begin, size_t __end) noexcept
  {
    _CCCL_TRY
    {
      constexpr bool __parallelize = _Policy() == par || _Policy() == par_unseq;
      __values_.__visit(__get_execute_bulk_fn<__parallelize>(_BulkTag(), __fn_, __shape_, __begin, __end), __values_);
    }
    _CCCL_CATCH_ALL
    {
      execution::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
    }
  }

private:
  template <class, class, class, class, class>
  friend struct __task_bulk_receiver;

  _Rcvr __rcvr_;
  size_t __shape_;
  _Fn __fn_;
  task_scheduler __sch_;
  _Values __values_{};
  __basic_any<__iopstate<>> __opstate2_;
};

template <class _BulkTag, class _Policy, class _Sndr, class _Fn, class _Rcvr>
struct __task_bulk_opstate
{
  using operation_state_concept = operation_state_t;

  _CCCL_API explicit __task_bulk_opstate(_Sndr&& __sndr, size_t __shape, _Fn __fn, _Rcvr __rcvr, task_scheduler __sch)
      : __state_{__shape, static_cast<_Fn&&>(__fn), static_cast<_Rcvr&&>(__rcvr), _CCCL_MOVE(__sch)}
      , __opstate1_(execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state_}))
  {}

  _CCCL_API void start() noexcept
  {
    __opstate1_.start();
  }

private:
  using __results_t =
    value_types_of_t<_Sndr, __fwd_env_t<env_of_t<_Rcvr>>, cuda::std::__decayed_tuple, __nullable_variant>;
  using __rcvr_t     = __task_bulk_receiver<_BulkTag, _Policy, _Fn, _Rcvr, __results_t>;
  using __opstate1_t = connect_result_t<_Sndr, __rcvr_t>;

  __task_bulk_state<_BulkTag, _Policy, _Fn, _Rcvr, __results_t> __state_;
  __opstate1_t __opstate1_;
};

template <class _Sndr>
struct __task_bulk_sender
{
  _CCCL_API explicit __task_bulk_sender(_Sndr __sndr, task_scheduler __sch)
      : __sndr_(static_cast<_Sndr&&>(__sndr))
      , __sch_(static_cast<task_scheduler&&>(__sch))
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
      static_cast<_Rcvr&&>(__rcvr),
      _CCCL_MOVE(__sch_)};
  }

  _CCCL_TEMPLATE(class _Self, class _Env)
  _CCCL_REQUIRES(__same_as<_Self, __task_bulk_sender>) // accept only rvalues.
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    // This calls get_completion_signatures on the wrapped bulk_[un]chunked sender. We
    // call it directly instead of using execution::get_completion_signatures to avoid
    // another trip through transform_sender, which would lead to infinite recursion.
    _CUDAX_LET_COMPLETIONS(auto(__completions) = _Sndr::template get_completion_signatures<_Sndr, _Env>())
    {
      return transform_completion_signatures(
        __completions, __decay_transform<set_value_t>(), {}, {}, __eptr_completion());
    }
    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept
  {
    return prop{get_completion_scheduler<set_value_t>, __sch_};
  }

private:
  _Sndr __sndr_;
  task_scheduler __sch_;
};

//! Helper class that maps from a chunk index to the start and end of the chunk.
struct __chunker
{
  _CCCL_API size_t __begin(size_t __chunk_index) const noexcept
  {
    return __chunk_index * __chunk_size_;
  }

  _CCCL_API size_t __end(size_t __chunk_index) const noexcept
  {
    auto __b = __begin(__chunk_index + 1);
    return __b < __max_size_ ? __b : __max_size_;
  }

  size_t __chunk_size_;
  size_t __max_size_;
};

//! Function called by the `bulk_chunked` operation; calls `execute` on the bulk_item_receiver_proxy.
struct __bulk_chunked_fn
{
  _CCCL_API void operator()(size_t __idx) noexcept
  {
    __rcvr_.execute(__chunker_.__begin(__idx), __chunker_.__end(__idx));
  }

  __bulk_item_receiver_ref __rcvr_;
  __chunker __chunker_;
};

//! Function called by the `bulk_unchunked` operation; calls `execute` on the bulk_item_receiver_proxy.
struct __bulk_unchunked_fn
{
  _CCCL_API void operator()(size_t __idx) noexcept
  {
    __rcvr_.execute(__idx, __idx + 1);
  }

  __bulk_item_receiver_ref __rcvr_;
};

//! A dummy operation state that calls set_error on the receiver with a stored exception_ptr when started.
struct __error_opstate
{
  using operation_state_concept = operation_state_t;

  _CCCL_API void start() noexcept
  {
    __rcvr_.set_error(_CCCL_MOVE(__eptr_));
  }

  __bulk_item_receiver_ref __rcvr_;
  ::std::exception_ptr __eptr_;
};

// The "virtual function" for task_scheduler::schedule()
template <class _Sch>
[[nodiscard]] _CCCL_PUBLIC_API auto __schedule_vfn(_Sch& __sch, task_scheduler __self) noexcept -> __task_sender
{
  return __task_sender(__sch.schedule(), _CCCL_MOVE(__self));
}

// The "virtual function" for task_scheduler::__schedule_bulk(bulk_chunked_t, ...)
template <class _Sch>
_CCCL_PUBLIC_API void __schedule_bulk_chunked_vfn(
  _Sch& __sch, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept
{
  _CCCL_TRY
  {
    // Determine the chunking size based on the ratio between the given size and the number of workers in our pool.
    // Aim at having 2 chunks per worker.
    size_t __available_parallelism_ = get_available_parallelism(__sch);
    size_t __chunk_size =
      (__available_parallelism_ > 0 && __size > 3 * __available_parallelism_)
        ? __size / __available_parallelism_ / 2
        : 1;
    size_t __num_chunks = (__size + __chunk_size - 1) / __chunk_size;

    auto __sndr = execution::bulk_unchunked(
      execution::schedule(__sch), par, __num_chunks, __bulk_chunked_fn{__rcvr, __chunker{__chunk_size, __size}});
    using __sndr_t    = decltype(__sndr);
    using __opstate_t = connect_result_t<__sndr_t, __bulk_item_receiver_ref>;
    __op.emplace_from<__opstate_t>(execution::connect, static_cast<__sndr_t&&>(__sndr), _CCCL_MOVE(__rcvr));
  }
  _CCCL_CATCH_ALL
  {
    __op.emplace<__error_opstate>(__rcvr, ::std::current_exception());
  }
}

// The "virtual function" for task_scheduler::__schedule_bulk(bulk_unchunked_t, ...)
template <class _Sch>
_CCCL_PUBLIC_API void __schedule_bulk_unchunked_vfn(
  _Sch& __sch, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept
{
  _CCCL_TRY
  {
    auto __sndr       = execution::bulk_unchunked(execution::schedule(__sch), par, __size, __bulk_unchunked_fn{__rcvr});
    using __sndr_t    = decltype(__sndr);
    using __opstate_t = connect_result_t<__sndr_t, __bulk_item_receiver_ref>;
    __op.emplace_from<__opstate_t>(execution::connect, static_cast<__sndr_t&&>(__sndr), _CCCL_MOVE(__rcvr));
  }
  _CCCL_CATCH_ALL
  {
    __op.emplace<__error_opstate>(__rcvr, ::std::current_exception());
  }
}

// Defined out-of-line so that the definitions of __task_scheduler and __task_sender are
// complete.
template <class... _Ts>
_CCCL_API auto __itask_scheduler<_Ts...>::schedule() noexcept -> __task_sender
{
  const auto& __base = ::cuda::__basic_any_from(*this);
  const auto& __self = static_cast<const task_scheduler&>(__base);
  return ::cuda::__virtcall<&__schedule_vfn<__itask_scheduler>>(this, __self);
}

} // namespace __detail

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TASK_SCHEDULER
