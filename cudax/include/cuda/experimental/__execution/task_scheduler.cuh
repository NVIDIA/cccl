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
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/bulk.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/parallel_scheduler_backend.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/variant.cuh>

#if !_CCCL_COMPILER(NVRTC)
#  include <memory> // IWYU pragma: keep for std::shared_ptr
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct task_scheduler;

struct task_scheduler_domain;

namespace __detail
{
// The concrete type-erased sender returned by task_scheduler::schedule()
struct __sndr_t;

template <class _Sndr>
struct __task_bulk_sender;

struct __parallel_scheduler_backend : parallel_scheduler_backend
{
  _CCCL_API virtual bool equal_to(const void* __other, ::cuda::std::__type_info_ref __type) = 0;
};

template <class _Sch>
_CCCL_CONCEPT __non_task_scheduler = _CCCL_REQUIRES_EXPR((_Sch))( //
  requires(__not_same_as<task_scheduler, _Sch>), //
  requires(scheduler<_Sch>));
} // namespace __detail

struct task_scheduler_domain : default_domain
{
  _CCCL_TEMPLATE(class _Sndr, class _Env, class _BulkTag = tag_of_t<_Sndr>)
  _CCCL_REQUIRES(__one_of<_BulkTag, bulk_chunked_t, bulk_unchunked_t> _CCCL_AND
                   __same_as<__call_result_t<get_completion_scheduler_t<set_value_t>, env_of_t<_Sndr>>, task_scheduler>)
  [[nodiscard]] _CCCL_HOST_API static constexpr auto transform_sender(set_value_t, _Sndr&& __sndr, const _Env& __env)
    -> __detail::__task_bulk_sender<_Sndr>
  {
    auto&& __sch = get_completion_scheduler<set_value_t>(get_env(__sndr));
    return __sch.__bulk_transform(static_cast<_Sndr&&>(__sndr), __env);
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
  template <class _Sch, class _Allocator>
  class _CCCL_TYPE_VISIBILITY_DEFAULT __backend_for;

public:
  using scheduler_concept = scheduler_t;

  _CCCL_TEMPLATE(class _Sch, class _Allocator = ::cuda::std::allocator<::cuda::std::byte>)
  _CCCL_REQUIRES(__detail::__non_task_scheduler<_Sch>)
  _CCCL_HOST_API explicit task_scheduler(_Sch sch, _Allocator alloc = {})
      : __backend_(::std::allocate_shared<__backend_for<_Sch, _Allocator>>(alloc, _CCCL_MOVE(sch), alloc))
  {}

  [[nodiscard]] _CCCL_HOST_API auto schedule() const noexcept -> __detail::__sndr_t;

  template <class Sndr, class Env>
  [[nodiscard]] _CCCL_HOST_API auto __bulk_transform(Sndr&& sndr, const Env& env)
  {
    // TODO(ericniebler): implement
  }

  [[nodiscard]] _CCCL_HOST_API friend bool operator==(const task_scheduler& lhs, const task_scheduler& rhs) noexcept
  {
    return lhs.__backend_ == rhs.__backend_;
  }

  [[nodiscard]] _CCCL_HOST_API friend bool operator!=(const task_scheduler& lhs, const task_scheduler& rhs) noexcept
  {
    return !(lhs == rhs);
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_HOST_API friend auto operator==(const task_scheduler& lhs, const _Sch& rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return lhs.__backend_->equal_to(::cuda::std::addressof(rhs), _CCCL_TYPEID(_Sch));
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_HOST_API friend auto operator!=(const task_scheduler& lhs, const _Sch& rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return !(lhs == rhs);
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_HOST_API friend auto operator==(const _Sch& lhs, const task_scheduler& rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return rhs == lhs;
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_HOST_API friend auto operator!=(const _Sch& lhs, const task_scheduler& rhs) noexcept
    _CCCL_TRAILING_REQUIRES(bool)(__detail::__non_task_scheduler<_Sch>)
  {
    return !(rhs == lhs);
  }

  [[nodiscard]] _CCCL_HOST_API auto query(get_completion_scheduler_t<set_value_t>) const noexcept
    -> const task_scheduler&
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept
  {
    return task_scheduler_domain{};
  }

private:
  friend struct __detail::__sndr_t;
  ::std::shared_ptr<__detail::__parallel_scheduler_backend> __backend_;
};

namespace __detail
{
//! @brief A type-erased opstate returned when connecting the result of
//! task_scheduler::schedule() to a receiver.
template <class _Rcvr>
class __opstate_t
{
public:
  using operation_state_concept = operation_state_t;

  _CCCL_HOST_API __opstate_t(::std::shared_ptr<__parallel_scheduler_backend> __backend, _Rcvr __rcvr)
      : __rcvr_(_CCCL_MOVE(__rcvr))
      , __backend_(_CCCL_MOVE(__backend))
  {}

  _CCCL_HOST_API void start() noexcept
  {
    _CCCL_TRY
    {
      __backend_->schedule(__rcvr_, ::cuda::std::span{__storage_});
    }
    _CCCL_CATCH_ALL
    {
      __rcvr_.set_error(execution::current_exception());
    }
  }

private:
  __detail::__proxy_receiver_impl<_Rcvr> __rcvr_;
  ::std::shared_ptr<__parallel_scheduler_backend> __backend_;
  ::cuda::std::byte __storage_[8 * sizeof(void*)];
};

//! @brief A type-erased sender returned by task_scheduler::schedule().
class __sndr_t
{
public:
  using sender_concept = sender_t;
  using __completions_t =
    completion_signatures<set_value_t(), //
                          set_error_t(exception_ptr),
                          set_error_t(cudaError_t),
                          set_stopped_t()>;

  _CCCL_HOST_API explicit __sndr_t(task_scheduler __sch)
      : __env_{{}, _CCCL_MOVE(__sch)}
  {}

  template <class _Rcvr>
  _CCCL_HOST_API auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr>
  {
    return __opstate_t<_Rcvr>(get_completion_scheduler<set_value_t>(__env_).__backend_, _CCCL_MOVE(__rcvr));
  }

  template <class _Self>
  _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept -> __completions_t
  {
    return {};
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> decltype(auto)
  {
    return (__env_); // parens are needed for decltype(auto) to return a reference
  }

private:
  using __env_t = prop<get_completion_scheduler_t<set_value_t>, task_scheduler>;
  __env_t __env_;
};
} // namespace __detail

template <class _Sch, class _Allocator>
class _CCCL_DECLSPEC_EMPTY_BASES task_scheduler::__backend_for
    : public __detail::__parallel_scheduler_backend
    , _Allocator
{
  template <class _RcvrProxy>
  struct __rcvr_t;
  template <class _RcvrProxy>
  friend struct __rcvr_t;
  class __opstate_with_allocator_t;
  using __opstate_t           = connect_result_t<schedule_result_t<_Sch>, __rcvr_t<receiver_proxy>>;
  using __opstate_allocator_t = typename ::cuda::std::allocator_traits<_Allocator>::template rebind_alloc<__opstate_t>;
  using __opstate_with_allocator_allocator_t =
    typename ::cuda::std::allocator_traits<_Allocator>::template rebind_alloc<__opstate_with_allocator_t>;

  _CCCL_API static void __delete_small(void* __ptr) noexcept
  {
    ::cuda::std::__destroy_at(static_cast<__opstate_t*>(__ptr));
  }

  _CCCL_API static void __delete_large(void* __ptr) noexcept
  {
    auto* __opstate_with_allocator = static_cast<__opstate_with_allocator_t*>(__ptr);

    __opstate_with_allocator_allocator_t __alloc(get_allocator(*__opstate_with_allocator));
    ::cuda::std::allocator_traits<__opstate_with_allocator_allocator_t>::destroy(__alloc, __opstate_with_allocator);
    ::cuda::std::allocator_traits<__opstate_with_allocator_allocator_t>::deallocate(
      __alloc, __opstate_with_allocator, 1);
  }

public:
  _CCCL_API explicit __backend_for(_Sch sch, _Allocator alloc)
      : _Allocator(_CCCL_MOVE(alloc))
      , __sch_(_CCCL_MOVE(sch))
  {}

  _CCCL_API void schedule(receiver_proxy& __rcvr_proxy,
                          ::cuda::std::span<::cuda::std::byte> __storage) noexcept final override
  {
    _CCCL_TRY
    {
      const bool __fits_in_storage = __storage.size() >= sizeof(__opstate_t);

      if (__fits_in_storage)
      {
        __opstate_allocator_t __alloc_copy{static_cast<_Allocator&>(*this)};
        auto __mk_opstate = [&] {
          return execution::connect(execution::schedule(__sch_),
                                    __rcvr_t<receiver_proxy>{__rcvr_proxy, __storage.data(), &__delete_small});
        };

        auto* __opstate_ptr = reinterpret_cast<__opstate_t*>(__storage.data());
        ::cuda::std::allocator_traits<__opstate_allocator_t>::construct(
          __alloc_copy, __opstate_ptr, __emplace_from{__mk_opstate});
        execution::start(*__opstate_ptr);
      }
      else
      {
        __opstate_with_allocator_allocator_t __alloc_copy{static_cast<_Allocator&>(*this)};

        auto* __opstate_ptr =
          ::cuda::std::allocator_traits<__opstate_with_allocator_allocator_t>::allocate(__alloc_copy, 1);
        ::cuda::std::allocator_traits<__opstate_with_allocator_allocator_t>::construct(
          __alloc_copy, __opstate_ptr, __alloc_copy, __sch_, __rcvr_proxy);
        execution::start(*__opstate_ptr);
      }
    }
    _CCCL_CATCH_ALL
    {
      __rcvr_proxy.set_error(execution::current_exception());
    }
  }

  _CCCL_API void schedule_bulk_chunked(
    size_t shape, bulk_item_receiver_proxy& r, ::cuda::std::span<::cuda::std::byte> s) noexcept final override
  {
    // TODO(ericniebler): implement
  }

  _CCCL_API void schedule_bulk_unchunked(
    size_t shape, bulk_item_receiver_proxy& r, ::cuda::std::span<::cuda::std::byte> s) noexcept override
  {
    // TODO(ericniebler): implement
  }

  _CCCL_API bool equal_to(const void* __other, ::cuda::std::__type_info_ref __type) final override
  {
    if (__type == _CCCL_TYPEID(_Sch))
    {
      const _Sch& __other_sch = *static_cast<const _Sch*>(__other);
      return __sch_ == __other_sch;
    }
    return false;
  }

private:
  class __opstate_with_allocator_t : __opstate_with_allocator_allocator_t
  {
  public:
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API
    __opstate_with_allocator_t(__opstate_with_allocator_allocator_t __alloc, _Sch& __sch, receiver_proxy& __rcvr_proxy)
        : __opstate_with_allocator_allocator_t(_CCCL_MOVE(__alloc))
        , __opstate_(execution::connect(
            execution::schedule(__sch), __rcvr_t<receiver_proxy>{__rcvr_proxy, this, &__delete_large}))
    {}
    __opstate_with_allocator_t(__opstate_with_allocator_t&&) = delete;

    _CCCL_API void start() noexcept
    {
      execution::start(__opstate_);
    }

    _CCCL_API auto query(get_allocator_t) const noexcept -> const __opstate_with_allocator_allocator_t&
    {
      return *this;
    }

  private:
    __opstate_t __opstate_;
  };

  template <class _RcvrProxy>
  struct __rcvr_t
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

    // TODO(ericniebler):
    // [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __fwd_env_t<env_of_t<_RcvrProxy>>
    // {
    //   return __fwd_env(execution::get_env(r));
    // }

    _RcvrProxy& __rcvr_proxy_;
    void* __opstate_storage_;
    __delete_fn_t* __delete_fn_;
  };

  _Sch __sch_;
};

namespace __detail
{
// template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
// class __task_bulk_state;

// //! @brief A receiver used to connect the predecessor of a bulk operation launched by a
// //! task_scheduler. It's set_value member stores the predecessor's values in the bulk
// //! operation state and then starts the bulk operation.
// template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
// struct __task_bulk_receiver
// {
//   using receiver_concept = receiver_t;

//   template <class... _As>
//   _CCCL_API void set_value(_As&&... __as) noexcept
//   {
//     _CCCL_TRY
//     {
//       using __tupl_t = cuda::std::__decayed_tuple<_As...>;
//       __state_->__values_.template __emplace<__tupl_t>(static_cast<_As&&>(__as)...);
//       __state_->__sch_.__schedule_bulk(
//         _BulkTag{}, __state_->__shape_, __bulk_item_receiver_ref{*__state_}, __state_->__opstate2_);
//       __state_->__opstate2_.start();
//     }
//     _CCCL_CATCH_ALL
//     {
//       execution::set_error(_CCCL_MOVE(__state_->__rcvr_), ::std::current_exception());
//     }
//   }

//   template <class _Error>
//   _CCCL_API void set_error(_Error&& __err) noexcept
//   {
//     execution::set_error(_CCCL_MOVE(__state_->__rcvr_), static_cast<_Error&&>(__err));
//   }

//   _CCCL_API void set_stopped() noexcept
//   {
//     execution::set_stopped(_CCCL_MOVE(__state_->__rcvr_));
//   }

//   [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
//   {
//     return __fwd_env(execution::get_env(__state_->__rcvr_));
//   }

//   __task_bulk_state<_BulkTag, _Policy, _Fn, _Rcvr, _Values>* __state_;
// };

// //! Returns a visitor (callable) used to invoke the bulk (unchunked) function with the
// //! predecessor's values, which are stored in a variant in the bulk operation state.
// template <bool _Parallelize, class _Fn>
// [[nodiscard]] _CCCL_API constexpr auto
// __get_execute_bulk_fn(bulk_unchunked_t, _Fn& __fn, size_t __shape, size_t __begin, size_t) noexcept
// {
//   return [=, &__fn](auto& __args) {
//     constexpr bool __valid_args = !__same_as<decltype(__args), cuda::std::monostate&>;
//     // runtime assert that we never take this path without valid args from the predecessor:
//     _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

//     if constexpr (__valid_args)
//     {
//       // If we are not parallelizing, we need to run all the iterations sequentially.
//       const size_t __increments = _Parallelize ? 1 : __shape;
//       // Precompose the function with the arguments so we don't have to do it every iteration.
//       auto __precomposed_fn = cuda::std::__apply(
//         [&](auto&... __as) {
//           return [&](size_t __i) -> void {
//             __fn(__i, __as...);
//           };
//         },
//         __args);
//       for (size_t __i = __begin; __i < __begin + __increments; ++__i)
//       {
//         __precomposed_fn(__i);
//       }
//     }
//   };
// }

// //! Returns a visitor (callable) used to invoke the bulk (chunked) function with the
// //! predecessor's values, which are stored in a variant in the bulk operation state.
// template <bool _Parallelize, class _Fn>
// [[nodiscard]] _CCCL_API constexpr auto
// __get_execute_bulk_fn(bulk_chunked_t, _Fn& __fn, size_t __shape, size_t __begin, size_t __end) noexcept
// {
//   return [=, &__fn](auto& __args) {
//     constexpr bool __valid_args = !__same_as<decltype(__args), cuda::std::monostate&>;
//     _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

//     if constexpr (__valid_args)
//     {
//       cuda::std::__apply(
//         [&](auto&... __as) -> void {
//           // If we are not parallelizing, we need to pass the entire range to the functor.
//           _Parallelize ? __fn(__begin, __end, __as...) : __fn(0, __shape, __as...);
//         },
//         __args);
//     }
//   };
// }

// //! Stores the state for a bulk operation launched by a task_scheduler. A type-erased
// //! reference to this object is passed to the task_scheduler's __schedule_bulk method,
// //! which is expected to call execute(begin, end) on it to run the bulk operation. After
// //! the bulk operation is complete, set_value is called, which forwards the predecessor's
// //! values to the downstream receiver.
// template <class _BulkTag, class _Policy, class _Fn, class _Rcvr, class _Values>
// class __task_bulk_state
// {
// public:
//   _CCCL_API explicit __task_bulk_state(size_t __shape, _Fn __fn, _Rcvr __rcvr, task_scheduler
//   __sch)
//       : __rcvr_(_CCCL_MOVE(__rcvr))
//       , __shape_(__shape)
//       , __fn_(_CCCL_MOVE(__fn))
//       , __sch_(_CCCL_MOVE(__sch))
//   {}

//   _CCCL_API void set_value() noexcept
//   {
//     // Send the stored values to the downstream receiver.
//     __values_.__visit(
//       [this](auto& __tupl) {
//         constexpr bool __valid_args = !__same_as<decltype(__tupl), cuda::std::monostate&>;
//         // runtime assert that we never take this path without valid args from the predecessor:
//         _CCCL_ASSERT(__valid_args, "internal error: predecessor results are not stored in the bulk operation state");

//         if constexpr (__valid_args)
//         {
//           cuda::std::__apply(execution::set_value, _CCCL_MOVE(__tupl), _CCCL_MOVE(__rcvr_));
//         }
//       },
//       __values_);
//   }

//   _CCCL_API void set_error(exception_ptr&& __eptr) noexcept
//   {
//     execution::set_error(_CCCL_MOVE(__rcvr_), _CCCL_MOVE(__eptr));
//   }

//   _CCCL_API void set_stopped() noexcept
//   {
//     execution::set_stopped(_CCCL_MOVE(__rcvr_));
//   }

//   [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
//   {
//     return __fwd_env(execution::get_env(__rcvr_));
//   }

//   //! Actually runs the bulk operation over the specified range.
//   _CCCL_API void execute(size_t __begin, size_t __end) noexcept
//   {
//     _CCCL_TRY
//     {
//       constexpr bool __parallelize = _Policy() == par || _Policy() == par_unseq;
//       __values_.__visit(__get_execute_bulk_fn<__parallelize>(_BulkTag(), __fn_, __shape_, __begin, __end),
//       __values_);
//     }
//     _CCCL_CATCH_ALL
//     {
//       execution::set_error(_CCCL_MOVE(__rcvr_), ::std::current_exception());
//     }
//   }

// private:
//   template <class, class, class, class, class>
//   friend struct __task_bulk_receiver;

//   _Rcvr __rcvr_;
//   size_t __shape_;
//   _Fn __fn_;
//   task_scheduler __sch_;
//   _Values __values_{};
//   __basic_any<__iopstate<>> __opstate2_;
// };

// template <class _BulkTag, class _Policy, class _Sndr, class _Fn, class _Rcvr>
// struct __task_bulk_opstate
// {
//   using operation_state_concept = operation_state_t;

//   _CCCL_API explicit __task_bulk_opstate(_Sndr&& __sndr, size_t __shape, _Fn __fn, _Rcvr __rcvr,
//   task_scheduler
//   __sch)
//       : __state_{__shape, _CCCL_MOVE(__fn), _CCCL_MOVE(__rcvr), _CCCL_MOVE(__sch)}
//       , __opstate1_(execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state_}))
//   {}

//   _CCCL_API void start() noexcept
//   {
//     __opstate1_.start();
//   }

// private:
//   using __results_t =
//     value_types_of_t<_Sndr, __fwd_env_t<env_of_t<_Rcvr>>, cuda::std::__decayed_tuple, __nullable_variant>;
//   using __rcvr_t     = __task_bulk_receiver<_BulkTag, _Policy, _Fn, _Rcvr, __results_t>;
//   using __opstate1_t = connect_result_t<_Sndr, __rcvr_t>;

//   __task_bulk_state<_BulkTag, _Policy, _Fn, _Rcvr, __results_t> __state_;
//   __opstate1_t __opstate1_;
// };

// template <class _Sndr>
// struct __task_bulk_sender
// {
//   _CCCL_API explicit __task_bulk_sender(_Sndr __sndr, task_scheduler __sch)
//       : __sndr_(_CCCL_MOVE(__sndr))
//       , __sch_(_CCCL_MOVE(__sch))
//   {}

//   template <class _Rcvr>
//   _CCCL_API auto connect(_Rcvr __rcvr) &&
//   {
//     auto& [__tag, __data, __child] = __sndr_;
//     auto& [__pol, __shape, __fn]   = __data;
//     return __task_bulk_opstate<decltype(__tag), decltype(__pol), decltype(__child), decltype(__fn), _Rcvr>{
//       _CCCL_MOVE(__child), static_cast<size_t>(__shape), _CCCL_MOVE(__fn), _CCCL_MOVE(__rcvr), _CCCL_MOVE(__sch_)};
//   }

//   _CCCL_TEMPLATE(class _Self, class _Env)
//   _CCCL_REQUIRES(__same_as<_Self, __task_bulk_sender>) // accept only rvalues.
//   [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
//   {
//     // This calls get_completion_signatures on the wrapped bulk_[un]chunked sender. We
//     // call it directly instead of using execution::get_completion_signatures to avoid
//     // another trip through transform_sender, which would lead to infinite recursion.
//     _CUDAX_LET_COMPLETIONS(auto(__completions) = _Sndr::template get_completion_signatures<_Sndr, _Env>())
//     {
//       return transform_completion_signatures(
//         __completions, __decay_transform<set_value_t>(), {}, {}, __eptr_completion());
//     }
//     _CCCL_UNREACHABLE();
//   }

//   [[nodiscard]] _CCCL_API auto get_env() const noexcept
//   {
//     return prop{get_completion_scheduler<set_value_t>, __sch_};
//   }

// private:
//   _Sndr __sndr_;
//   task_scheduler __sch_;
// };

// //! Helper class that maps from a chunk index to the start and end of the chunk.
// struct __chunker
// {
//   _CCCL_API size_t __begin(size_t __chunk_index) const noexcept
//   {
//     return __chunk_index * __chunk_size_;
//   }

//   _CCCL_API size_t __end(size_t __chunk_index) const noexcept
//   {
//     auto __b = __begin(__chunk_index + 1);
//     return __b < __max_size_ ? __b : __max_size_;
//   }

//   size_t __chunk_size_;
//   size_t __max_size_;
// };

// //! Function called by the `bulk_chunked` operation; calls `execute` on the bulk_item_receiver_proxy.
// struct __bulk_chunked_fn
// {
//   _CCCL_API void operator()(size_t __idx) noexcept
//   {
//     __rcvr_.execute(__chunker_.__begin(__idx), __chunker_.__end(__idx));
//   }

//   __bulk_item_receiver_ref __rcvr_;
//   __chunker __chunker_;
// };

// //! Function called by the `bulk_unchunked` operation; calls `execute` on the bulk_item_receiver_proxy.
// struct __bulk_unchunked_fn
// {
//   _CCCL_API void operator()(size_t __idx) noexcept
//   {
//     __rcvr_.execute(__idx, __idx + 1);
//   }

//   __bulk_item_receiver_ref __rcvr_;
// };

// //! A dummy operation state that calls set_error on the receiver with a stored exception_ptr when started.
// struct __error_opstate
// {
//   using operation_state_concept = operation_state_t;

//   _CCCL_API void start() noexcept
//   {
//     __rcvr_.set_error(_CCCL_MOVE(__eptr_));
//   }

//   __bulk_item_receiver_ref __rcvr_;
//   exception_ptr __eptr_;
// };

// // The "virtual function" for task_scheduler::schedule()
// template <class _Sch>
// [[nodiscard]] _CCCL_PUBLIC_API auto __schedule_vfn(_Sch& __sch, task_scheduler __self) noexcept -> __detail::__sndr_t
// {
//   return __detail::__sndr_t(__sch.schedule(), _CCCL_MOVE(__self));
// }

// // The "virtual function" for task_scheduler::__schedule_bulk(bulk_chunked_t, ...)
// template <class _Sch>
// _CCCL_PUBLIC_API void __schedule_bulk_chunked_vfn(
//   _Sch& __sch, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept
// {
//   _CCCL_TRY
//   {
//     // Determine the chunking size based on the ratio between the given size and the number of workers in our pool.
//     // Aim at having 2 chunks per worker.
//     size_t __available_parallelism_ = get_available_parallelism(__sch);
//     size_t __chunk_size =
//       (__available_parallelism_ > 0 && __size > 3 * __available_parallelism_)
//         ? __size / __available_parallelism_ / 2
//         : 1;
//     size_t __num_chunks = (__size + __chunk_size - 1) / __chunk_size;

//     auto __sndr = execution::bulk_unchunked(
//       execution::schedule(__sch), par, __num_chunks, __bulk_chunked_fn{__rcvr, __chunker{__chunk_size, __size}});
//     using __sndr_t    = decltype(__sndr);
//     using __opstate_t = connect_result_t<__sndr_t, __bulk_item_receiver_ref>;
//     __op.emplace_from<__opstate_t>(execution::connect, _CCCL_MOVE(__sndr), _CCCL_MOVE(__rcvr));
//   }
//   _CCCL_CATCH_ALL
//   {
//     __op.emplace<__error_opstate>(__rcvr, ::std::current_exception());
//   }
// }

// // The "virtual function" for task_scheduler::__schedule_bulk(bulk_unchunked_t, ...)
// template <class _Sch>
// _CCCL_PUBLIC_API void __schedule_bulk_unchunked_vfn(
//   _Sch& __sch, size_t __size, __bulk_item_receiver_ref __rcvr, __basic_any<__iopstate<>>& __op) noexcept
// {
//   _CCCL_TRY
//   {
//     auto __sndr       = execution::bulk_unchunked(execution::schedule(__sch), par, __size,
//     __bulk_unchunked_fn{__rcvr}); using __sndr_t    = decltype(__sndr); using __opstate_t =
//     connect_result_t<__sndr_t, __bulk_item_receiver_ref>;
//     __op.emplace_from<__opstate_t>(execution::connect, _CCCL_MOVE(__sndr), _CCCL_MOVE(__rcvr));
//   }
//   _CCCL_CATCH_ALL
//   {
//     __op.emplace<__error_opstate>(__rcvr, ::std::current_exception());
//   }
// }

// // Defined out-of-line so that the definitions of __task_scheduler and __sndr_t are
// // complete.
// template <class... _Ts>
// _CCCL_API auto __itask_scheduler<_Ts...>::schedule() noexcept -> __detail::__sndr_t
// {
//   const auto& __base = ::cuda::__basic_any_from(*this);
//   const auto& __self = static_cast<const task_scheduler&>(__base);
//   return ::cuda::__virtcall<&__schedule_vfn<__itask_scheduler>>(this, __self);
// }

// Implementation of the get_scheduler_t query for __proxy_receiver_impl from
// parallel_scheduler_backend.cuh
template <class _Rcvr>
_CCCL_HOST_API void __proxy_receiver_impl<_Rcvr>::operator()(
  const get_scheduler_t& __query, ::cuda::std::optional<task_scheduler>& out) const noexcept
{
  if constexpr (__callable<const get_scheduler_t&, env_of_t<_Rcvr>>)
  {
    out.emplace(__query(get_env(__rcvr_)));
  }
}
} // namespace __detail
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TASK_SCHEDULER
