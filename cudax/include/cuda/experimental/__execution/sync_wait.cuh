//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_SYNC_WAIT
#define __CUDAX_EXECUTION_SYNC_WAIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/optional>
#include <cuda/std/tuple>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/apply_sender.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/run_loop.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/write_env.cuh>

#include <exception>
#include <system_error>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
/// @brief Function object type for synchronously waiting for the result of a
/// sender.
struct sync_wait_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_base_t
  {
    // FUTURE: if _Env provides a delegation scheduler, we don't need the run_loop
    run_loop __loop_;
    _Env __env_;
  };

  template <class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__queryable_with<_Env, _Query>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<_Env, _Query>)
      -> __query_result_t<_Env, _Query>
    {
      return __state_->__env_.query(_Query{});
    }

    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_API constexpr auto query(get_scheduler_t) const noexcept
    {
      if constexpr (__queryable_with<_Env, get_scheduler_t>)
      {
        return __state_->__env_.query(get_scheduler);
      }
      else
      {
        return __state_->__loop_.get_scheduler();
      }
      _CCCL_UNREACHABLE();
    }

    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_API constexpr auto query(get_delegation_scheduler_t) const noexcept
    {
      if constexpr (__queryable_with<_Env, get_delegation_scheduler_t>)
      {
        return __state_->__env_.query(get_delegation_scheduler);
      }
      else
      {
        return __state_->__loop_.get_scheduler();
      }
      _CCCL_UNREACHABLE();
    }

    __state_base_t<_Env>* __state_;
  };

  template <class... _Ts>
  using __decayed_tuple = ::cuda::std::tuple<decay_t<_Ts>...>;

  template <class _Values, class _Errors, class _Env = env<>>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t : __state_base_t<_Env>
  {
    ::cuda::std::optional<_Values>* __values_;
    _Errors __errors_;
  };

  template <class _Values, class _Errors, class _Env = env<>>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _As>
    _CCCL_API void set_value(_As&&... __as) noexcept
    {
      _CCCL_TRY
      {
        __state_->__values_->emplace(static_cast<_As&&>(__as)...);
      }
      _CCCL_CATCH_ALL
      {
        // avoid ODR-using a call to __emplace(exception_ptr) if this code is unreachable.
        if constexpr (!__nothrow_decay_copyable<_As...>)
        {
          __state_->__errors_.__emplace(::std::current_exception());
        }
      }
      __state_->__loop_.finish();
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __err) noexcept
    {
      _CCCL_TRY
      {
        __state_->__errors_.__emplace(static_cast<_Error&&>(__err));
      }
      _CCCL_CATCH_ALL
      {
        // avoid ODR-using a call to __emplace(exception_ptr) if this code is unreachable.
        if constexpr (!__nothrow_decay_copyable<_Error>)
        {
          __state_->__errors_.__emplace(::std::current_exception());
        }
      }
      __state_->__loop_.finish();
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      __state_->__loop_.finish();
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __env_t<_Env>
    {
      return __env_t<_Env>{__state_};
    }

    __state_t<_Values, _Errors, _Env>* __state_;
  };

  struct __throw_error_fn
  {
    template <class _Error>
    _CCCL_HOST_API void operator()(_Error __err) const
    {
      if constexpr (__same_as<_Error, ::std::exception_ptr>)
      {
        ::std::rethrow_exception(static_cast<_Error&&>(__err));
      }
      else if constexpr (__same_as<_Error, ::std::error_code>)
      {
        throw ::std::system_error(__err);
      }
      else if constexpr (__same_as<_Error, cudaError_t>)
      {
        ::cuda::__throw_cuda_error(__err, "sync_wait failed with cudaError_t");
      }
      else
      {
        throw static_cast<_Error&&>(__err);
      }
    }
  };

  template <class _Diagnostic>
  struct __bad_sync_wait
  {
    static_assert(::cuda::std::__always_false_v<_Diagnostic>,
                  "sync_wait cannot compute the completions of the sender passed to it.");
    _CCCL_HOST_API static auto __result() -> __bad_sync_wait;

    _CCCL_HOST_API auto value() const -> const __bad_sync_wait&;
    _CCCL_HOST_API auto operator*() const -> const __bad_sync_wait&;

    // Attempt to suppress follow-on errors about non-convertibility after the one already
    // reported.
    template <class _Ty>
    _CCCL_API operator _Ty&&() const noexcept;

    int i{}; // so that structured bindings kinda work
  };

public:
  // This is the actual default sync_wait implementation.
  template <class _Sndr, class _Env>
  _CCCL_API static auto apply_sender(_Sndr&& __sndr, _Env&& __env)
  {
    using __partial_completions_t = completion_signatures_of_t<_Sndr, __env_t<_Env>>;
    using __all_nothrow_t =
      typename __partial_completions_t::template __transform_q<__nothrow_decay_copyable_t, ::cuda::std::_And>;

    using __completions_t =
      __concat_completion_signatures_t<__partial_completions_t, __eptr_completion_if_t<!__all_nothrow_t::value>>;

    using __values_t = __value_types<__completions_t, __decayed_tuple, ::cuda::std::__type_self_t>;
    using __errors_t = __error_types<__completions_t, __decayed_variant>;

    ::cuda::std::optional<__values_t> __result{};
    __state_t<__values_t, __errors_t, _Env> __state{{{}, static_cast<_Env&&>(__env)}, &__result, {}};

    // Launch the sender with a continuation that will fill in a variant
    auto __opstate = execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t<__values_t, __errors_t, _Env>{&__state});
    execution::start(__opstate);

    // While waiting for the variant to be filled in, process any work that may be
    // delegated to this thread.
    __state.__loop_.run();

    if (__state.__errors_.__index() != __npos)
    {
      __errors_t::__visit(__throw_error_fn{}, static_cast<__errors_t&&>(__state.__errors_));
    }

    return __result; // uses NRVO to return the result
  }

  template <class _Sndr>
  _CCCL_API static auto apply_sender(_Sndr&& __sndr)
  {
    return apply_sender(static_cast<_Sndr&&>(__sndr), env{});
  }

  // clang-format off
  /// @brief Synchronously wait for the result of a sender, blocking the
  ///         current thread.
  ///
  /// `sync_wait` connects and starts the given sender, and then drives a
  ///         `run_loop` instance until the sender completes. Additional work
  ///         can be delegated to the `run_loop` by scheduling work on the
  ///         scheduler returned by calling `get_delegation_scheduler` on the
  ///         receiver's environment.
  ///
  /// @pre The sender must have a exactly one value completion signature. That
  ///         is, it can only complete successfully in one way, with a single
  ///         set of values.
  ///
  /// @retval success Returns an engaged `cuda::std::optional` containing the result
  ///         values in a `cuda::std::tuple`.
  /// @retval canceled Returns an empty `cuda::std::optional`.
  /// @retval error Throws the error.
  ///
  /// @throws ::std::rethrow_exception(error) if the error has type
  ///         `::std::exception_ptr`.
  /// @throws ::std::system_error(error) if the error has type
  ///         `::std::error_code`.
  /// @throws ::cuda::cuda_error(error, "...") if the error has type
  ///         `cudaError_t`.
  /// @throws error otherwise
  // clang-format on
  template <class _Sndr, class... _Env>
  _CCCL_API auto operator()(_Sndr&& __sndr, _Env&&... __env) const
  {
    using __env_t                = sync_wait_t::__env_t<::cuda::std::__type_index_c<0, _Env..., env<>>>;
    constexpr auto __completions = get_completion_signatures<_Sndr, __env_t>();
    using __completions_t        = decltype(__completions);

    if constexpr (!__valid_completion_signatures<__completions_t>)
    {
      return __bad_sync_wait<__completions_t>::__result();
    }
    else if constexpr (__completions.count(set_value) != 1)
    {
      static_assert(__completions.count(set_value) == 1,
                    "sync_wait requires a sender with exactly one value completion signature.");
    }
    else
    {
      using __dom_t _CCCL_NODEBUG_ALIAS = __late_domain_of_t<_Sndr, __env_t, __early_domain_of_t<_Sndr>>;
      return execution::apply_sender(__dom_t{}, *this, static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env)...);
    }
  }
};

_CCCL_GLOBAL_CONSTANT sync_wait_t sync_wait{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SYNC_WAIT
