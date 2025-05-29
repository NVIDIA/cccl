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

#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/optional>
#include <cuda/std/tuple>

#include <cuda/experimental/__execution/apply_sender.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/run_loop.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/write_env.cuh>

#include <system_error>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
/// @brief Function object type for synchronously waiting for the result of a
/// sender.
struct sync_wait_t
{
private:
  template <class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_state_t
  {
    // FUTURE: if _Env provides a delegation scheduler, we don't need the run_loop
    run_loop __loop_;
    _CCCL_NO_UNIQUE_ADDRESS _Env __env_;
  };

  template <class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    __env_state_t<_Env>* __state_;

    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__queryable_with<_Env, _Query>)
    [[nodiscard]] _CCCL_API auto query(_Query) const noexcept(__nothrow_queryable_with<_Env, _Query>)
      -> __query_result_t<_Env, _Query>
    {
      return __state_->__env_.query(_Query{});
    }

    [[nodiscard]] _CCCL_API auto query(get_scheduler_t) const noexcept
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

    [[nodiscard]] _CCCL_API auto query(get_delegation_scheduler_t) const noexcept
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
  };

  template <class _Values, class _Errors, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t : __env_state_t<_Env>
  {
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
    {
      using receiver_concept _CCCL_NODEBUG_ALIAS = receiver_t;

      template <class... _As>
      _CCCL_API void set_value(_As&&... __as) && noexcept
      {
        _CUDAX_TRY( //
          ({ //
            __state_->__values_->emplace(static_cast<_As&&>(__as)...);
          }), //
          _CUDAX_CATCH(...) //
          ({ //
            if constexpr (!__nothrow_constructible<_Values, _As...>)
            {
              __state_->__errors_.__emplace(::std::current_exception());
            }
          }) //
        )
        __state_->__loop_.finish();
      }

      template <class _Error>
      _CCCL_API void set_error(_Error __err) && noexcept
      {
        __state_->__errors_.__emplace(static_cast<_Error&&>(__err));
        __state_->__loop_.finish();
      }

      _CCCL_API void set_stopped() && noexcept
      {
        __state_->__loop_.finish();
      }

      [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __env_t<_Env>
      {
        return __env_t<_Env>{__state_};
      }

      __state_t* __state_;
    };

    _CUDA_VSTD::optional<_Values>* __values_;
    _Errors __errors_;
  };

  struct __throw_error_fn
  {
    template <class _Error>
    _CCCL_HOST_API void operator()(_Error&& __err) const
    {
      if constexpr (_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Error>, ::std::exception_ptr>)
      {
        ::std::rethrow_exception(static_cast<_Error&&>(__err));
      }
      else if constexpr (_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Error>, ::std::error_code>)
      {
        throw ::std::system_error(static_cast<_Error&&>(__err));
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
    static_assert(_CUDA_VSTD::__always_false_v<_Diagnostic>,
                  "sync_wait cannot compute the completions of the sender passed to it.");
    _CCCL_HOST_API static auto __result() -> __bad_sync_wait;

    _CCCL_HOST_API auto value() const -> const __bad_sync_wait&;
    _CCCL_HOST_API auto operator*() const -> const __bad_sync_wait&;

    int i{}; // so that structured bindings kinda work
  };

public:
  // This is the actual default sync_wait implementation.
  template <class _Sndr, class _Env>
  _CCCL_HOST_API static auto apply_sender(_Sndr&& __sndr, _Env&& __env)
  {
    using __completions _CCCL_NODEBUG_ALIAS = completion_signatures_of_t<_Sndr, __env_t<_Env>>;

    if constexpr (!__valid_completion_signatures<__completions>)
    {
      return __bad_sync_wait<__completions>::__result();
    }
    else if constexpr (__completions().count(set_value) != 1)
    {
      static_assert(__completions().count(set_value) == 1,
                    "sync_wait requires a sender with exactly one value completion signature.");
    }
    else
    {
      using __values_t _CCCL_NODEBUG_ALIAS = __value_types<__completions, _CUDA_VSTD::tuple, _CUDA_VSTD::__type_self_t>;
      using __errors_t _CCCL_NODEBUG_ALIAS = __error_types<__completions, __decayed_variant>;
      _CUDA_VSTD::optional<__values_t> __result{};
      __state_t<__values_t, __errors_t, _Env> __state{{{}, static_cast<_Env&&>(__env)}, &__result, {}};

      // Launch the sender with a continuation that will fill in a variant
      using __rcvr_t = typename __state_t<__values_t, __errors_t, _Env>::__rcvr_t;
      auto __opstate = execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state});

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
  }

  template <class _Sndr>
  _CCCL_HOST_API static auto apply_sender(_Sndr&& __sndr)
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
  template <class _Sndr, class _Env>
  _CCCL_HOST_API auto operator()(_Sndr&& __sndr, _Env&& __env) const
  {
    using __dom_t = __late_domain_of_t<_Sndr, __env_t<_Env>>;
    return execution::apply_sender(__dom_t{}, *this, static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env));
  }

  template <class _Sndr, class... _Env>
  _CCCL_HOST_API auto operator()(_Sndr&& __sndr) const
  {
    using __dom_t = __late_domain_of_t<_Sndr, __env_t<env<>>>;
    return execution::apply_sender(__dom_t{}, *this, static_cast<_Sndr&&>(__sndr));
  }
};

_CCCL_GLOBAL_CONSTANT sync_wait_t sync_wait{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SYNC_WAIT
