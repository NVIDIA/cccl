//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_SYNC_WAIT
#define __CUDAX_ASYNC_DETAIL_SYNC_WAIT

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
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    run_loop* __loop_;

    _CCCL_API auto query(get_scheduler_t) const noexcept
    {
      return __loop_->get_scheduler();
    }

    _CCCL_API auto query(get_delegation_scheduler_t) const noexcept
    {
      return __loop_->get_scheduler();
    }
  };

  template <class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
    {
      using receiver_concept _CCCL_NODEBUG_ALIAS = receiver_t;
      __state_t* __state_;

      template <class... _As>
      _CCCL_API void set_value(_As&&... __as) && noexcept
      {
        _CUDAX_TRY( //
          ({ //
            __state_->__values_->emplace(static_cast<_As&&>(__as)...);
          }), //
          _CUDAX_CATCH(...) //
          ({ //
            __state_->__eptr_ = ::std::current_exception();
          }) //
        )
        __state_->__loop_.finish();
      }

      template <class _Error>
      _CCCL_API void set_error(_Error __err) && noexcept
      {
        if constexpr (_CUDA_VSTD::is_same_v<_Error, ::std::exception_ptr>)
        {
          __state_->__eptr_ = static_cast<_Error&&>(__err);
        }
        else if constexpr (_CUDA_VSTD::is_same_v<_Error, ::std::error_code>)
        {
          __state_->__eptr_ = ::std::make_exception_ptr(::std::system_error(__err));
        }
        else
        {
          __state_->__eptr_ = ::std::make_exception_ptr(static_cast<_Error&&>(__err));
        }
        __state_->__loop_.finish();
      }

      _CCCL_API void set_stopped() && noexcept
      {
        __state_->__loop_.finish();
      }

      auto get_env() const noexcept -> __env_t
      {
        return __env_t{&__state_->__loop_};
      }
    };

    _CUDA_VSTD::optional<_Values>* __values_;
    ::std::exception_ptr __eptr_;
    run_loop __loop_;
  };

  template <class _Diagnostic>
  struct __bad_sync_wait
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Diagnostic>(),
                  "sync_wait cannot compute the completions of the sender passed to it.");
    static auto __result() -> __bad_sync_wait;

    auto value() const -> const __bad_sync_wait&;
    auto operator*() const -> const __bad_sync_wait&;

    int i{}; // so that structured bindings kinda work
  };

public:
  // This is the actual default sync_wait implementation.
  template <class _Sndr>
  _CCCL_HOST_API static auto apply_sender(_Sndr&& __sndr)
  {
    using __completions _CCCL_NODEBUG_ALIAS = completion_signatures_of_t<_Sndr, __env_t>;

    if constexpr (!__valid_completion_signatures<__completions>)
    {
      return __bad_sync_wait<__completions>::__result();
    }
    else
    {
      using __values _CCCL_NODEBUG_ALIAS = __value_types<__completions, _CUDA_VSTD::tuple, _CUDA_VSTD::__type_self_t>;
      _CUDA_VSTD::optional<__values> __result{};
      __state_t<__values> __state{&__result, {}, {}};

      // Launch the sender with a continuation that will fill in a variant
      using __rcvr   = typename __state_t<__values>::__rcvr_t;
      auto __opstate = execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr{&__state});
      execution::start(__opstate);

      // Wait for the variant to be filled in, and process any work that
      // may be delegated to this thread.
      __state.__loop_.run();

      if (__state.__eptr_)
      {
        ::std::rethrow_exception(__state.__eptr_);
      }

      return __result; // uses NRVO to "return" the result
    }
  }

  template <class _Sndr, class _Env>
  _CCCL_HOST_API static auto apply_sender(_Sndr&& __sndr, _Env&& __env)
  {
    return apply_sender(execution::write_env(static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env)));
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
  _CCCL_HOST_API auto operator()(_Sndr&& __sndr, _Env&&... __env) const
  {
    using __dom_t _CCCL_NODEBUG_ALIAS = domain_for_t<_Sndr, _Env...>;
    return execution::apply_sender(__dom_t{}, *this, static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env)...);
  }
};

_CCCL_GLOBAL_CONSTANT sync_wait_t sync_wait{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_SYNC_WAIT
