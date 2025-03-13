//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__detail/config.cuh>

// run_loop isn't supported on-device yet, so neither can sync_wait be.
#if !defined(__CUDA_ARCH__)

#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/type_identity.h>
#  include <cuda/std/optional>
#  include <cuda/std/tuple>

#  include <cuda/experimental/__async/sender/exception.cuh>
#  include <cuda/experimental/__async/sender/meta.cuh>
#  include <cuda/experimental/__async/sender/run_loop.cuh>
#  include <cuda/experimental/__async/sender/utility.cuh>
#  include <cuda/experimental/__async/sender/write_env.cuh>

#  include <system_error>

#  include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
/// @brief Function object type for synchronously waiting for the result of a
/// sender.
struct sync_wait_t
{
#  if !_CCCL_CUDA_COMPILER(NVCC)

private:
#  endif // !_CCCL_CUDA_COMPILER(NVCC)

  struct __env_t
  {
    run_loop* __loop_;

    _CUDAX_API auto query(get_scheduler_t) const noexcept
    {
      return __loop_->get_scheduler();
    }

    _CUDAX_API auto query(get_delegation_scheduler_t) const noexcept
    {
      return __loop_->get_scheduler();
    }
  };

  template <class _Sndr>
  struct __state_t
  {
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
    {
      using receiver_concept = receiver_t;
      __state_t* __state_;

      template <class... _As>
      _CUDAX_API void set_value(_As&&... __as) noexcept
      {
        _CUDAX_TRY( //
          ({ //
            __state_->__values_->emplace(static_cast<_As&&>(__as)...);
          }), //
          _CUDAX_CATCH(...)( //
            { //
              __state_->__eptr_ = ::std::current_exception();
            }))
        __state_->__loop_.finish();
      }

      template <class _Error>
      _CUDAX_API void set_error(_Error __err) noexcept
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

      _CUDAX_API void set_stopped() noexcept
      {
        __state_->__loop_.finish();
      }

      __env_t get_env() const noexcept
      {
        return __env_t{&__state_->__loop_};
      }
    };

    using __completions_t = completion_signatures_of_t<_Sndr, __rcvr_t>;

    struct __on_success
    {
      using type = __value_types<__completions_t, _CUDA_VSTD::tuple, _CUDA_VSTD::__type_self_t>;
    };

    using __on_error = _CUDA_VSTD::type_identity<_CUDA_VSTD::tuple<__completions_t>>;

    using __values_t =
      typename _CUDA_VSTD::_If<__is_completion_signatures<__completions_t>, __on_success, __on_error>::type;

    _CUDA_VSTD::optional<__values_t>* __values_;
    ::std::exception_ptr __eptr_;
    run_loop __loop_;
  };

  template <class _Diagnostic>
  struct __bad_sync_wait
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Diagnostic>(),
                  "sync_wait cannot compute the completions of the sender passed to it.");
    static __bad_sync_wait __result();

    const __bad_sync_wait& value() const;
    const __bad_sync_wait& operator*() const;

    int i{}; // so that structured bindings kinda work
  };

public:
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
    /// @retval success Returns an engaged `::std::optional` containing the result
    ///         values in a `::std::tuple`.
    /// @retval canceled Returns an empty `::std::optional`.
    /// @retval error Throws the error.
    ///
    /// @throws ::std::rethrow_exception(error) if the error has type
    ///         `::std::exception_ptr`.
    /// @throws ::std::system_error(error) if the error has type
    ///         `::std::error_code`.
    /// @throws error otherwise
  // clang-format on
  template <class _Sndr>
  auto operator()(_Sndr&& __sndr) const
  {
    using __rcvr_t      = typename __state_t<_Sndr>::__rcvr_t;
    using __values_t    = typename __state_t<_Sndr>::__values_t;
    using __completions = typename __state_t<_Sndr>::__completions_t;

    if constexpr (!__is_completion_signatures<__completions>)
    {
      return __bad_sync_wait<__completions>::__result();
    }
    else
    {
      _CUDA_VSTD::optional<__values_t> __result{};
      __state_t<_Sndr> __state{&__result, {}, {}};

      // Launch the sender with a continuation that will fill in a variant
      auto __opstate = __async::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state});
      __async::start(__opstate);

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
  auto operator()(_Sndr&& __sndr, _Env&& __env) const
  {
    return (*this)(__async::write_env(static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env)));
  }
};

_CCCL_GLOBAL_CONSTANT sync_wait_t sync_wait{};
} // namespace cuda::experimental::__async

#  include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // !defined(__CUDA_ARCH__)

#endif
