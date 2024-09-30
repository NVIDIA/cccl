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

#include <cuda/experimental/__async/config.cuh>

// run_loop isn't supported on-device yet, so neither can sync_wait be.
#if !defined(__CUDA_ARCH__)

#  include <cuda/std/optional>
#  include <cuda/std/tuple>

#  include <cuda/experimental/__async/exception.cuh>
#  include <cuda/experimental/__async/meta.cuh>
#  include <cuda/experimental/__async/run_loop.cuh>
#  include <cuda/experimental/__async/utility.cuh>

#  include <system_error>

#  include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
/// @brief Function object type for synchronously waiting for the result of a
/// sender.
struct sync_wait_t
{
#  if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#  endif // _CCCL_CUDA_COMPILER_NVCC
  struct __env_t
  {
    run_loop* __loop_;

    _CCCL_HOST_DEVICE auto query(get_scheduler_t) const noexcept
    {
      return __loop_->get_scheduler();
    }

    _CCCL_HOST_DEVICE auto query(get_delegatee_scheduler_t) const noexcept
    {
      return __loop_->get_scheduler();
    }
  };

  template <class _Sndr>
  struct __state_t
  {
    struct __rcvr_t
    {
      using receiver_concept = receiver_t;
      __state_t* __state_;

      template <class... _As>
      _CCCL_HOST_DEVICE void set_value(_As&&... __as) noexcept
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
      _CCCL_HOST_DEVICE void set_error(_Error __err) noexcept
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

      _CCCL_HOST_DEVICE void set_stopped() noexcept
      {
        __state_->__loop_.finish();
      }

      __env_t get_env() const noexcept
      {
        return __env_t{&__state_->__loop_};
      }
    };

    using __values_t = value_types_of_t<_Sndr, __rcvr_t, _CUDA_VSTD::tuple, __midentity::__f>;

    _CUDA_VSTD::optional<__values_t>* __values_;
    ::std::exception_ptr __eptr_;
    run_loop __loop_;
  };

  struct __invalid_sync_wait
  {
    const __invalid_sync_wait& value() const
    {
      return *this;
    }

    const __invalid_sync_wait& operator*() const
    {
      return *this;
    }

    int __i_;
  };

public:
  // clang-format off
    /// @brief Synchronously wait for the result of a sender, blocking the
    ///         current thread.
    ///
    /// `sync_wait` connects and starts the given sender, and then drives a
    ///         `run_loop` instance until the sender completes. Additional work
    ///         can be delegated to the `run_loop` by scheduling work on the
    ///         scheduler returned by calling `get_delegatee_scheduler` on the
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
    using __completions = completion_signatures_of_t<_Sndr, __rcvr_t>;
    static_assert(__is_completion_signatures<__completions>);

    if constexpr (!__is_completion_signatures<__completions>)
    {
      return __invalid_sync_wait{0};
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
};

_CCCL_GLOBAL_CONSTANT sync_wait_t sync_wait{};
} // namespace cuda::experimental::__async

#  include <cuda/experimental/__async/epilogue.cuh>

#endif // !defined(__CUDA_ARCH__)

#endif
