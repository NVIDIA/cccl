//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_SYNC_WAIT_H
#define __CUDAX_ASYNC_DETAIL_SYNC_WAIT_H

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
#  ifndef __CUDACC__

private:
#  endif
  struct _env_t
  {
    run_loop* _loop;

    _CCCL_HOST_DEVICE auto query(get_scheduler_t) const noexcept
    {
      return _loop->get_scheduler();
    }

    _CCCL_HOST_DEVICE auto query(get_delegatee_scheduler_t) const noexcept
    {
      return _loop->get_scheduler();
    }
  };

  template <class Sndr>
  struct _state_t
  {
    struct _rcvr_t
    {
      using receiver_concept = receiver_t;
      _state_t* _state;

      template <class... As>
      _CCCL_HOST_DEVICE void set_value(As&&... _as) noexcept
      {
        _CUDAX_TRY( //
          ({ //
            _state->_values->emplace(static_cast<As&&>(_as)...);
          }), //
          _CUDAX_CATCH(...)( //
            { //
              _state->_eptr = ::std::current_exception();
            }))
        _state->_loop.finish();
      }

      template <class Error>
      _CCCL_HOST_DEVICE void set_error(Error _err) noexcept
      {
        if constexpr (_CUDA_VSTD::is_same_v<Error, ::std::exception_ptr>)
        {
          _state->_eptr = static_cast<Error&&>(_err);
        }
        else if constexpr (_CUDA_VSTD::is_same_v<Error, ::std::error_code>)
        {
          _state->_eptr = ::std::make_exception_ptr(::std::system_error(_err));
        }
        else
        {
          _state->_eptr = ::std::make_exception_ptr(static_cast<Error&&>(_err));
        }
        _state->_loop.finish();
      }

      _CCCL_HOST_DEVICE void set_stopped() noexcept
      {
        _state->_loop.finish();
      }

      _env_t get_env() const noexcept
      {
        return _env_t{&_state->_loop};
      }
    };

    using _values_t = value_types_of_t<Sndr, _rcvr_t, _CUDA_VSTD::tuple, _midentity::_f>;

    _CUDA_VSTD::optional<_values_t>* _values;
    ::std::exception_ptr _eptr;
    run_loop _loop;
  };

  struct _invalid_sync_wait
  {
    const _invalid_sync_wait& value() const
    {
      return *this;
    }

    const _invalid_sync_wait& operator*() const
    {
      return *this;
    }

    int i;
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
  template <class Sndr>
  auto operator()(Sndr&& sndr) const
  {
    using _rcvr_t      = typename _state_t<Sndr>::_rcvr_t;
    using _values_t    = typename _state_t<Sndr>::_values_t;
    using _completions = completion_signatures_of_t<Sndr, _rcvr_t>;
    static_assert(_is_completion_signatures<_completions>);

    if constexpr (!_is_completion_signatures<_completions>)
    {
      return _invalid_sync_wait{0};
    }
    else
    {
      _CUDA_VSTD::optional<_values_t> result{};
      _state_t<Sndr> state{&result};

      // Launch the sender with a continuation that will fill in a variant
      auto opstate = __async::connect(static_cast<Sndr&&>(sndr), _rcvr_t{&state});
      __async::start(opstate);

      // Wait for the variant to be filled in, and process any work that
      // may be delegated to this thread.
      state._loop.run();

      if (state._eptr)
      {
        ::std::rethrow_exception(state._eptr);
      }

      return result; // uses NRVO to "return" the result
    }
  }
};

_CCCL_GLOBAL_CONSTANT sync_wait_t sync_wait{};
} // namespace cuda::experimental::__async

#  include <cuda/experimental/__async/epilogue.cuh>

#endif // !defined(__CUDA_ARCH__)

#endif
