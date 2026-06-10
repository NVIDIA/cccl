/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2025 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __CUDAX_EXECUTION_TRAMPOLINE_SCHEDULER
#define __CUDAX_EXECUTION_TRAMPOLINE_SCHEDULER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstdlib/abs.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/stop_token.cuh>

// include this last:
#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __detail
{
template <class _Operation>
struct __trampoline_state
{
  static thread_local __trampoline_state* __current_;

  _CCCL_HOST_API __trampoline_state(size_t __max_recursion_depth, size_t __max_recursion_size) noexcept
      : __max_recursion_size_(__max_recursion_size)
      , __max_recursion_depth_(__max_recursion_depth)
  {
    __current_ = this;
  }

  _CCCL_HOST_API ~__trampoline_state()
  {
    __current_ = nullptr;
  }

  _CCCL_HOST_API void __drain() noexcept;

  // these origin schedule frame limits will apply to all
  // nested trampoline instances on this thread
  const size_t __max_recursion_size_;
  const size_t __max_recursion_depth_;

  // track state of origin schedule frame
  intptr_t __recursion_origin_ = 0;
  size_t __recursion_depth_    = 1;
  _Operation* __head_          = nullptr;
  _Operation* __tail_          = nullptr;
};
} // namespace __detail

class trampoline_scheduler
{
  struct __schedule_sender;

public:
  using scheduler_concept = scheduler_t;

  _CCCL_HOST_API trampoline_scheduler() noexcept
      : __max_recursion_size_(4096)
      , __max_recursion_depth_(16)
  {}

  _CCCL_HOST_API explicit trampoline_scheduler(size_t __max_recursion_depth) noexcept
      : __max_recursion_size_(4096)
      , __max_recursion_depth_(__max_recursion_depth)
  {}

  _CCCL_HOST_API explicit trampoline_scheduler(size_t __max_recursion_depth, size_t __max_recursion_size) noexcept
      : __max_recursion_size_(__max_recursion_size)
      , __max_recursion_depth_(__max_recursion_depth)
  {}

  [[nodiscard]]
  _CCCL_HOST_API auto schedule() const noexcept -> __schedule_sender
  {
    return __schedule_sender{__max_recursion_size_, __max_recursion_depth_};
  }

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  _CCCL_HOST_API auto operator==(const trampoline_scheduler&) const noexcept -> bool = default;

#else

  [[nodiscard]]
  _CCCL_HOST_API friend constexpr bool
  operator==(const trampoline_scheduler& __a, const trampoline_scheduler& __b) noexcept
  {
    return __a.__max_recursion_size_ == __b.__max_recursion_size_
        && __a.__max_recursion_depth_ == __b.__max_recursion_depth_;
  }

  [[nodiscard]]
  _CCCL_HOST_API friend constexpr bool
  operator!=(const trampoline_scheduler& __a, const trampoline_scheduler& __b) noexcept
  {
    return !(__a == __b);
  }

#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

private:
  struct __operation_base
  {
    using operation_state_concept = operation_state_t;
    using __execute_fn            = void(__operation_base*) noexcept;

    _CCCL_HOST_API explicit __operation_base(__execute_fn* __execute, size_t __max_size, size_t __max_depth) noexcept
        : __execute_(__execute)
        , __max_recursion_size_(__max_size)
        , __max_recursion_depth_(__max_depth)
    {}

    _CCCL_HOST_API void __execute() noexcept
    {
      __execute_(this);
    }

    _CCCL_HOST_API void start() & noexcept
    {
      auto* __current_state = __detail::__trampoline_state<__operation_base>::__current_;

      if (__current_state == nullptr)
      {
        // origin schedule frame on this thread
        __detail::__trampoline_state<__operation_base> __state{__max_recursion_depth_, __max_recursion_size_};
        __execute();
        __state.__drain();
      }
      else
      {
        // recursive schedule frame on this thread

        // calculate stack consumption for this schedule
        size_t __current_size =
          ::cuda::std::abs(reinterpret_cast<intptr_t>(&__current_state) - __current_state->__recursion_origin_);

        if (__current_size < __current_state->__max_recursion_size_
            && __current_state->__recursion_depth_ < __current_state->__max_recursion_depth_)
        {
          // inline this recursive schedule
          ++__current_state->__recursion_depth_;
          __execute();
        }
        else
        {
          // Exceeded recursion limit.

          // push this recursive schedule to list tail
          __prev_ = ::cuda::std::exchange(__current_state->__tail_, static_cast<__operation_base*>(this));
          if (__prev_ != nullptr)
          {
            // was not empty
            ::cuda::std::exchange(__prev_->__next_, static_cast<__operation_base*>(this));
          }
          else
          {
            // was empty
            ::cuda::std::exchange(__current_state->__head_, static_cast<__operation_base*>(this));
          }
        }
      }
    }

    __operation_base* __prev_ = nullptr;
    __operation_base* __next_ = nullptr;
    __execute_fn* __execute_;
    const size_t __max_recursion_size_;
    const size_t __max_recursion_depth_;
  };

  template <class _Rcvr>
  struct __operation : __operation_base
  {
    _CCCL_HOST_API explicit __operation(_Rcvr __rcvr, size_t __max_size, size_t __max_depth) noexcept
        : __operation_base(&__operation::__execute_impl, __max_size, __max_depth)
        , __rcvr_(static_cast<_Rcvr&&>(__rcvr))
    {}

    _CCCL_HOST_API static void __execute_impl(__operation_base* __op) noexcept
    {
      auto& __self = *static_cast<__operation*>(__op);
      if constexpr (unstoppable_token<stop_token_of_t<env_of_t<_Rcvr&>>>)
      {
        execution::set_value(static_cast<_Rcvr&&>(__self.__rcvr_));
      }
      else
      {
        if (execution::get_stop_token(get_env(__self.__rcvr_)).stop_requested())
        {
          execution::set_stopped(static_cast<_Rcvr&&>(__self.__rcvr_));
        }
        else
        {
          execution::set_value(static_cast<_Rcvr&&>(__self.__rcvr_));
        }
      }
    }

  private:
    _Rcvr __rcvr_;
  };

  struct __schedule_sender
  {
    using sender_concept = sender_t;
    template <class _Env>
    using __completions_in_t =
      ::cuda::std::_If<unstoppable_token<stop_token_of_t<_Env>>,
                       completion_signatures<set_value_t()>,
                       completion_signatures<set_value_t(), set_stopped_t()>>;

    _CCCL_HOST_API explicit __schedule_sender(size_t __max_size, size_t __max_depth) noexcept
        : __max_recursion_size_(__max_size)
        , __max_recursion_depth_(__max_depth)
    {}

    _CCCL_TEMPLATE(class _Rcvr)
    _CCCL_REQUIRES(receiver_of<_Rcvr, __completions_in_t<env_of_t<_Rcvr&>>>)
    [[nodiscard]]
    _CCCL_HOST_API auto connect(_Rcvr __rcvr) const noexcept -> __operation<_Rcvr>
    {
      return __operation<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __max_recursion_size_, __max_recursion_depth_};
    }

    template <class _Self, class _Env>
    _CCCL_HOST_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept -> __completions_in_t<_Env>
    {
      return {};
    }

    [[nodiscard]]
    _CCCL_HOST_API auto query(get_completion_scheduler_t<set_value_t>, ::cuda::std::__ignore_t = {}) const noexcept
      -> trampoline_scheduler
    {
      return trampoline_scheduler{__max_recursion_depth_};
    }

    [[nodiscard]]
    _CCCL_HOST_API auto get_env() const noexcept -> const __schedule_sender&
    {
      return *this;
    }

    const size_t __max_recursion_size_;
    const size_t __max_recursion_depth_;
  };

  size_t __max_recursion_size_;
  size_t __max_recursion_depth_;
};

namespace __detail
{
template <class _Operation>
thread_local __trampoline_state<_Operation>* __trampoline_state<_Operation>::__current_ = nullptr;

template <class _Operation>
_CCCL_HOST_API void __trampoline_state<_Operation>::__drain() noexcept
{
  while (__head_ != nullptr)
  {
    // pop the head of the list
    _Operation* __op = ::cuda::std::exchange(__head_, __head_->__next_);
    __op->__next_    = nullptr;
    __op->__prev_    = nullptr;
    if (__head_ != nullptr)
    {
      // is not empty
      __head_->__prev_ = nullptr;
    }
    else
    {
      // is empty
      __tail_ = nullptr;
    }

    // reset the origin schedule frame state
    __recursion_origin_ = reinterpret_cast<intptr_t>(&__op);
    __recursion_depth_  = 1;

    __op->__execute();
  }
}
} // namespace __detail
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TRAMPOLINE_SCHEDULER
