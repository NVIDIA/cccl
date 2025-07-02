//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_SEQUENCE
#define __CUDAX_EXECUTION_SEQUENCE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__tuple_dir/ignore.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/lazy.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Rcvr, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    _CCCL_API constexpr explicit __state_t(_Rcvr&& __rcvr, _Sndr2&& __sndr2)
        : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
        , __opstate2_(execution::connect(static_cast<_Sndr2&&>(__sndr2), __ref_rcvr(__rcvr_)))
    {}

    _Rcvr __rcvr_;
    connect_result_t<_Sndr2, __rcvr_ref_t<_Rcvr>> __opstate2_;
  };

  template <class _Rcvr, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _Values>
    _CCCL_API constexpr void set_value(_Values&&...) noexcept
    {
      execution::start(__state_->__opstate2_);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __error) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__error));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __state_t<_Rcvr, _Sndr2>* __state_;
  };

  template <class _Rcvr, class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;

    _CCCL_API __opstate_t(_Sndr1&& __sndr1, _Sndr2&& __sndr2, _Rcvr&& __rcvr)
        : __state_(static_cast<_Rcvr&&>(__rcvr), static_cast<_Sndr2&&>(__sndr2))
        , __opstate1_(execution::connect(static_cast<_Sndr1&&>(__sndr1), __rcvr_t<_Rcvr, _Sndr2>{&__state_}))
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate1_);
    }

    __state_t<_Rcvr, _Sndr2> __state_;
    connect_result_t<_Sndr1, __rcvr_t<_Rcvr, _Sndr2>> __opstate1_;
  };

public:
  template <class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sndr1, class _Sndr2>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const;
};

template <class _Sndr1, class _Sndr2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__completions1) = get_child_completion_signatures<_Self, _Sndr1, _Env...>())
    {
      _CUDAX_LET_COMPLETIONS(auto(__completions2) = get_child_completion_signatures<_Self, _Sndr2, _Env...>())
      {
        // ignore the first sender's value completions
        return __completions2 + transform_completion_signatures(__completions1, __swallow_transform{});
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) &&
  {
    using __opstate_t = __opstate_t<_Rcvr, _Sndr1, _Sndr2>;
    return __opstate_t{static_cast<_Sndr1&&>(__sndr1_), static_cast<_Sndr2>(__sndr2_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const&
  {
    using __opstate_t = __opstate_t<_Rcvr, const _Sndr1&, const _Sndr2&>;
    return __opstate_t{__sndr1_, __sndr2_, static_cast<_Rcvr&&>(__rcvr)};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr2>>
  {
    return __fwd_env(execution::get_env(__sndr2_));
  }

  _CCCL_NO_UNIQUE_ADDRESS sequence_t __tag_;
  _CCCL_NO_UNIQUE_ADDRESS _CUDA_VSTD::__ignore_t __ign_;
  _Sndr1 __sndr1_;
  _Sndr2 __sndr2_;
};

template <class _Sndr1, class _Sndr2>
_CCCL_TRIVIAL_API constexpr auto sequence_t::operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const
{
  using __dom_t _CCCL_NODEBUG_ALIAS  = __early_domain_of_t<_Sndr1>;
  using __sndr_t _CCCL_NODEBUG_ALIAS = sequence_t::__sndr_t<_Sndr1, _Sndr2>;
  return transform_sender(__dom_t{}, __sndr_t{{}, {}, static_cast<_Sndr1&&>(__sndr1), static_cast<_Sndr2&&>(__sndr2)});
}

template <class _Sndr1, class _Sndr2>
inline constexpr size_t structured_binding_size<sequence_t::__sndr_t<_Sndr1, _Sndr2>> = 4;

_CCCL_GLOBAL_CONSTANT sequence_t sequence{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SEQUENCE
