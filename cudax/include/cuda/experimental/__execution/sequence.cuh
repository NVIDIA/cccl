//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_SEQUENCE
#define __CUDAX_ASYNC_DETAIL_SEQUENCE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/lazy.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT __seq_t
{
private:
  template <class _Rcvr, class _Sndr1, class _Sndr2>
  struct __args
  {
    using __rcvr_t _CCCL_NODEBUG_ALIAS  = _Rcvr;
    using __sndr1_t _CCCL_NODEBUG_ALIAS = _Sndr1;
    using __sndr2_t _CCCL_NODEBUG_ALIAS = _Sndr2;
  };

  template <class _Zip>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;

    using __args_t _CCCL_NODEBUG_ALIAS  = __unzip<_Zip>; // __unzip<_Zip> is __args<_Rcvr, _Sndr1, _Sndr2>
    using __rcvr_t _CCCL_NODEBUG_ALIAS  = typename __args_t::__rcvr_t;
    using __sndr1_t _CCCL_NODEBUG_ALIAS = typename __args_t::__sndr1_t;
    using __sndr2_t _CCCL_NODEBUG_ALIAS = typename __args_t::__sndr2_t;
    using __env_t _CCCL_NODEBUG_ALIAS   = env_of_t<__rcvr_t>;

    _CCCL_API __opstate(__sndr1_t&& __sndr1, __sndr2_t&& __sndr2, __rcvr_t&& __rcvr)
        : __rcvr_(static_cast<__rcvr_t&&>(__rcvr))
        , __opstate1_(execution::connect(static_cast<__sndr1_t&&>(__sndr1), __rcvr_ref{*this}))
        , __opstate2_(execution::connect(static_cast<__sndr2_t&&>(__sndr2), __rcvr_ref{__rcvr_}))
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate);

    _CCCL_API void start() noexcept
    {
      execution::start(__opstate1_);
    }

    template <class... _Values>
    _CCCL_API void set_value(_Values&&...) && noexcept
    {
      execution::start(__opstate2_);
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __error) && noexcept
    {
      execution::set_error(static_cast<__rcvr_t&&>(__rcvr_), static_cast<_Error&&>(__error));
    }

    _CCCL_API void set_stopped() && noexcept
    {
      execution::set_stopped(static_cast<__rcvr_t&&>(__rcvr_));
    }

    _CCCL_API auto get_env() const noexcept -> __env_t
    {
      return execution::get_env(__rcvr_);
    }

    __rcvr_t __rcvr_;
    connect_result_t<__sndr1_t, __rcvr_ref<__opstate, __env_t>> __opstate1_;
    connect_result_t<__sndr2_t, __rcvr_ref<__rcvr_t>> __opstate2_;
  };

public:
  template <class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sndr1, class _Sndr2>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const;
};

template <class _Sndr1, class _Sndr2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __seq_t::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;
  using __sndr1_t _CCCL_NODEBUG_ALIAS      = _Sndr1;
  using __sndr2_t _CCCL_NODEBUG_ALIAS      = _Sndr2;

  template <class _Self, class... _Env>
  _CCCL_API static constexpr auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__completions1) = get_child_completion_signatures<_Self, _Sndr1, _Env...>())
    {
      _CUDAX_LET_COMPLETIONS(auto(__completions2) = get_child_completion_signatures<_Self, _Sndr2, _Env...>())
      {
        // ignore the first sender's value completions
        return __completions2 + transform_completion_signatures(__completions1, __swallow_transform());
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) &&
  {
    using __opstate_t _CCCL_NODEBUG_ALIAS = __opstate<__zip<__args<_Rcvr, _Sndr1, _Sndr2>>>;
    return __opstate_t{static_cast<_Sndr1&&>(__sndr1_), static_cast<_Sndr2>(__sndr2_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) const&
  {
    using __opstate_t _CCCL_NODEBUG_ALIAS = __opstate<__zip<__args<_Rcvr, const _Sndr1&, const _Sndr2&>>>;
    return __opstate_t{__sndr1_, __sndr2_, static_cast<_Rcvr&&>(__rcvr)};
  }

  _CCCL_API auto get_env() const noexcept -> env_of_t<_Sndr2>
  {
    return execution::get_env(__sndr2_);
  }

  _CCCL_NO_UNIQUE_ADDRESS __seq_t __tag_;
  _CCCL_NO_UNIQUE_ADDRESS __ignore __ign_;
  __sndr1_t __sndr1_;
  __sndr2_t __sndr2_;
};

template <class _Sndr1, class _Sndr2>
_CCCL_TRIVIAL_API constexpr auto __seq_t::operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const
{
  using __dom_t _CCCL_NODEBUG_ALIAS  = domain_for_t<_Sndr1>;
  using __sndr_t _CCCL_NODEBUG_ALIAS = __seq_t::__sndr_t<_Sndr1, _Sndr2>;
  return transform_sender(__dom_t{}, __sndr_t{{}, {}, static_cast<_Sndr1&&>(__sndr1), static_cast<_Sndr2&&>(__sndr2)});
}

template <class _Sndr1, class _Sndr2>
inline constexpr size_t structured_binding_size<__seq_t::__sndr_t<_Sndr1, _Sndr2>> = 4;

using sequence_t _CCCL_NODEBUG_ALIAS = __seq_t;
_CCCL_GLOBAL_CONSTANT sequence_t sequence{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif
