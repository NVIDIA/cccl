//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/exception.cuh>
#include <cuda/experimental/__async/lazy.cuh>
#include <cuda/experimental/__async/rcvr_ref.cuh>
#include <cuda/experimental/__async/variant.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
struct __seq
{
  template <class _Rcvr, class _Sndr1, class _Sndr2>
  struct __args
  {
    using __rcvr_t  = _Rcvr;
    using __sndr1_t = _Sndr1;
    using __sndr2_t = _Sndr2;
  };

  template <class _Zip>
  struct __opstate
  {
    using operation_state_concept = operation_state_t;

    using __args_t  = __unzip<_Zip>; // __unzip<_Zip> is __args<_Rcvr, _Sndr1, _Sndr2>
    using __rcvr_t  = typename __args_t::__rcvr_t;
    using __sndr1_t = typename __args_t::__sndr1_t;
    using __sndr2_t = typename __args_t::__sndr2_t;

    using completion_signatures = //
      transform_completion_signatures_of< //
        __sndr1_t,
        __opstate*,
        completion_signatures_of_t<__sndr2_t, __rcvr_ref_t<__rcvr_t&>>,
        _CUDA_VSTD::__type_always<__async::completion_signatures<>>::__call>; // swallow the first sender's value
                                                                              // completions

    _CUDAX_API friend env_of_t<__rcvr_t> get_env(const __opstate* __self) noexcept
    {
      return __async::get_env(__self->__rcvr_);
    }

    __rcvr_t __rcvr_;
    connect_result_t<__sndr1_t, __opstate*> __opstate1_;
    connect_result_t<__sndr2_t, __rcvr_ref_t<__rcvr_t&>> __opstate2_;

    _CUDAX_API __opstate(__sndr1_t&& __sndr1, __sndr2_t&& __sndr2, __rcvr_t&& __rcvr)
        : __rcvr_(static_cast<__rcvr_t&&>(__rcvr))
        , __opstate1_(__async::connect(static_cast<__sndr1_t&&>(__sndr1), this))
        , __opstate2_(__async::connect(static_cast<__sndr2_t&&>(__sndr2), __rcvr_ref(__rcvr_)))
    {}

    _CUDAX_API void start() noexcept
    {
      __async::start(__opstate1_);
    }

    template <class... _Values>
    _CUDAX_API void set_value(_Values&&...) && noexcept
    {
      __async::start(__opstate2_);
    }

    template <class _Error>
    _CUDAX_API void set_error(_Error&& __error) && noexcept
    {
      __async::set_error(static_cast<__rcvr_t&&>(__rcvr_), static_cast<_Error&&>(__error));
    }

    _CUDAX_API void set_stopped() && noexcept
    {
      __async::set_stopped(static_cast<__rcvr_t&&>(__rcvr_));
    }
  };

  template <class _Sndr1, class _Sndr2>
  struct __sndr_t;

  template <class _Sndr1, class _Sndr2>
  _CUDAX_API auto operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const -> __sndr_t<_Sndr1, _Sndr2>;
};

template <class _Sndr1, class _Sndr2>
struct __seq::__sndr_t
{
  using sender_concept = sender_t;
  using __sndr1_t      = _Sndr1;
  using __sndr2_t      = _Sndr2;

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) &&
  {
    using __opstate_t = __opstate<__zip<__args<_Rcvr, _Sndr1, _Sndr2>>>;
    return __opstate_t{static_cast<_Sndr1&&>(__sndr1_), static_cast<_Sndr2>(__sndr2_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const&
  {
    using __opstate_t = __opstate<__zip<__args<_Rcvr, const _Sndr1&, const _Sndr2&>>>;
    return __opstate_t{__sndr1_, __sndr2_, static_cast<_Rcvr&&>(__rcvr)};
  }

  _CUDAX_API env_of_t<_Sndr2> get_env() const noexcept
  {
    return __async::get_env(__sndr2_);
  }

  _CCCL_NO_UNIQUE_ADDRESS __seq __tag_;
  _CCCL_NO_UNIQUE_ADDRESS __ignore __ign_;
  __sndr1_t __sndr1_;
  __sndr2_t __sndr2_;
};

template <class _Sndr1, class _Sndr2>
_CUDAX_API auto __seq::operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const -> __sndr_t<_Sndr1, _Sndr2>
{
  return __sndr_t<_Sndr1, _Sndr2>{{}, {}, static_cast<_Sndr1&&>(__sndr1), static_cast<_Sndr2&&>(__sndr2)};
}

using sequence_t = __seq;
_CCCL_GLOBAL_CONSTANT sequence_t sequence{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
