//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_JUST_FROM
#define __CUDAX_ASYNC_DETAIL_JUST_FROM

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/config.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/rcvr_ref.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
// Forward declarations of the just* tag types:
struct just_from_t;
struct just_error_from_t;
struct just_stopped_from_t;

// Map from a disposition to the corresponding tag types:
namespace __detail
{
template <__disposition_t, class _Void = void>
extern __undefined<_Void> __just_from_tag;
template <class _Void>
extern __fn_t<just_from_t>* __just_from_tag<__value, _Void>;
template <class _Void>
extern __fn_t<just_error_from_t>* __just_from_tag<__error, _Void>;
template <class _Void>
extern __fn_t<just_stopped_from_t>* __just_from_tag<__stopped, _Void>;
} // namespace __detail

struct _AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT;
struct _A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS;

template <__disposition_t _Disposition>
struct __just_from
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC

  using _JustTag = decltype(__detail::__just_from_tag<_Disposition>());
  using _SetTag  = decltype(__detail::__set_tag<_Disposition>());

  using __diag_t = _CUDA_VSTD::_If<_CUDA_VSTD::is_same_v<_SetTag, set_error_t>,
                                   _AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT,
                                   _A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS>;

  template <class... _Ts>
  using __error_t =
    _ERROR<_WHERE(_IN_ALGORITHM, _JustTag), _WHAT(__diag_t), _WITH_COMPLETION_SIGNATURE<_SetTag(_Ts...)>>;

  struct __probe_fn
  {
    template <class... _Ts>
    auto operator()(_Ts&&... __ts) const noexcept
      -> _CUDA_VSTD::
        _If<__is_valid_signature<_SetTag(_Ts...)>, completion_signatures<_SetTag(_Ts...)>, __error_t<_Ts...>>;
  };

  template <class _Rcvr = receiver_archetype>
  struct __complete_fn
  {
    _Rcvr& __rcvr_;

    template <class... _Ts>
    _CCCL_HOST_DEVICE auto operator()(_Ts&&... __ts) const noexcept
    {
      _SetTag()(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Ts&&>(__ts)...);
    }
  };

  template <class _Rcvr, class _Fn>
  struct __opstate
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = __call_result_t<_Fn, __probe_fn>;
    static_assert(__is_completion_signatures<completion_signatures>);

    _Rcvr __rcvr_;
    _Fn __fn_;

    _CCCL_HOST_DEVICE void start() & noexcept
    {
      static_cast<_Fn&&>(__fn_)(__complete_fn<_Rcvr>{__rcvr_});
    }
  };

  template <class _Fn>
  struct __sndr_t
  {
    using sender_concept = sender_t;

    _CCCL_NO_UNIQUE_ADDRESS _JustTag __tag_;
    _Fn __fn_;

    template <class _Rcvr>
    _CCCL_HOST_DEVICE __opstate<_Rcvr, _Fn> connect(_Rcvr __rcvr) && //
      noexcept(__nothrow_decay_copyable<_Rcvr, _Fn>)
    {
      return __opstate<_Rcvr, _Fn>{static_cast<_Rcvr&&>(__rcvr), static_cast<_Fn&&>(__fn_)};
    }

    template <class _Rcvr>
    _CCCL_HOST_DEVICE __opstate<_Rcvr, _Fn> connect(_Rcvr __rcvr) const& //
      noexcept(__nothrow_decay_copyable<_Rcvr, _Fn const&>)
    {
      return __opstate<_Rcvr, _Fn>{static_cast<_Rcvr&&>(__rcvr), __fn_};
    }
  };

public:
  template <class _Fn>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(_Fn __fn) const noexcept
  {
    using __completions = __call_result_t<_Fn, __probe_fn>;
    static_assert(__is_completion_signatures<__completions>,
                  "The function passed to just_from must return an instance of a specialization of "
                  "completion_signatures<>.");
    return __sndr_t<_Fn>{{}, static_cast<_Fn&&>(__fn)};
  }
};

_CCCL_GLOBAL_CONSTANT struct just_from_t : __just_from<__value>
{
} just_from{};

_CCCL_GLOBAL_CONSTANT struct just_error_from_t : __just_from<__error>
{
} just_error_from{};

_CCCL_GLOBAL_CONSTANT struct just_stopped_from_t : __just_from<__stopped>
{
} just_stopped_from{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
