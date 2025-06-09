//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_JUST_FROM
#define __CUDAX_EXECUTION_JUST_FROM

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// Map from a disposition to the corresponding tag types:
namespace __detail
{
template <__disposition_t, class _Void = void>
extern _CUDA_VSTD::__undefined<_Void> __just_from_tag;
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
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(just_from_t) _CCCL_PREFERRED_NAME(just_error_from_t)
  _CCCL_PREFERRED_NAME(just_stopped_from_t) __just_from_t
{
private:
  using _JustTag _CCCL_NODEBUG_ALIAS = decltype(__detail::__just_from_tag<_Disposition>());
  using _SetTag _CCCL_NODEBUG_ALIAS  = decltype(__detail::__set_tag<_Disposition>());

  using __diag_t _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::conditional_t<_SetTag{} == set_error,
                              _AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT,
                              _A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS>;

  template <class... _Ts>
  using __error_t _CCCL_NODEBUG_ALIAS =
    _ERROR<_WHERE(_IN_ALGORITHM, _JustTag), _WHAT(__diag_t), _WITH_COMPLETION_SIGNATURE<_SetTag(_Ts...)>>;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __probe_fn
  {
    template <class... _Ts>
    auto operator()(_Ts&&... __ts) const noexcept
      -> _CUDA_VSTD::conditional_t<__signature_disposition<_SetTag(_Ts...)> != __invalid_disposition,
                                   completion_signatures<_SetTag(_Ts...)>,
                                   __error_t<_Ts...>>;
  };

  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __complete_fn
  {
    _Rcvr& __rcvr_;

    template <class... _Ts>
    _CCCL_API auto operator()(_Ts&&... __ts) const noexcept
    {
      _SetTag{}(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Ts&&>(__ts)...);
    }
  };

  template <class _Rcvr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API void start() & noexcept
    {
      static_cast<_Fn&&>(__fn_)(__complete_fn<_Rcvr>{__rcvr_});
    }

    _Rcvr __rcvr_;
    _Fn __fn_;
  };

public:
  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Fn __fn) const noexcept -> __sndr_t<_Fn>;
};

template <__disposition_t _Disposition>
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __just_from_t<_Disposition>::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;

  _CCCL_NO_UNIQUE_ADDRESS _JustTag __tag_;
  _Fn __fn_;

  template <class _Self, class...>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    return _CUDA_VSTD::__call_result_t<_Fn, __probe_fn>{};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) && //
    noexcept(__nothrow_decay_copyable<_Rcvr, _Fn>) -> __opstate<_Rcvr, _Fn>
  {
    return __opstate<_Rcvr, _Fn>{static_cast<_Rcvr&&>(__rcvr), static_cast<_Fn&&>(__fn_)};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) const& //
    noexcept(__nothrow_decay_copyable<_Rcvr, _Fn const&>) -> __opstate<_Rcvr, _Fn>
  {
    return __opstate<_Rcvr, _Fn>{static_cast<_Rcvr&&>(__rcvr), __fn_};
  }
};

template <__disposition_t _Disposition>
template <class _Fn>
_CCCL_TRIVIAL_API constexpr auto __just_from_t<_Disposition>::operator()(_Fn __fn) const noexcept -> __sndr_t<_Fn>
{
  using __completions _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__call_result_t<_Fn, __probe_fn>;
  static_assert(__valid_completion_signatures<__completions>,
                "The function passed to just_from must return an instance of a specialization of "
                "completion_signatures<>.");
  return __sndr_t<_Fn>{{}, static_cast<_Fn&&>(__fn)};
}

template <class _Fn>
inline constexpr size_t structured_binding_size<just_from_t::__sndr_t<_Fn>> = 2;
template <class _Fn>
inline constexpr size_t structured_binding_size<just_error_from_t::__sndr_t<_Fn>> = 2;
template <class _Fn>
inline constexpr size_t structured_binding_size<just_stopped_from_t::__sndr_t<_Fn>> = 2;

_CCCL_GLOBAL_CONSTANT auto just_from         = just_from_t{};
_CCCL_GLOBAL_CONSTANT auto just_error_from   = just_error_from_t{};
_CCCL_GLOBAL_CONSTANT auto just_stopped_from = just_stopped_from_t{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_JUST_FROM
