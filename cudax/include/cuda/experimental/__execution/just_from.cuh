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

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT;
struct _A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS;

template <class _JustFromTag, class _SetTag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __just_from_t
{
  _CUDAX_SEMI_PRIVATE:
  friend struct just_from_t;
  friend struct just_error_from_t;
  friend struct just_stopped_from_t;

  using __just_from_tag_t = _JustFromTag;

  using __diag_t _CCCL_NODEBUG_ALIAS =
    ::cuda::std::conditional_t<_SetTag{} == set_error,
                               _AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT,
                               _A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS>;

  template <class... _Ts>
  using __error_t _CCCL_NODEBUG_ALIAS =
    _ERROR<_WHERE(_IN_ALGORITHM, _JustFromTag), _WHAT(__diag_t), _WITH_COMPLETION_SIGNATURE<_SetTag(_Ts...)>>;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __probe_fn
  {
    template <class... _Ts>
    _CCCL_API auto operator()(_Ts&&... __ts) const noexcept
      -> ::cuda::std::_If<__detail::__signature_disposition<_SetTag(_Ts...)> != __disposition::__invalid,
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
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API constexpr void start() noexcept
    {
      static_cast<_Fn&&>(__fn_)(__complete_fn<_Rcvr>{__rcvr_});
    }

    _Rcvr __rcvr_;
    _Fn __fn_;
  };

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_base_t;

public:
  template <class _Fn>
  _CCCL_NODEBUG_API constexpr auto operator()(_Fn __fn) const noexcept;
};

struct just_from_t : __just_from_t<just_from_t, set_value_t>
{
  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;
};

struct just_error_from_t : __just_from_t<just_error_from_t, set_error_t>
{
  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;
};

struct just_stopped_from_t : __just_from_t<just_stopped_from_t, set_stopped_t>
{
  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;
};

template <class _JustFromTag, class _SetTag>
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __just_from_t<_JustFromTag, _SetTag>::__sndr_base_t
{
  using sender_concept = sender_t;

  template <class _Self, class...>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    return __call_result_t<_Fn, __probe_fn>{};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && //
    noexcept(__nothrow_decay_copyable<_Rcvr, _Fn>) -> __opstate_t<_Rcvr, _Fn>
  {
    return __opstate_t<_Rcvr, _Fn>{static_cast<_Rcvr&&>(__rcvr), static_cast<_Fn&&>(__fn_)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& //
    noexcept(__nothrow_decay_copyable<_Rcvr, _Fn const&>) -> __opstate_t<_Rcvr, _Fn>
  {
    return __opstate_t<_Rcvr, _Fn>{static_cast<_Rcvr&&>(__rcvr), __fn_};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept
  {
    return __inln_attrs_t<_SetTag>{};
  }

  _CCCL_NO_UNIQUE_ADDRESS __just_from_tag_t __tag_;
  _Fn __fn_;
};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_from_t::__sndr_t : __just_from_t<just_t, set_value_t>::__sndr_base_t<_Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_error_from_t::__sndr_t
    : __just_from_t<just_error_t, set_error_t>::__sndr_base_t<_Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_stopped_from_t::__sndr_t
    : __just_from_t<just_stopped_t, set_stopped_t>::__sndr_base_t<_Fn>
{};

template <class _JustFromTag, class _SetTag>
template <class _Fn>
_CCCL_NODEBUG_API constexpr auto __just_from_t<_JustFromTag, _SetTag>::operator()(_Fn __fn) const noexcept
{
  using __sndr_t                          = typename _JustFromTag::template __sndr_t<_Fn>;
  using __completions _CCCL_NODEBUG_ALIAS = __call_result_t<_Fn, __probe_fn>;
  static_assert(__valid_completion_signatures<__completions>,
                "The function passed to just_from must return an instance of a specialization of "
                "completion_signatures<>.");
  return __sndr_t{{{}, static_cast<_Fn&&>(__fn)}};
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
