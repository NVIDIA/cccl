//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_THEN
#define __CUDAX_EXECUTION_THEN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // warning C4702: unreachable code

namespace cuda::experimental::execution
{
namespace __upon
{
template <bool IsVoid, bool _Nothrow>
struct __completion_fn
{ // non-void, potentially throwing case
  template <class _Result>
  using __call _CCCL_NODEBUG_ALIAS = completion_signatures<set_value_t(_Result), set_error_t(::std::exception_ptr)>;
};

template <>
struct __completion_fn<true, false>
{ // void, potentially throwing case
  template <class>
  using __call _CCCL_NODEBUG_ALIAS = completion_signatures<set_value_t(), set_error_t(::std::exception_ptr)>;
};

template <>
struct __completion_fn<false, true>
{ // non-void, non-throwing case
  template <class _Result>
  using __call _CCCL_NODEBUG_ALIAS = completion_signatures<set_value_t(_Result)>;
};

template <>
struct __completion_fn<true, true>
{ // void, non-throwing case
  template <class>
  using __call _CCCL_NODEBUG_ALIAS = completion_signatures<set_value_t()>;
};

template <class _Result, bool _Nothrow>
using __completion_ _CCCL_NODEBUG_ALIAS =
  ::cuda::std::__type_call1<__completion_fn<__same_as<_Result, void>, _Nothrow>, _Result>;

template <class _Fn, class... _Ts>
using __completion _CCCL_NODEBUG_ALIAS = __completion_<__call_result_t<_Fn, _Ts...>, __nothrow_callable<_Fn, _Ts...>>;

template <class _Fn, class _Rcvr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
{
  _Rcvr __rcvr_;
  _Fn __fn_;
};
} // namespace __upon

template <class _UponTag, class _SetTag>
struct __upon_t
{
  _CUDAX_SEMI_PRIVATE :
  friend struct then_t;
  friend struct upon_error_t;
  friend struct upon_stopped_t;

  using __upon_tag_t = _UponTag;
  using __set_tag_t  = _SetTag;

  template <class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    _CCCL_EXEC_CHECK_DISABLE
    template <bool _CanThrow = false, class... _Ts>
    _CCCL_API void __set(_Ts&&... __ts) noexcept(!_CanThrow)
    {
      if constexpr (_CanThrow || __nothrow_callable<_Fn, _Ts...>)
      {
        if constexpr (__same_as<void, __call_result_t<_Fn, _Ts...>>)
        {
          static_cast<_Fn&&>(__state_->__fn_)(static_cast<_Ts&&>(__ts)...);
          execution::set_value(static_cast<_Rcvr&&>(__state_->__rcvr_));
        }
        else
        {
          // msvc warns that this is unreachable code, but it is reachable.
          execution::set_value(static_cast<_Rcvr&&>(__state_->__rcvr_),
                               static_cast<_Fn&&>(__state_->__fn_)(static_cast<_Ts&&>(__ts)...));
        }
      }
      else
      {
        _CCCL_TRY
        {
          __set<true>(static_cast<_Ts&&>(__ts)...);
        }
        _CCCL_CATCH_ALL
        {
          execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), ::std::current_exception());
        }
      }
    }

    template <class _Tag, class... _Ts>
    _CCCL_NODEBUG_API void __complete(_Tag, _Ts&&... __ts) noexcept
    {
      if constexpr (_Tag{} == _SetTag{})
      {
        __set(static_cast<_Ts&&>(__ts)...);
      }
      else
      {
        _Tag{}(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Ts&&>(__ts)...);
      }
    }

    template <class... _Ts>
    _CCCL_API void set_value(_Ts&&... __ts) noexcept
    {
      __complete(set_value_t{}, static_cast<_Ts&&>(__ts)...);
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __error) noexcept
    {
      __complete(set_error_t{}, static_cast<_Error&&>(__error));
    }

    _CCCL_API void set_stopped() noexcept
    {
      __complete(set_stopped_t{});
    }

    _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __upon::__state_t<_Fn, _Rcvr>* __state_;
  };

  template <class _CvSndr, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __rcvr_t                = __upon_t::__rcvr_t<_Fn, _Rcvr>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Fn __fn)
        : __state_{static_cast<_Rcvr&&>(__rcvr), static_cast<_Fn&&>(__fn)}
        , __opstate_{execution::connect(static_cast<_CvSndr&&>(__sndr), __rcvr_t{&__state_})}
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate_);
    }

    __upon::__state_t<_Fn, _Rcvr> __state_;
    connect_result_t<_CvSndr, __rcvr_t> __opstate_;
  };

  template <class _Fn>
  struct __transform_args_fn
  {
    template <class... _Ts>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const
    {
      if constexpr (__callable<_Fn, _Ts...>)
      {
        return __upon::__completion<_Fn, _Ts...>{};
      }
      else
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _UponTag),
                                            _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                            _WITH_FUNCTION(_Fn),
                                            _WITH_ARGUMENTS(_Ts...)>();
      }
    }
  };

  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_base_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_HIDDEN __closure_base_t // hidden visibility because member __fn_ is hidden if it is an
                                                       // extended (host/device) lambda
  {
    template <class _Sndr>
    _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr) -> __call_result_t<__upon_tag_t, _Sndr, _Fn>
    {
      return __upon_tag_t{}(static_cast<_Sndr&&>(__sndr), static_cast<_Fn&&>(__fn_));
    }

    template <class _Sndr>
    _CCCL_NODEBUG_API friend constexpr auto operator|(_Sndr __sndr, __closure_base_t __self) //
      -> __call_result_t<__upon_tag_t, _Sndr, _Fn>
    {
      return __upon_tag_t{}(static_cast<_Sndr&&>(__sndr), static_cast<_Fn&&>(__self.__fn_));
    }

    _Fn __fn_;
  };

public:
  template <class _Sndr, class _Fn>
  _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr, _Fn __fn) const;

  template <class _Fn>
  _CCCL_NODEBUG_API constexpr auto operator()(_Fn __fn) const;
};

struct then_t : __upon_t<then_t, set_value_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

struct upon_error_t : __upon_t<upon_error_t, set_error_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

struct upon_stopped_t : __upon_t<upon_stopped_t, set_stopped_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

template <class _UponTag, class _SetTag>
template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __upon_t<_UponTag, _SetTag>::__sndr_base_t
{
  using sender_concept = sender_t;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      if constexpr (__set_tag_t{} == execution::set_value)
      {
        return transform_completion_signatures(__child_completions, __transform_args_fn<_Fn>{});
      }
      else if constexpr (__set_tag_t{} == execution::set_error)
      {
        return transform_completion_signatures(__child_completions, {}, __transform_args_fn<_Fn>{});
      }
      else
      {
        return transform_completion_signatures(__child_completions, {}, {}, __transform_args_fn<_Fn>{});
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && //
    noexcept(__nothrow_constructible<__opstate_t<_Sndr, _Fn, _Rcvr>, _Sndr, _Rcvr, _Fn>) //
    -> __opstate_t<_Sndr, _Fn, _Rcvr>
  {
    return __opstate_t<_Sndr, _Fn, _Rcvr>{
      static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr), static_cast<_Fn&&>(__fn_)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& //
    noexcept(__nothrow_constructible<__opstate_t<_Sndr const&, _Fn, _Rcvr>,
                                     const _Sndr&,
                                     _Rcvr,
                                     const _Fn&>) //
    -> __opstate_t<_Sndr const&, _Fn, _Rcvr>
  {
    return __opstate_t<_Sndr const&, _Fn, _Rcvr>{__sndr_, static_cast<_Rcvr&&>(__rcvr), __fn_};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr>>
  {
    return __fwd_env(execution::get_env(__sndr_));
  }

  _CCCL_NO_UNIQUE_ADDRESS __upon_tag_t __tag_;
  _Fn __fn_;
  _Sndr __sndr_;
};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT then_t::__sndr_t : __upon_t<then_t, set_value_t>::__sndr_base_t<_Sndr, _Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT then_t::__closure_t : __upon_t<then_t, set_value_t>::__closure_base_t<_Fn>
{};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT upon_error_t::__sndr_t
    : __upon_t<upon_error_t, set_error_t>::__sndr_base_t<_Sndr, _Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT upon_error_t::__closure_t
    : __upon_t<upon_error_t, set_error_t>::__closure_base_t<_Fn>
{};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT upon_stopped_t::__sndr_t
    : __upon_t<upon_stopped_t, set_stopped_t>::__sndr_base_t<_Sndr, _Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT upon_stopped_t::__closure_t
    : __upon_t<upon_stopped_t, set_stopped_t>::__closure_base_t<_Fn>
{};

template <class _UponTag, class _SetTag>
template <class _Sndr, class _Fn>
_CCCL_NODEBUG_API constexpr auto __upon_t<_UponTag, _SetTag>::operator()(_Sndr __sndr, _Fn __fn) const
{
  using __sndr_t   = typename _UponTag::template __sndr_t<_Sndr, _Fn>;
  using __domain_t = __early_domain_of_t<_Sndr>;

  // If the incoming sender is non-dependent, we can check the completion
  // signatures of the composed sender immediately.
  if constexpr (!dependent_sender<_Sndr>)
  {
    __assert_valid_completion_signatures(get_completion_signatures<__sndr_t>());
  }

  return transform_sender(__domain_t{}, __sndr_t{{{}, static_cast<_Fn&&>(__fn), static_cast<_Sndr&&>(__sndr)}});
}

template <class _UponTag, class _SetTag>
template <class _Fn>
_CCCL_NODEBUG_API constexpr auto __upon_t<_UponTag, _SetTag>::operator()(_Fn __fn) const
{
  using __closure_t = typename _UponTag::template __closure_t<_Fn>;
  return __closure_t{{static_cast<_Fn&&>(__fn)}};
}

template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<then_t::__sndr_t<_Sndr, _Fn>> = 3;
template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<upon_error_t::__sndr_t<_Sndr, _Fn>> = 3;
template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<upon_stopped_t::__sndr_t<_Sndr, _Fn>> = 3;

_CCCL_GLOBAL_CONSTANT auto then         = then_t{};
_CCCL_GLOBAL_CONSTANT auto upon_error   = upon_error_t{};
_CCCL_GLOBAL_CONSTANT auto upon_stopped = upon_stopped_t{};

} // namespace cuda::experimental::execution

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_THEN
