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

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/pod_tuple.h>

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

namespace cuda::experimental::execution
{
// Map from a disposition to the corresponding tag types:
namespace __detail
{
template <__disposition_t, class _Void = void>
extern _CUDA_VSTD::__undefined<_Void> __upon_tag;
template <class _Void>
extern __fn_t<then_t>* __upon_tag<__value, _Void>;
template <class _Void>
extern __fn_t<upon_error_t>* __upon_tag<__error, _Void>;
template <class _Void>
extern __fn_t<upon_stopped_t>* __upon_tag<__stopped, _Void>;
} // namespace __detail

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
  _CUDA_VSTD::__type_call1<__completion_fn<_CUDA_VSTD::is_same_v<_Result, void>, _Nothrow>, _Result>;

template <class _Fn, class... _Ts>
using __completion _CCCL_NODEBUG_ALIAS =
  __completion_<_CUDA_VSTD::__call_result_t<_Fn, _Ts...>, __nothrow_callable<_Fn, _Ts...>>;
} // namespace __upon

template <__disposition_t _Disposition>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(then_t) _CCCL_PREFERRED_NAME(upon_error_t)
  _CCCL_PREFERRED_NAME(upon_stopped_t) __upon_t
{
private:
  using _UponTag _CCCL_NODEBUG_ALIAS = decltype(__detail::__upon_tag<_Disposition>());
  using _SetTag _CCCL_NODEBUG_ALIAS  = decltype(__detail::__set_tag<_Disposition>());

  template <class _Rcvr, class _CvSndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;
    using __env_t _CCCL_NODEBUG_ALIAS                 = env_of_t<_Rcvr>;
    using __rcvr_t _CCCL_NODEBUG_ALIAS                = __rcvr_ref_t<__opstate_t, __env_t>;

    _CCCL_API __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Fn __fn)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __fn_{static_cast<_Fn&&>(__fn)}
        , __opstate_{execution::connect(static_cast<_CvSndr&&>(__sndr), __ref_rcvr(*this))}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API void start() & noexcept
    {
      execution::start(__opstate_);
    }

    _CCCL_EXEC_CHECK_DISABLE
    template <bool _CanThrow = false, class... _Ts>
    _CCCL_API void __set(_Ts&&... __ts) noexcept(!_CanThrow)
    {
      if constexpr (_CanThrow || __nothrow_callable<_Fn, _Ts...>)
      {
        if constexpr (_CUDA_VSTD::is_same_v<void, _CUDA_VSTD::__call_result_t<_Fn, _Ts...>>)
        {
          static_cast<_Fn&&>(__fn_)(static_cast<_Ts&&>(__ts)...);
          execution::set_value(static_cast<_Rcvr&&>(__rcvr_));
        }
        else
        {
          execution::set_value(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Fn&&>(__fn_)(static_cast<_Ts&&>(__ts)...));
        }
      }
      else
      {
        _CUDAX_TRY( //
          ({ //
            __set<true>(static_cast<_Ts&&>(__ts)...); //
          }), //
          _CUDAX_CATCH(...) //
          ({ //
            execution::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
          }) //
        )
      }
    }

    template <class _Tag, class... _Ts>
    _CCCL_TRIVIAL_API void __complete(_Tag, _Ts&&... __ts) noexcept
    {
      if constexpr (_CUDA_VSTD::is_same_v<_Tag, _SetTag>)
      {
        __set(static_cast<_Ts&&>(__ts)...);
      }
      else
      {
        _Tag()(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Ts&&>(__ts)...);
      }
    }

    template <class... _Ts>
    _CCCL_API void set_value(_Ts&&... __ts) noexcept
    {
      __complete(set_value_t(), static_cast<_Ts&&>(__ts)...);
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __error) noexcept
    {
      __complete(set_error_t(), static_cast<_Error&&>(__error));
    }

    _CCCL_API void set_stopped() noexcept
    {
      __complete(set_stopped_t());
    }

    _CCCL_API auto get_env() const noexcept -> __env_t
    {
      return execution::get_env(__rcvr_);
    }

    _Rcvr __rcvr_;
    _Fn __fn_;
    connect_result_t<_CvSndr, __rcvr_t> __opstate_;
  };

  template <class _Fn>
  struct __transform_args_fn
  {
    template <class... _Ts>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const
    {
      if constexpr (_CUDA_VSTD::__is_callable_v<_Fn, _Ts...>)
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

public:
  template <class _Fn, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;

  template <class _Sndr, class _Fn>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr __sndr, _Fn __fn) const;

  template <class _Fn>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Fn __fn) const;
};

template <__disposition_t _Disposition>
template <class _Fn, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __upon_t<_Disposition>::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS _UponTag __tag_;
  _Fn __fn_;
  _Sndr __sndr_;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      if constexpr (_Disposition == __disposition_t::__value)
      {
        return transform_completion_signatures(__child_completions, __transform_args_fn<_Fn>{});
      }
      else if constexpr (_Disposition == __disposition_t::__error)
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
  _CCCL_API auto connect(_Rcvr __rcvr) && //
    noexcept(__nothrow_constructible<__opstate_t<_Rcvr, _Sndr, _Fn>, _Sndr, _Rcvr, _Fn>) //
    -> __opstate_t<_Rcvr, _Sndr, _Fn>
  {
    return __opstate_t<_Rcvr, _Sndr, _Fn>{
      static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr), static_cast<_Fn&&>(__fn_)};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) const& //
    noexcept(__nothrow_constructible<__opstate_t<_Rcvr, const _Sndr&, _Fn>,
                                     const _Sndr&,
                                     _Rcvr,
                                     const _Fn&>) //
    -> __opstate_t<_Rcvr, const _Sndr&, _Fn>
  {
    return __opstate_t<_Rcvr, const _Sndr&, _Fn>{__sndr_, static_cast<_Rcvr&&>(__rcvr), __fn_};
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr>>
  {
    return __fwd_env(execution::get_env(__sndr_));
  }
};

template <__disposition_t _Disposition>
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __upon_t<_Disposition>::__closure_t
{
  using _UponTag _CCCL_NODEBUG_ALIAS = decltype(__detail::__upon_tag<_Disposition>());
  _Fn __fn_;

  template <class _Sndr>
  _CCCL_TRIVIAL_API auto operator()(_Sndr __sndr) -> _CUDA_VSTD::__call_result_t<_UponTag, _Sndr, _Fn>
  {
    return _UponTag()(static_cast<_Sndr&&>(__sndr), static_cast<_Fn&&>(__fn_));
  }

  template <class _Sndr>
  _CCCL_TRIVIAL_API friend auto operator|(_Sndr __sndr, __closure_t&& __self) //
    -> _CUDA_VSTD::__call_result_t<_UponTag, _Sndr, _Fn>
  {
    return _UponTag()(static_cast<_Sndr&&>(__sndr), static_cast<_Fn&&>(__self.__fn_));
  }
};

template <__disposition_t _Disposition>
template <class _Sndr, class _Fn>
_CCCL_TRIVIAL_API constexpr auto __upon_t<_Disposition>::operator()(_Sndr __sndr, _Fn __fn) const
{
  using __dom_t _CCCL_NODEBUG_ALIAS = __early_domain_of_t<_Sndr>;
  // If the incoming sender is non-dependent, we can check the completion
  // signatures of the composed sender immediately.
  if constexpr (!dependent_sender<_Sndr>)
  {
    __assert_valid_completion_signatures(get_completion_signatures<__sndr_t<_Fn, _Sndr>>());
  }
  return transform_sender(__dom_t{}, __sndr_t<_Fn, _Sndr>{{}, static_cast<_Fn&&>(__fn), static_cast<_Sndr&&>(__sndr)});
}

template <__disposition_t _Disposition>
template <class _Fn>
_CCCL_TRIVIAL_API constexpr auto __upon_t<_Disposition>::operator()(_Fn __fn) const
{
  return __closure_t<_Fn>{static_cast<_Fn&&>(__fn)};
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

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_THEN
