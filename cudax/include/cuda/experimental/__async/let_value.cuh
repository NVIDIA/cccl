//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_LET_VALUE
#define __CUDAX_ASYNC_DETAIL_LET_VALUE

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
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/exception.cuh>
#include <cuda/experimental/__async/rcvr_ref.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/variant.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
// Declare types to use for diagnostics:
struct _FUNCTION_MUST_RETURN_A_SENDER;

// Forward-declate the let_* algorithm tag types:
struct let_value_t;
struct let_error_t;
struct let_stopped_t;

// Map from a disposition to the corresponding tag types:
namespace __detail
{
template <__disposition_t, class _Void = void>
extern __undefined<_Void> __let_tag;
template <class _Void>
extern __fn_t<let_value_t>* __let_tag<__value, _Void>;
template <class _Void>
extern __fn_t<let_error_t>* __let_tag<__error, _Void>;
template <class _Void>
extern __fn_t<let_stopped_t>* __let_tag<__stopped, _Void>;
} // namespace __detail

template <__disposition_t _Disposition>
struct __let
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC
  using _LetTag = decltype(__detail::__let_tag<_Disposition>());
  using _SetTag = decltype(__detail::__set_tag<_Disposition>());

  template <class...>
  using __empty_tuple = __tuple<>;

  /// @brief Computes the type of a variant of tuples to hold the results of
  /// the predecessor sender.
  template <class _CvSndr, class _Rcvr>
  using __results =
    __gather_completion_signatures<completion_signatures_of_t<_CvSndr, _Rcvr>,
                                   _SetTag,
                                   __decayed_tuple,
                                   __empty_tuple,
                                   __variant>;

  template <class _Fn, class _Rcvr>
  struct __opstate_fn
  {
    template <class... _As>
    using __call = connect_result_t<__call_result_t<_Fn, __decay_t<_As>&...>, __rcvr_ref_t<_Rcvr&>>;
  };

  /// @brief Computes the type of a variant of operation states to hold
  /// the second operation state.
  template <class _CvSndr, class _Fn, class _Rcvr>
  using __opstate2_t =
    __gather_completion_signatures<completion_signatures_of_t<_CvSndr, _Rcvr>,
                                   _SetTag,
                                   __opstate_fn<_Fn, _Rcvr>::template __call,
                                   __empty_tuple,
                                   __variant>;

  template <class _Fn, class _Rcvr>
  struct __completions_fn
  {
    using __error_non_sender_return = //
      _ERROR<_WHERE(_IN_ALGORITHM, _LetTag), _WHAT(_FUNCTION_MUST_RETURN_A_SENDER), _WITH_FUNCTION(_Fn)>;

    template <class _Ty>
    using __ensure_sender = //
      _CUDA_VSTD::conditional_t<__is_sender<_Ty> || __type_is_error<_Ty>, _Ty, __error_non_sender_return>;

    template <class... _As>
    using __error_not_callable_with = //
      _ERROR<_WHERE(_IN_ALGORITHM, _LetTag),
             _WHAT(_FUNCTION_IS_NOT_CALLABLE),
             _WITH_FUNCTION(_Fn),
             _WITH_ARGUMENTS(_As...)>;

    // This computes the result of calling the function with the
    // predecessor sender's results. If the function is not callable with
    // the results, it returns an _ERROR.
    template <class... _As>
    using __call_result = _CUDA_VSTD::
      __type_call<__type_try_quote<__call_result_t, __error_not_callable_with<_As...>>, _Fn, __decay_t<_As>&...>;

    // This computes the completion signatures of sender returned by the
    // function when called with the given arguments. It returns an _ERROR if
    // the function is not callable with the arguments or if the function
    // returns a non-sender.
    template <class... _As>
    using __call =
      __type_try_call_quote<completion_signatures_of_t, __ensure_sender<__call_result<_As...>>, __rcvr_ref_t<_Rcvr&>>;
  };

  /// @brief Computes the completion signatures of the
  /// `let_(value|error|stopped)` sender.
  template <class _CvSndr, class _Fn, class _Rcvr>
  using __completions = __gather_completion_signatures<
    completion_signatures_of_t<_CvSndr, _Rcvr>,
    _SetTag,
    __completions_fn<_Fn, _Rcvr>::template __call,
    __default_completions,
    _CUDA_VSTD::__type_bind_front<__type_try_quote<__concat_completion_signatures>, __eptr_completion>::__call>;

  /// @brief The `let_(value|error|stopped)` operation state.
  /// @tparam _CvSndr The cvref-qualified predecessor sender type.
  /// @tparam _Fn The function to be called when the predecessor sender
  /// completes.
  /// @tparam _Rcvr The receiver connected to the `let_(value|error|stopped)`
  /// sender.
  template <class _Rcvr, class _CvSndr, class _Fn>
  struct __opstate_t
  {
    _CUDAX_API friend env_of_t<_Rcvr> get_env(const __opstate_t* __self) noexcept
    {
      return __async::get_env(__self->__rcvr_);
    }

    using operation_state_concept = operation_state_t;
    using completion_signatures   = __completions<_CvSndr, _Fn, _Rcvr>;

    // Don't try to compute the type of the variant of operation states
    // if the computation of the completion signatures failed.
    using __deferred_opstate_fn = _CUDA_VSTD::__type_bind_back<__type_try_quote<__opstate2_t>, _CvSndr, _Fn, _Rcvr>;
    using __opstate_variant_fn  = _CUDA_VSTD::
      conditional_t<__type_is_error<completion_signatures>, _CUDA_VSTD::__type_always<__empty>, __deferred_opstate_fn>;
    using __opstate_variant_t = __type_try_call<__opstate_variant_fn>;

    _Rcvr __rcvr_;
    _Fn __fn_;
    __results<_CvSndr, __opstate_t*> __result_;
    connect_result_t<_CvSndr, __opstate_t*> __opstate1_;
    __opstate_variant_t __opstate2_;

    _CUDAX_API __opstate_t(_CvSndr&& __sndr, _Fn __fn, _Rcvr __rcvr) noexcept(
      __nothrow_decay_copyable<_Fn, _Rcvr> && __nothrow_connectable<_CvSndr, __opstate_t*>)
        : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
        , __fn_(static_cast<_Fn&&>(__fn))
        , __opstate1_(__async::connect(static_cast<_CvSndr&&>(__sndr), this))
    {}

    _CUDAX_API void start() noexcept
    {
      __async::start(__opstate1_);
    }

    template <class _Tag, class... _As>
    _CUDAX_API void __complete(_Tag, _As&&... __as) noexcept
    {
      if constexpr (_CUDA_VSTD::is_same_v<_Tag, _SetTag>)
      {
        _CUDAX_TRY( //
          ({ //
            // Store the results so the lvalue refs we pass to the function
            // will be valid for the duration of the async op.
            auto& __tupl = __result_.template __emplace<__decayed_tuple<_As...>>(static_cast<_As&&>(__as)...);
            if constexpr (!__type_is_error<completion_signatures>)
            {
              // Call the function with the results and connect the resulting
              // sender, storing the operation state in __opstate2_.
              auto& __nextop = __opstate2_.__emplace_from(
                __async::connect, __tupl.__apply(static_cast<_Fn&&>(__fn_), __tupl), __async::__rcvr_ref(__rcvr_));
              __async::start(__nextop);
            }
          }),
          _CUDAX_CATCH(...)( //
            { //
              __async::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
            }))
      }
      else
      {
        // Forward the completion to the receiver unchanged.
        _Tag()(static_cast<_Rcvr&&>(__rcvr_), static_cast<_As&&>(__as)...);
      }
    }

    template <class... _As>
    _CUDAX_TRIVIAL_API void set_value(_As&&... __as) noexcept
    {
      __complete(set_value_t(), static_cast<_As&&>(__as)...);
    }

    template <class _Error>
    _CUDAX_TRIVIAL_API void set_error(_Error&& __error) noexcept
    {
      __complete(set_error_t(), static_cast<_Error&&>(__error));
    }

    _CUDAX_TRIVIAL_API void set_stopped() noexcept
    {
      __complete(set_stopped_t());
    }
  };

  /// @brief The `let_(value|error|stopped)` sender.
  /// @tparam _Sndr The predecessor sender.
  /// @tparam _Fn The function to be called when the predecessor sender
  /// completes.
  template <class _Sndr, class _Fn>
  struct __sndr_t
  {
    using sender_concept = sender_t;
    _CCCL_NO_UNIQUE_ADDRESS _LetTag __tag_;
    _Fn __fn_;
    _Sndr __sndr_;

    template <class _Rcvr>
    _CUDAX_API auto connect(_Rcvr __rcvr) && noexcept(
      __nothrow_constructible<__opstate_t<_Rcvr, _Sndr, _Fn>, _Sndr, _Fn, _Rcvr>) -> __opstate_t<_Rcvr, _Sndr, _Fn>
    {
      return __opstate_t<_Rcvr, _Sndr, _Fn>(
        static_cast<_Sndr&&>(__sndr_), static_cast<_Fn&&>(__fn_), static_cast<_Rcvr&&>(__rcvr));
    }

    template <class _Rcvr>
    _CUDAX_API auto connect(_Rcvr __rcvr) const& noexcept( //
      __nothrow_constructible<__opstate_t<_Rcvr, const _Sndr&, _Fn>,
                              const _Sndr&,
                              const _Fn&,
                              _Rcvr>) //
      -> __opstate_t<_Rcvr, const _Sndr&, _Fn>
    {
      return __opstate_t<_Rcvr, const _Sndr&, _Fn>(__sndr_, __fn_, static_cast<_Rcvr&&>(__rcvr));
    }

    _CUDAX_API env_of_t<_Sndr> get_env() const noexcept
    {
      return __async::get_env(__sndr_);
    }
  };

  template <class _Fn>
  struct __closure_t
  {
    using _LetTag = decltype(__detail::__let_tag<_Disposition>());
    _Fn __fn_;

    template <class _Sndr>
    _CUDAX_TRIVIAL_API auto operator()(_Sndr __sndr) const //
      -> __call_result_t<_LetTag, _Sndr, _Fn>
    {
      return _LetTag()(static_cast<_Sndr&&>(__sndr), __fn_);
    }

    template <class _Sndr>
    _CUDAX_TRIVIAL_API friend auto operator|(_Sndr __sndr, const __closure_t& __self) //
      -> __call_result_t<_LetTag, _Sndr, _Fn>
    {
      return _LetTag()(static_cast<_Sndr&&>(__sndr), __self.__fn_);
    }
  };

public:
  template <class _Sndr, class _Fn>
  _CUDAX_API __sndr_t<_Sndr, _Fn> operator()(_Sndr __sndr, _Fn __fn) const
  {
    // If the incoming sender is non-dependent, we can check the completion
    // signatures of the composed sender immediately.
    if constexpr (__is_non_dependent_sender<_Sndr>)
    {
      using __completions = completion_signatures_of_t<__sndr_t<_Sndr, _Fn>>;
      static_assert(__is_completion_signatures<__completions>);
    }
    return __sndr_t<_Sndr, _Fn>{{}, static_cast<_Fn&&>(__fn), static_cast<_Sndr&&>(__sndr)};
  }

  template <class _Fn>
  _CUDAX_TRIVIAL_API auto operator()(_Fn __fn) const noexcept
  {
    return __closure_t<_Fn>{static_cast<_Fn&&>(__fn)};
  }
};

_CCCL_GLOBAL_CONSTANT struct let_value_t : __let<__value>
{
} let_value{};

_CCCL_GLOBAL_CONSTANT struct let_error_t : __let<__error>
{
} let_error{};

_CCCL_GLOBAL_CONSTANT struct let_stopped_t : __let<__stopped>
{
} let_stopped{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
