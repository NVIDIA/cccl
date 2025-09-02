//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_BULK
#define __CUDAX_EXECUTION_BULK

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__utility/immovable.h>
#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__launch/configuration.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // warning: unreachable code

namespace cuda::experimental::execution
{
namespace __bulk
{
template <class _Shape, class _Fn, class _Rcvr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
{
  _Rcvr __rcvr_;
  _Shape __shape_;
  _Fn __fn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// attributes for bulk senders
template <class _Sndr, class _Shape>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
{
  [[nodiscard]] _CCCL_HOST_API constexpr auto query(get_launch_config_t) const noexcept
  {
    constexpr int __block_threads = 256;
    const int __grid_blocks       = ::cuda::ceil_div(static_cast<int>(__shape_), __block_threads);
    return experimental::make_config(block_dims<__block_threads>(), grid_dims(__grid_blocks));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Query, class... _Args)
  _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query, _Args...>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Query, _Args&&... __args) const
    noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query, _Args...>)
      -> __query_result_t<env_of_t<_Sndr>, _Query, _Args...>
  {
    return execution::get_env(__sndr_).query(_Query{}, static_cast<_Args&&>(__args)...);
  }

  _Shape __shape_;
  const _Sndr& __sndr_;
};
} // namespace __bulk

////////////////////////////////////////////////////////////////////////////////////////////////////
// generic bulk utilities
template <class _BulkTag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __bulk_t
{
  // This is a function object that is used to transform the value completion signatures
  // of a bulk sender's child operation. It does type checking and "throws" if the bulk
  // function is not callable with the value datums of the predecessor.
  template <class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __transform_value_completion_fn
  {
    template <class... _Ts>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const
    {
      // The function objects passed to the "chunked" and "unchunked" flavors of bulk have
      // different signatures, so we need to type-check them separately.
      if constexpr (_BulkTag::__is_chunked())
      {
        if constexpr (__callable<_Fn&, _Shape, _Shape, _Ts&...>)
        {
          return completion_signatures<set_value_t(_Ts...)>{}
               + __eptr_completion_if<!__nothrow_callable<_Fn&, _Shape, _Shape, _Ts&...>>();
        }
        else
        {
          return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _BulkTag),
                                              _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                              _WITH_FUNCTION(_Fn&),
                                              _WITH_ARGUMENTS(_Shape, _Shape, _Ts & ...)>();
        }
      }
      else if constexpr (__callable<_Fn&, _Shape, _Ts&...>)
      {
        return completion_signatures<set_value_t(_Ts...)>{}
             + __eptr_completion_if<!__nothrow_callable<_Fn&, _Shape, _Ts&...>>();
      }
      else
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _BulkTag),
                                            _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                            _WITH_FUNCTION(_Fn&),
                                            _WITH_ARGUMENTS(_Shape, _Ts & ...)>();
      }
    }
  };

  template <class _Shape, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_base_t
  {
    using receiver_concept = receiver_t;

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__err));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
    }

    [[nodiscard]] _CCCL_NODEBUG_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __bulk::__state_t<_Shape, _Fn, _Rcvr>* __state_;
  };

  // This is the operation state for bulk senders. It connects the child sender with
  // a receiver defined by _BulkTag.
  template <class _CvSndr, class _Shape, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __rcvr_t                = typename _BulkTag::template __rcvr_t<_Shape, _Fn, _Rcvr>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __state_{static_cast<_Rcvr&&>(__rcvr), __shape, static_cast<_Fn&&>(__fn)}
        , __opstate_{execution::connect(static_cast<_CvSndr&&>(__sndr), __rcvr_t{{&__state_}})}
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate_);
    }

    __bulk::__state_t<_Shape, _Fn, _Rcvr> __state_;
    connect_result_t<_CvSndr, __rcvr_t> __opstate_;
  };

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_base_t
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API friend constexpr auto operator|(_Sndr&& __sndr, __closure_base_t __self)
    {
      static_assert(__is_sender<_Sndr>);

      using __domain_t = __early_domain_of_t<_Sndr>;
      using __sndr_t   = typename _BulkTag::template __sndr_t<_Sndr, _Policy, _Shape, _Fn>;

      if constexpr (!dependent_sender<_Sndr>)
      {
        __assert_valid_completion_signatures(get_completion_signatures<__sndr_t>());
      }

      return transform_sender(__domain_t{},
                              __sndr_t{{{}, static_cast<__closure_base_t&&>(__self), static_cast<_Sndr&&>(__sndr)}});
    }

    _CCCL_NO_UNIQUE_ADDRESS _Policy __policy_;
    _Shape __shape_;
    _Fn __fn_;
  };

  // This is the sender type for the three bulk algorithms.
  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_base_t
  {
    using sender_concept = sender_t;

    template <class _Self, class... _Env>
    [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__child_completions) = execution::get_child_completion_signatures<_Self, _Sndr, _Env...>())
      {
        return transform_completion_signatures(__child_completions, __transform_value_completion_fn<_Shape, _Fn>{});
      }
    }

    // The bulk algorithm lowers to a bulk_chunked sender. The bulk sender itself should
    // not have `connect` functions, since they should never be called. Hence, we
    // constrain these functions with !same_as<_BulkTag, bulk_t>.
    _CCCL_TEMPLATE(class _Rcvr)
    _CCCL_REQUIRES((!::cuda::std::same_as<_BulkTag, bulk_t>) )
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Shape, _Fn, _Rcvr>
    {
      return __opstate_t<_Sndr, _Shape, _Fn, _Rcvr>{
        static_cast<_Sndr&&>(__sndr_),
        static_cast<_Rcvr&&>(__rcvr),
        __state_.__shape_,
        static_cast<_Fn&&>(__state_.__fn_)};
    }

    _CCCL_TEMPLATE(class _Rcvr)
    _CCCL_REQUIRES((!::cuda::std::same_as<_BulkTag, bulk_t>) )
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Shape, _Fn, _Rcvr>
    {
      return __opstate_t<const _Sndr&, _Shape, _Fn, _Rcvr>{
        __sndr_, static_cast<_Rcvr&&>(__rcvr), __state_.__shape_, __state_.__fn_};
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __bulk::__attrs_t<_Sndr, _Shape>
    {
      return {__state_.__shape_, __sndr_};
    }

    _CCCL_NO_UNIQUE_ADDRESS _BulkTag __tag_;
    __closure_base_t<_Policy, _Shape, _Fn> __state_;
    _Sndr __sndr_;
  };

  // This function call operator is the entry point for the bulk algorithms. It takes a
  // predecessor sender, a policy, a shape, and a function, and returns a sender that can
  // be connected to a receiver.
  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr&& __sndr, _Policy __policy, _Shape __shape, _Fn __fn) const
  {
    return (static_cast<_Sndr&&>(__sndr) | (*this)(__policy, __shape, static_cast<_Fn&&>(__fn)));
  }

  // This function call operator creates a sender adaptor closure object that can appear
  // on the right-hand side of a pipe operator, like: sndr | bulk(par, shape, fn).
  template <class _Policy, class _Shape, class _Fn>
  [[nodiscard]] _CCCL_NODEBUG_API auto operator()(_Policy __policy, _Shape __shape, _Fn __fn) const
  {
    static_assert(::cuda::std::integral<_Shape>);
    static_assert(::cuda::std::is_execution_policy_v<_Policy>);
    using __closure_t = typename _BulkTag::template __closure_t<_Policy, _Shape, _Fn>;
    return __closure_t{{__policy, __shape, static_cast<_Fn&&>(__fn)}};
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk_chunked
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_chunked_t : __bulk_t<bulk_chunked_t>
{
  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __bulk_t::__sndr_base_t<_Sndr, _Policy, _Shape, _Fn>
  {};

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t : __bulk_t::__closure_base_t<_Policy, _Shape, _Fn>
  {};

  // This is the receiver for the bulk_chunked sender. It provides the implementation for
  // `set_value` that calls the function with the begin and end shapes, and the value
  // results of the predecessor.
  template <class _Shape, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t : __bulk_t::__rcvr_base_t<_Shape, _Fn, _Rcvr>
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Values>
    _CCCL_API void set_value(_Values&&... __values) noexcept
    {
      _CCCL_TRY //
      {
        this->__state_->__fn_(_Shape(0), _Shape(this->__state_->__shape_), __values...);
        execution::set_value(static_cast<_Rcvr&&>(this->__state_->__rcvr_), static_cast<_Values&&>(__values)...);
      }
      _CCCL_CATCH_ALL //
      {
        if constexpr (!__nothrow_callable<_Fn&, _Shape, _Shape, _Values&...>)
        {
          execution::set_error(static_cast<_Rcvr&&>(this->__state_->__rcvr_), ::std::current_exception());
        }
      }
    }
  };

  [[nodiscard]] _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk_chunked = bulk_chunked_t{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk_unchunked
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_unchunked_t : __bulk_t<bulk_unchunked_t>
{
  // This is the receiver for the bulk_unchunked sender. It provides the implementation
  // for `set_value` that calls the function repeatedly with an index and the value
  // results of the predecessor. The index is monotonically increasing from 0 to the shape
  // minus one.
  template <class _Shape, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t : __bulk_t::__rcvr_base_t<_Shape, _Fn, _Rcvr>
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Values>
    _CCCL_API void set_value(_Values&&... __values) noexcept
    {
      _CCCL_TRY //
      {
        for (_Shape __index{}; __index != this->__state_->__shape_; ++__index)
        {
          this->__state_->__fn_(_Shape(__index), __values...);
        }
        execution::set_value(static_cast<_Rcvr&&>(this->__state_->__rcvr_), static_cast<_Values&&>(__values)...);
      }
      _CCCL_CATCH_ALL //
      {
        if constexpr (!__nothrow_callable<_Fn&, _Shape, _Values&...>)
        {
          execution::set_error(static_cast<_Rcvr&&>(this->__state_->__rcvr_), ::std::current_exception());
        }
      }
    }
  };

  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __bulk_t::__sndr_base_t<_Sndr, _Policy, _Shape, _Fn>
  {};

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t : __bulk_t::__closure_base_t<_Policy, _Shape, _Fn>
  {};

  [[nodiscard]] _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return false;
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk_unchunked = bulk_unchunked_t{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_t : __bulk_t<bulk_t>
{
  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __bulk_t::__sndr_base_t<_Sndr, _Policy, _Shape, _Fn>
  {};

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t : __bulk_t::__closure_base_t<_Policy, _Shape, _Fn>
  {};

  // This is a function adaptor that transforms a `bulk` function that takes a single
  // shape to a `bulk_chunked` function that takes a begin and end shape.
  template <class _Shape, class _Fn>
  struct __bulk_chunked_fn
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Ts>
    _CCCL_NODEBUG_API void operator()(_Shape __begin, _Shape __end, _Ts&&... __values) noexcept(
      __nothrow_callable<_Fn&, _Shape, decltype(__values)&...>)
    {
      for (; __begin != __end; ++__begin)
      {
        // Pass a copy of `__begin` to the function so it can't do anything funny with it.
        __fn_(_Shape(__begin), __values...);
      }
    }

    _Fn __fn_;
  };

  // This function is called when `connect` is called on a `bulk` sender. It transforms
  // the `bulk` sender into a `bulk_chunked` sender.
  template <class _Sndr>
  [[nodiscard]] _CCCL_API static auto transform_sender(_Sndr&& __sndr, ::cuda::std::__ignore_t)
  {
    static_assert(__same_as<tag_of_t<_Sndr>, bulk_t>);
    auto& [__tag, __data, __child]  = __sndr;
    auto& [__policy, __shape, __fn] = __data;

    using __chunked_fn_t = __bulk_chunked_fn<decltype(__shape), decltype(__fn)>;

    // Lower `bulk` to `bulk_chunked`. If `bulk_chunked` has a late customization, we will
    // see the customization.
    return bulk_chunked(::cuda::std::forward_like<_Sndr>(__child),
                        __policy,
                        __shape,
                        __chunked_fn_t{::cuda::std::forward_like<_Sndr>(__fn)});
  }

  [[nodiscard]] _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return false;
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk = bulk_t{};

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<bulk_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<bulk_chunked_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<bulk_unchunked_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

} // namespace cuda::experimental::execution

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_BULK
