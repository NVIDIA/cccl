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

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/domain.cuh>
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
template <class _Policy, class _Shape, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __bulk_state_t
{
  _CCCL_NO_UNIQUE_ADDRESS _Policy __policy_;
  _Shape __shape_;
  _Fn __fn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// attributes for bulk senders
template <class _Sndr, class _Shape>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __bulk_attrs_t
{
  _CCCL_HOST_API constexpr auto query(get_launch_config_t) const noexcept
  {
    constexpr int __block_threads = 256;
    const int __grid_blocks       = (static_cast<int>(__shape_) + __block_threads - 1) / __block_threads;
    return experimental::make_config(block_dims<__block_threads>, grid_dims(__grid_blocks));
  }

  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query>)
  _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>)
    -> decltype(auto)
  {
    return execution::get_env(__sndr_).query(_Query{});
  }

  _Shape __shape_;
  const _Sndr& __sndr_;
};

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
        if constexpr (_CUDA_VSTD::__is_callable_v<_Fn&, _Shape, _Shape, _Ts&...>)
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
      else if constexpr (_CUDA_VSTD::__is_callable_v<_Fn&, _Shape, _Ts&...>)
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

  // This is the base operation state for bulk senders. It provides the common
  // functionality for all bulk senders, such as starting the operation, setting errors,
  // and getting the environment. The bulk operation state types inherit from this and
  // provide the implementation for `set_value`.
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __derived_opstate_t     = typename _BulkTag::template __opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;
    static_assert(!_CUDA_VSTD::is_same_v<__derived_opstate_t, __opstate_t>,
                  "The derived operation state must not be the same as the base operation state");
    using __rcvr_t = __rcvr_ref_t<__derived_opstate_t, __fwd_env_t<env_of_t<_Rcvr>>>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __shape_{__shape}
        , __fn_{static_cast<_Fn&&>(__fn)}
        , __opstate_{
            execution::connect(static_cast<_CvSndr&&>(__sndr), __ref_rcvr(*static_cast<__derived_opstate_t*>(this)))}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate_);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Error&&>(__err));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__rcvr_));
    }

    _Rcvr __rcvr_;
    _Shape __shape_;
    _Fn __fn_;
    connect_result_t<_CvSndr, __rcvr_t> __opstate_;
  };

  // This is the sender type for the three bulk algorithms.
  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    // Look in _BulkTag (bulk_t, bulk_chunked_t, or bulk_unchunked_t) for the derived
    // operation state type.
    template <class _CvSndr, class _Rcvr>
    using __opstate_t = typename _BulkTag::template __opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

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
    // not have `connect` functions, shince they should never be called. Hence, we
    // constrain these functions with !same_as<_BulkTag, bulk_t>.
    _CCCL_TEMPLATE(class _Rcvr)
    _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_BulkTag, bulk_t>) )
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Rcvr>
    {
      return __opstate_t<_Sndr, _Rcvr>{
        static_cast<_Sndr&&>(__sndr_),
        static_cast<_Rcvr&&>(__rcvr),
        __state_.__shape_,
        static_cast<_Fn&&>(__state_.__fn_)};
    }

    _CCCL_TEMPLATE(class _Rcvr)
    _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_BulkTag, bulk_t>) )
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Rcvr>
    {
      return __opstate_t<const _Sndr&, _Rcvr>{__sndr_, static_cast<_Rcvr&&>(__rcvr), __state_.__shape_, __state_.__fn_};
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __bulk_attrs_t<_Sndr, _Shape>
    {
      return {__state_.__shape_, __sndr_};
    }

    _CCCL_NO_UNIQUE_ADDRESS _BulkTag __tag_;
    __bulk_state_t<_Policy, _Shape, _Fn> __state_;
    _Sndr __sndr_;
  };

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t : __bulk_state_t<_Policy, _Shape, _Fn>
  {
    template <class _Sndr>
    _CCCL_TRIVIAL_API friend constexpr auto operator|(_Sndr&& __sndr, __closure_t __self)
    {
      using __domain_t = __early_domain_of_t<_Sndr>;
      using __sndr_t   = __bulk_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>;
      return transform_sender(__domain_t{},
                              __sndr_t{{}, static_cast<__closure_t&&>(__self), static_cast<_Sndr&&>(__sndr)});
    }
  };

  // This function call operator is the entry point for the bulk algorithms. It takes a
  // predecessor sender, a policy, a shape, and a function, and returns a sender that can
  // be connected to a receiver.
  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  _CCCL_API auto operator()(_Sndr&& __sndr, _Policy __policy, _Shape __shape, _Fn __fn) const
  {
    static_assert(__is_sender<_Sndr>);
    static_assert(_CUDA_VSTD::integral<_Shape>);
    static_assert(is_execution_policy_v<_Policy>);

    using __domain_t = __early_domain_of_t<_Sndr>;
    using __sndr_t   = __bulk_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>;

    if constexpr (!dependent_sender<_Sndr>)
    {
      __assert_valid_completion_signatures(get_completion_signatures<__sndr_t>());
    }

    return transform_sender(__domain_t{},
                            __sndr_t{{}, {__policy, __shape, static_cast<_Fn&&>(__fn)}, static_cast<_Sndr&&>(__sndr)});
  }

  // This function call operator creates a sender adaptor closure object that can appear
  // on the right-hand side of a pipe operator, like: sndr | bulk(par, shape, fn).
  template <class _Policy, class _Shape, class _Fn>
  _CCCL_TRIVIAL_API auto operator()(_Policy __policy, _Shape __shape, _Fn __fn) const
    -> __closure_t<_Policy, _Shape, _Fn>
  {
    static_assert(_CUDA_VSTD::integral<_Shape>);
    static_assert(is_execution_policy_v<_Policy>);
    return {__policy, __shape, static_cast<_Fn&&>(__fn)};
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk_chunked
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_chunked_t : __bulk_t<bulk_chunked_t>
{
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  using __base_opstate_t = __bulk_t::__opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

  // This is the operation state for the bulk_chunked sender. It provides the
  // implementation for `set_value` that calls the function with the begin and end shapes,
  // and the value results of the predecessor.
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>
  {
    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>{
            static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __shape, static_cast<_Fn&&>(__fn)}
    {}

    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Values>
    _CCCL_API void set_value(_Values&&... __values) noexcept
    {
      _CUDAX_TRY( //
        ({
          this->__fn_(_Shape(0), _Shape(this->__shape_), __values...);
          execution::set_value(static_cast<_Rcvr&&>(this->__rcvr_), static_cast<_Values&&>(__values)...);
        }),
        _CUDAX_CATCH(...) //
        ({
          if constexpr (!__nothrow_callable<_Fn&, _Shape, _Shape, _Values&...>)
          {
            execution::set_error(static_cast<_Rcvr&&>(this->__rcvr_), ::std::current_exception());
          }
        }))
    }
  };

  _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk_chunked = bulk_chunked_t{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk_unchunked
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_unchunked_t : __bulk_t<bulk_unchunked_t>
{
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  using __base_opstate_t = __bulk_t::__opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

  // This is the operation state for the bulk_unchunked sender. It provides the
  // implementation for `set_value` that calls the function repeatedly with an index and
  // the value results of the predecessor. The index is monotonically increasing from 0 to
  // the shape minus one.
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>
  {
    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>{
            static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __shape, static_cast<_Fn&&>(__fn)}
    {
      __check_forward_progress(__sndr);
    }

    _CCCL_API static constexpr void __check_forward_progress(_CvSndr& __sndr)
    {
      if constexpr (_CUDA_VSTD::__is_callable_v<get_completion_scheduler_t<set_value_t>, env_of_t<_CvSndr>>)
      {
        // If the scheduler is queryable, we can check the forward progress guarantee to
        // make sure the user hasn't asked the default implementation to do something that
        // it cannot do.
        using __sched_t = _CUDA_VSTD::__call_result_t<get_completion_scheduler_t<set_value_t>, env_of_t<_CvSndr>>;
        if constexpr (__statically_queryable_with<__sched_t, get_forward_progress_guarantee_t>)
        {
          constexpr auto __guarantee = __sched_t::query(get_forward_progress_guarantee);
          static_assert(__guarantee != forward_progress_guarantee::concurrent,
                        "The default implementation cannot provide the concurrent progress guarantee");
        }
        else
        {
          const auto __guarantee =
            get_forward_progress_guarantee(get_completion_scheduler<set_value_t>(execution::get_env(__sndr)));
          _CCCL_ASSERT(__guarantee != forward_progress_guarantee::concurrent,
                       "The default implementation cannot provide the concurrent progress guarantee");
        }
      }
    }

    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Values>
    _CCCL_API void set_value(_Values&&... __values) noexcept
    {
      _CUDAX_TRY( //
        ({
          for (_Shape __index{}; __index != this->__shape_; ++__index)
          {
            this->__fn_(_Shape(__index), __values...);
          }
          execution::set_value(static_cast<_Rcvr&&>(this->__rcvr_), static_cast<_Values&&>(__values)...);
        }),
        _CUDAX_CATCH(...) //
        ({
          if constexpr (!__nothrow_callable<_Fn&, _Shape, _Values&...>)
          {
            execution::set_error(static_cast<_Rcvr&&>(this->__rcvr_), ::std::current_exception());
          }
        }))
    }
  };

  // The bulk_unchunked algorithm is different from the other bulk algorithms in that it
  // does not take an execution policy, so we must hide the base class's `operator()` that
  // takes a policy with an overload that does not take a policy.
  template <class _Sndr, class _Shape, class _Fn>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr&& __sndr, _Shape __shape, _Fn __fn) const
  {
    return this->__bulk_t::operator()(static_cast<_Sndr&&>(__sndr), par, __shape, static_cast<_Fn&&>(__fn));
  }

  template <class _Shape, class _Fn>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Shape __shape, _Fn __fn) const
  {
    return this->__bulk_t::operator()(par, __shape, static_cast<_Fn&&>(__fn));
  }

  _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return false;
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk_unchunked = bulk_unchunked_t{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bulk
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_t : __bulk_t<bulk_t>
{
  // This is a function adaptor that transforms a `bulk` function that takes a single
  // shape to a `bulk_chunked` function that takes a begin and end shape.
  template <class _Shape, class _Fn>
  struct __bulk_chunked_fn
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Ts>
    _CCCL_TRIVIAL_API auto operator()(_Shape __begin, _Shape __end, _Ts&&... __values) noexcept(
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
  _CCCL_API static auto transform_sender(_Sndr&& __sndr, _CUDA_VSTD::__ignore_t)
  {
    static_assert(_CUDA_VSTD::is_same_v<tag_of_t<_Sndr>, bulk_t>);
    auto& [__tag, __data, __child]  = __sndr;
    auto& [__policy, __shape, __fn] = __data;

    using __chunked_fn_t = __bulk_chunked_fn<decltype(__shape), decltype(__fn)>;

    // Lower `bulk` to `bulk_chunked`. If `bulk_chunked` has a late customization, we will
    // see the customization.
    return bulk_chunked(_CUDA_VSTD::forward_like<_Sndr>(__child),
                        __policy,
                        __shape,
                        __chunked_fn_t{_CUDA_VSTD::forward_like<_Sndr>(__fn)});
  }

  _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return false;
  }
};

_CCCL_GLOBAL_CONSTANT auto bulk = bulk_t{};

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__bulk_t<bulk_t>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__bulk_t<bulk_chunked_t>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__bulk_t<bulk_unchunked_t>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

} // namespace cuda::experimental::execution

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_BULK
