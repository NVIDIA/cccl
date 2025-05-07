//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CONTINUES_ON
#define __CUDAX_ASYNC_DETAIL_CONTINUES_ON

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
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t
{
private:
  template <class... _As>
  using __set_value_tuple_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<set_value_t, __decay_t<_As>...>;

  template <class _Error>
  using __set_error_tuple_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<set_error_t, __decay_t<_Error>>;

  using __set_stopped_tuple_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<set_stopped_t>;

  using __complete_fn _CCCL_NODEBUG_ALIAS = void (*)(void*) noexcept;

  template <class _Rcvr, class _Result>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept _CCCL_NODEBUG_ALIAS = receiver_t;
    _Rcvr __rcvr_;
    _Result __result_;
    __complete_fn __complete_;

    template <class _Tag, class... _As>
    _CCCL_API void operator()(_Tag, _As&... __as) noexcept
    {
      _Tag()(static_cast<_Rcvr&&>(__rcvr_), static_cast<_As&&>(__as)...);
    }

    template <class _Tag, class... _As>
    _CCCL_API void __set_result(_Tag, _As&&... __as) noexcept
    {
      using __tupl_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<_Tag, __decay_t<_As>...>;
      if constexpr (__nothrow_decay_copyable<_As...>)
      {
        __result_.template __emplace<__tupl_t>(_Tag(), static_cast<_As&&>(__as)...);
      }
      else
      {
        _CUDAX_TRY( //
          ({ //
            __result_.template __emplace<__tupl_t>(_Tag(), static_cast<_As&&>(__as)...);
          }),
          _CUDAX_CATCH(...) //
          ({ //
            execution::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
          }) //
        )
      }
      __complete_ = +[](void* __ptr) noexcept {
        auto& __self = *static_cast<__rcvr_t*>(__ptr);
        auto& __tupl = *static_cast<__tupl_t*>(__self.__result_.__ptr());
        _CUDA_VSTD::__apply(__self, __tupl);
      };
    }

    _CCCL_API void set_value() noexcept
    {
      __complete_(this);
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __error) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Error&&>(__error));
    }

    _CCCL_API void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
    }

    _CCCL_API auto get_env() const noexcept -> env_of_t<_Rcvr>
    {
      return execution::get_env(__rcvr_);
    }
  };

  template <class _Rcvr, class _CvSndr, class _Sch>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;
    using __env_t _CCCL_NODEBUG_ALIAS                 = _FWD_ENV_T<env_of_t<_Rcvr>>;

    using __result_t _CCCL_NODEBUG_ALIAS =
      typename completion_signatures_of_t<_CvSndr,
                                          __env_t>::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;

    _CCCL_API __opstate_t(_CvSndr&& __sndr, _Sch __sch, _Rcvr __rcvr)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr), {}, nullptr}
        , __opstate1_{execution::connect(static_cast<_CvSndr&&>(__sndr), __rcvr_ref{*this})}
        , __opstate2_{execution::connect(schedule(__sch), __rcvr_ref{__rcvr_})}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API void start() noexcept
    {
      execution::start(__opstate1_);
    }

    template <class... _As>
    _CCCL_API void set_value(_As&&... __as) noexcept
    {
      __rcvr_.__set_result(set_value_t(), static_cast<_As&&>(__as)...);
      execution::start(__opstate2_);
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __error) noexcept
    {
      __rcvr_.__set_result(set_error_t(), static_cast<_Error&&>(__error));
      execution::start(__opstate2_);
    }

    _CCCL_API void set_stopped() noexcept
    {
      __rcvr_.__set_result(set_stopped_t());
      execution::start(__opstate2_);
    }

    _CCCL_API auto get_env() const noexcept -> __env_t
    {
      return execution::get_env(__rcvr_.__rcvr);
    }

    __rcvr_t<_Rcvr, __result_t> __rcvr_;
    connect_result_t<_CvSndr, __rcvr_ref<__opstate_t, __env_t>> __opstate1_;
    connect_result_t<schedule_result_t<_Sch>, __rcvr_ref<__rcvr_t<_Rcvr, __result_t>>> __opstate2_;
  };

public:
  template <class _Sndr, class _Sch>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;

  template <class _Sndr, class _Sch>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr __sndr, _Sch __sch) const;

  template <class _Sch>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sch __sch) const noexcept -> __closure_t<_Sch>;
};

template <class _Sch>
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t::__closure_t
{
  _Sch __sch;

  template <class _Sndr>
  _CCCL_TRIVIAL_API friend constexpr auto operator|(_Sndr __sndr, __closure_t __self)
  {
    return continues_on_t()(static_cast<_Sndr&&>(__sndr), static_cast<_Sch&&>(__self.__sch));
  }
};

template <class _Tag>
struct __decay_args
{
  template <class... _Ts>
  _CCCL_TRIVIAL_API constexpr auto operator()() const noexcept
  {
    if constexpr (!__decay_copyable<_Ts...>)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, continues_on_t),
                                          _WHAT(_ARGUMENTS_ARE_NOT_DECAY_COPYABLE),
                                          _WITH_ARGUMENTS(_Ts...)>();
    }
    else if constexpr (!__nothrow_decay_copyable<_Ts...>)
    {
      return completion_signatures<_Tag(__decay_t<_Ts>...), set_error_t(::std::exception_ptr)>{};
    }
    else
    {
      return completion_signatures<_Tag(__decay_t<_Ts>...)>{};
    }
  }
};

template <class _Sndr, class _Sch>
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS continues_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
  {
    template <class _SetTag>
    _CCCL_API auto query(get_completion_scheduler_t<_SetTag>) const noexcept -> _Sch
    {
      return __sndr_->__sch_;
    }

    template <class _Query>
    _CCCL_API auto query(_Query) const -> __query_result_t<_Query, env_of_t<_Sndr>>
    {
      return execution::get_env(__sndr_->__sndr_).query(_Query{});
    }

    const __sndr_t* __sndr_;
  };

  template <class _Self, class... _Env>
  _CCCL_API static constexpr auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__sch_completions) = execution::get_completion_signatures<schedule_result_t<_Sch>, _Env...>())
      {
        // The scheduler contributes error and stopped completions.
        return concat_completion_signatures(
          transform_completion_signatures(__sch_completions, __swallow_transform()),
          transform_completion_signatures(
            __child_completions, __decay_args<set_value_t>{}, __decay_args<set_error_t>{}));
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, _Sndr, _Sch>
  {
    return {static_cast<_Sndr&&>(__sndr_), __sch_, static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, const _Sndr&, _Sch>
  {
    return {__sndr_, __sch_, static_cast<_Rcvr&&>(__rcvr)};
  }

  _CCCL_API auto get_env() const noexcept -> __attrs_t
  {
    return __attrs_t{this};
  }
};

template <class _Sndr, class _Sch>
_CCCL_TRIVIAL_API constexpr auto continues_on_t::operator()(_Sndr __sndr, _Sch __sch) const
{
  using __dom_t _CCCL_NODEBUG_ALIAS = domain_for_t<_Sndr>;
  return transform_sender(__dom_t{}, __sndr_t<_Sndr, _Sch>{{}, __sch, static_cast<_Sndr&&>(__sndr)});
}

template <class _Sch>
_CCCL_TRIVIAL_API constexpr auto continues_on_t::operator()(_Sch __sch) const noexcept -> __closure_t<_Sch>
{
  return __closure_t<_Sch>{__sch};
}

template <class _Sndr, class _Sch>
inline constexpr size_t structured_binding_size<continues_on_t::__sndr_t<_Sndr, _Sch>> = 3;

_CCCL_GLOBAL_CONSTANT continues_on_t continues_on{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif
