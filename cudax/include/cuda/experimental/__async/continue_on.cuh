//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CONTINUE_ON
#define __CUDAX_ASYNC_DETAIL_CONTINUE_ON

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
#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__async/variant.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
struct continue_on_t
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC
  template <class... _As>
  using __set_value_tuple_t = __tuple<set_value_t, __decay_t<_As>...>;

  template <class _Error>
  using __set_error_tuple_t = __tuple<set_error_t, __decay_t<_Error>>;

  using __set_stopped_tuple_t = __tuple<set_stopped_t>;

  using __complete_fn = void (*)(void*) noexcept;

  template <class... _Ts>
  using __set_value_completion =
    _CUDA_VSTD::conditional_t<__nothrow_decay_copyable<_Ts...>,
                              completion_signatures<set_value_t(__decay_t<_Ts>...)>,
                              completion_signatures<set_value_t(__decay_t<_Ts>...), set_error_t(::std::exception_ptr)>>;

  template <class _Error>
  using __set_error_completion =
    _CUDA_VSTD::conditional_t<__nothrow_decay_copyable<_Error>,
                              completion_signatures<set_error_t(__decay_t<_Error>)>,
                              completion_signatures<set_error_t(__decay_t<_Error>), set_error_t(::std::exception_ptr)>>;

  template <class _Rcvr, class _Result>
  struct __rcvr_t
  {
    using receiver_concept = receiver_t;
    _Rcvr __rcvr_;
    _Result __result_;
    __complete_fn __complete_;

    template <class _Tag, class... _As>
    _CUDAX_API void operator()(_Tag, _As&... __as) noexcept
    {
      _Tag()(static_cast<_Rcvr&&>(__rcvr_), static_cast<_As&&>(__as)...);
    }

    template <class _Tag, class... _As>
    _CUDAX_API void __set_result(_Tag, _As&&... __as) noexcept
    {
      using __tupl_t = __tuple<_Tag, __decay_t<_As>...>;
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
          _CUDAX_CATCH(...)( //
            { //
              __async::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
            }))
      }
      __complete_ = +[](void* __ptr) noexcept {
        auto& __self = *static_cast<__rcvr_t*>(__ptr);
        auto& __tupl = *static_cast<__tupl_t*>(__self.__result_.__ptr());
        __tupl.__apply(__self, __tupl);
      };
    }

    _CUDAX_API void set_value() noexcept
    {
      __complete_(this);
    }

    template <class _Error>
    _CUDAX_API void set_error(_Error&& __error) noexcept
    {
      __async::set_error(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Error&&>(__error));
    }

    _CUDAX_API void set_stopped() noexcept
    {
      __async::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
    }

    _CUDAX_API env_of_t<_Rcvr> get_env() const noexcept
    {
      return __async::get_env(__rcvr_);
    }
  };

  template <class _Rcvr, class _CvSndr, class _Sch>
  struct __opstate_t
  {
    _CUDAX_API friend auto get_env(const __opstate_t* __self) noexcept -> env_of_t<_Rcvr>
    {
      return __async::get_env(__self->__rcvr_.__rcvr);
    }

    using operation_state_concept = operation_state_t;
    using __result_t =
      __transform_completion_signatures<completion_signatures_of_t<_CvSndr, __opstate_t*>,
                                        __set_value_tuple_t,
                                        __set_error_tuple_t,
                                        __set_stopped_tuple_t,
                                        __variant>;

    // The scheduler contributes error and stopped completions.
    // This causes its set_value_t() completion to be ignored.
    using __scheduler_completions = //
      transform_completion_signatures<completion_signatures_of_t<schedule_result_t<_Sch>, __rcvr_t<_Rcvr, __result_t>*>,
                                      __async::completion_signatures<>,
                                      _CUDA_VSTD::__type_always<__async::completion_signatures<>>::__call>;

    // The continue_on completions are the scheduler's error
    // and stopped completions, plus the sender's completions
    // with all the result data types decayed.
    using completion_signatures = //
      transform_completion_signatures<completion_signatures_of_t<_CvSndr, __opstate_t*>,
                                      __scheduler_completions,
                                      __set_value_completion,
                                      __set_error_completion>;

    __rcvr_t<_Rcvr, __result_t> __rcvr_;
    connect_result_t<_CvSndr, __opstate_t*> __opstate1_;
    connect_result_t<schedule_result_t<_Sch>, __rcvr_t<_Rcvr, __result_t>*> __opstate2_;

    _CUDAX_API __opstate_t(_CvSndr&& __sndr, _Sch __sch, _Rcvr __rcvr)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr), {}, nullptr}
        , __opstate1_{__async::connect(static_cast<_CvSndr&&>(__sndr), this)}
        , __opstate2_{__async::connect(schedule(__sch), &__rcvr_)}
    {}

    _CUDAX_IMMOVABLE(__opstate_t);

    _CUDAX_API void start() noexcept
    {
      __async::start(__opstate1_);
    }

    template <class... _As>
    _CUDAX_API void set_value(_As&&... __as) noexcept
    {
      __rcvr_.__set_result(set_value_t(), static_cast<_As&&>(__as)...);
      __async::start(__opstate2_);
    }

    template <class _Error>
    _CUDAX_API void set_error(_Error&& __error) noexcept
    {
      __rcvr_.__set_result(set_error_t(), static_cast<_Error&&>(__error));
      __async::start(__opstate2_);
    }

    _CUDAX_API void set_stopped() noexcept
    {
      __rcvr_.__set_result(set_stopped_t());
      __async::start(__opstate2_);
    }
  };

  template <class _Sndr, class _Sch>
  struct __sndr_t;

  template <class _Sch>
  struct __closure_t;

public:
  template <class _Sndr, class _Sch>
  _CUDAX_API __sndr_t<_Sndr, _Sch> operator()(_Sndr __sndr, _Sch __sch) const noexcept;

  template <class _Sch>
  _CUDAX_TRIVIAL_API __closure_t<_Sch> operator()(_Sch __sch) const noexcept;
};

template <class _Sch>
struct continue_on_t::__closure_t
{
  _Sch __sch;

  template <class _Sndr>
  _CUDAX_TRIVIAL_API friend auto operator|(_Sndr __sndr, __closure_t&& __self)
  {
    return continue_on_t()(static_cast<_Sndr&&>(__sndr), static_cast<_Sch&&>(__self.__sch));
  }
};

template <class _Sndr, class _Sch>
struct continue_on_t::__sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS continue_on_t __tag;
  _Sch __sch;
  _Sndr __sndr;

  struct __attrs_t
  {
    __sndr_t* __sndr;

    template <class _SetTag>
    _CUDAX_API auto query(get_completion_scheduler_t<_SetTag>) const noexcept
    {
      return __sndr->__sch;
    }

    template <class _Query>
    _CUDAX_API auto query(_Query) const //
      -> __query_result_t<_Query, env_of_t<_Sndr>>
    {
      return __async::get_env(__sndr->__sndr).query(_Query{});
    }
  };

  template <class _Rcvr>
  _CUDAX_API __opstate_t<_Rcvr, _Sndr, _Sch> connect(_Rcvr __rcvr) &&
  {
    return {static_cast<_Sndr&&>(__sndr), __sch, static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  _CUDAX_API __opstate_t<_Rcvr, const _Sndr&, _Sch> connect(_Rcvr __rcvr) const&
  {
    return {__sndr, __sch, static_cast<_Rcvr&&>(__rcvr)};
  }

  _CUDAX_API __attrs_t get_env() const noexcept
  {
    return __attrs_t{this};
  }
};

template <class _Sndr, class _Sch>
_CUDAX_API auto
continue_on_t::operator()(_Sndr __sndr, _Sch __sch) const noexcept -> continue_on_t::__sndr_t<_Sndr, _Sch>
{
  return __sndr_t<_Sndr, _Sch>{{}, __sch, static_cast<_Sndr&&>(__sndr)};
}

template <class _Sch>
_CUDAX_TRIVIAL_API continue_on_t::__closure_t<_Sch> continue_on_t::operator()(_Sch __sch) const noexcept
{
  return __closure_t<_Sch>{__sch};
}

_CCCL_GLOBAL_CONSTANT continue_on_t continue_on{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
