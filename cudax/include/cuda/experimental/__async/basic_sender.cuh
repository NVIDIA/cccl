//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_BASIC_SENDER
#define __CUDAX_ASYNC_DETAIL_BASIC_SENDER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
template <class _Data, class _Rcvr>
struct __state
{
  _Data __data_;
  _Rcvr __receiver_;
};

struct receiver_defaults
{
  using receiver_concept = __async::receiver_t;

  template <class _Rcvr, class... _Args>
  _CUDAX_TRIVIAL_API static auto set_value(__ignore, _Rcvr& __rcvr, _Args&&... __args) noexcept
    -> __async::completion_signatures<__async::set_value_t(_Args...)>
  {
    __async::set_value(static_cast<_Rcvr&&>(__rcvr), static_cast<_Args&&>(__args)...);
    return {};
  }

  template <class _Rcvr, class _Error>
  _CUDAX_TRIVIAL_API static auto set_error(__ignore, _Rcvr& __rcvr, _Error&& __error) noexcept
    -> __async::completion_signatures<__async::set_error_t(_Error)>
  {
    __async::set_error(static_cast<_Rcvr&&>(__rcvr), static_cast<_Error&&>(__error));
    return {};
  }

  template <class _Rcvr>
  _CUDAX_TRIVIAL_API static auto
  set_stopped(__ignore, _Rcvr& __rcvr) noexcept -> __async::completion_signatures<__async::set_stopped_t()>
  {
    __async::set_stopped(static_cast<_Rcvr&&>(__rcvr));
    return {};
  }

  template <class _Rcvr>
  _CUDAX_TRIVIAL_API static decltype(auto) get_env(__ignore, const _Rcvr& __rcvr) noexcept
  {
    return __async::get_env(__rcvr);
  }
};

template <class _Data, class _Rcvr>
struct basic_receiver
{
  using receiver_concept = __async::receiver_t;
  using __rcvr_t         = typename _Data::receiver_tag;
  __state<_Data, _Rcvr>& __state_;

  template <class... _Args>
  _CUDAX_TRIVIAL_API void set_value(_Args&&... __args) noexcept
  {
    __rcvr_t::set_value(__state_.__data_, __state_.__receiver_, (_Args&&) __args...);
  }

  template <class _Error>
  _CUDAX_TRIVIAL_API void set_error(_Error&& __error) noexcept
  {
    __rcvr_t::set_error(__state_.__data_, __state_.__receiver_, (_Error&&) __error);
  }

  _CUDAX_TRIVIAL_API void set_stopped() noexcept
  {
    __rcvr_t::set_stopped(__state_.__data_, __state_.__receiver_);
  }

  _CUDAX_TRIVIAL_API decltype(auto) get_env() const noexcept
  {
    return __rcvr_t::get_env(__state_.__data_, __state_.__receiver_);
  }
};

template <class _Rcvr>
inline constexpr bool has_no_environment = _CUDA_VSTD::is_same_v<_Rcvr, receiver_archetype>;

template <bool _HasStopped, class _Data, class _Rcvr>
struct __mk_completions
{
  using __rcvr_t = typename _Data::receiver_tag;

  template <class... _Args>
  using __set_value_t =
    decltype(+*__rcvr_t::set_value(__declval<_Data&>(), __declval<receiver_archetype&>(), __declval<_Args>()...));

  template <class _Error>
  using __set_error_t =
    decltype(+*__rcvr_t::set_error(__declval<_Data&>(), __declval<receiver_archetype&>(), __declval<_Error>()));

  using __set_stopped_t = __async::completion_signatures<>;
};

template <class _Data, class _Rcvr>
struct __mk_completions<true, _Data, _Rcvr> : __mk_completions<false, _Data, _Rcvr>
{
  using __rcvr_t = typename _Data::receiver_tag;

  using __set_stopped_t = decltype(+*__rcvr_t::set_stopped(__declval<_Data&>(), __declval<receiver_archetype&>()));
};

template <class...>
using __ignore_value_signature = __async::completion_signatures<>;

template <class>
using __ignore_error_signature = __async::completion_signatures<>;

template <class _Completions>
constexpr bool __has_stopped =
  !_CUDA_VSTD::is_same_v<__async::completion_signatures<>,
                         __async::transform_completion_signatures<_Completions,
                                                                  __async::completion_signatures<>,
                                                                  __ignore_value_signature,
                                                                  __ignore_error_signature>>;

template <bool _PotentiallyThrowing, class _Rcvr>
void set_current_exception_if([[maybe_unused]] _Rcvr& __rcvr) noexcept
{
  if constexpr (_PotentiallyThrowing)
  {
    __async::set_error(static_cast<_Rcvr&&>(__rcvr), ::std::current_exception());
  }
}

// A generic type that holds the data for an async operation, and
// that provides a `start` method for enqueuing the work.
template <class _Sndr, class _Data, class _Rcvr>
struct __basic_opstate
{
  using __rcvr_t        = basic_receiver<_Data, _Rcvr>;
  using __completions_t = completion_signatures_of_t<_Sndr, __rcvr_t>;
  using __traits_t      = __mk_completions<__has_stopped<__completions_t>, _Data, _Rcvr>;

  using completion_signatures = //
    transform_completion_signatures<__completions_t,
                                    // TODO: add set_error_t(exception_ptr) if constructing
                                    // the state or connecting the sender is potentially throwing.
                                    __async::completion_signatures<>,
                                    __traits_t::template __set_value_t,
                                    __traits_t::template __set_error_t,
                                    typename __traits_t::__set_stopped_t>;

  _CUDAX_API __basic_opstate(_Sndr&& __sndr, _Data __data, _Rcvr __rcvr)
      : __state_{static_cast<_Data&&>(__data), static_cast<_Rcvr&&>(__rcvr)}
      , __op_(__async::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{__state_}))
  {}

  _CUDAX_TRIVIAL_API void start() noexcept
  {
    __async::start(__op_);
  }

  __state<_Data, _Rcvr> __state_;
  __async::connect_result_t<_Sndr, __rcvr_t> __op_;
};

template <class _Sndr, class _Rcvr>
_CUDAX_TRIVIAL_API auto __make_opstate(_Sndr __sndr, _Rcvr __rcvr)
{
  auto [__tag, __data, __child] = static_cast<_Sndr&&>(__sndr);
  using __data_t                = decltype(__data);
  using __child_t               = decltype(__child);
  (void) __tag;
  return __basic_opstate(
    static_cast<__child_t&&>(__child), static_cast<__data_t&&>(__data), static_cast<_Rcvr&&>(__rcvr));
}

template <class _Data, class... _Sndrs>
_CUDAX_TRIVIAL_API auto
__get_attrs(int, const _Data& __data, const _Sndrs&... __sndrs) noexcept -> decltype(__data.get_attrs(__sndrs...))
{
  return __data.get_attrs(__sndrs...);
}

template <class _Data, class... _Sndrs>
_CUDAX_TRIVIAL_API auto
__get_attrs(long, const _Data&, const _Sndrs&... __sndrs) noexcept -> decltype(__async::get_env(__sndrs...))
{
  return __async::get_env(__sndrs...);
}

template <class _Data, class... _Sndrs>
struct basic_sender;

template <class _Data, class _Sndr>
struct basic_sender<_Data, _Sndr>
{
  using sender_concept = __async::sender_t;
  using __tag_t        = typename _Data::sender_tag;
  using __rcvr_t       = typename _Data::receiver_tag;

  _CCCL_NO_UNIQUE_ADDRESS __tag_t __tag_;
  _Data __data_;
  _Sndr __sndr_;

  // Connect the sender to the receiver (the continuation) and
  // return the state_type object for this operation.
  template <class _Rcvr>
  _CUDAX_TRIVIAL_API auto connect(_Rcvr __rcvr) &&
  {
    return __make_opstate(static_cast<basic_sender&&>(*this), static_cast<_Rcvr&&>(__rcvr));
  }

  template <class _Rcvr>
  _CUDAX_TRIVIAL_API auto connect(_Rcvr __rcvr) const&
  {
    return __make_opstate(*this, static_cast<_Rcvr&&>(__rcvr));
  }

  _CUDAX_TRIVIAL_API decltype(auto) get_env() const noexcept
  {
    return __async::__get_attrs(0, __data_, __sndr_);
  }
};

template <class _Data, class... _Sndrs>
basic_sender(__ignore, _Data, _Sndrs...) -> basic_sender<_Data, _Sndrs...>;

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
