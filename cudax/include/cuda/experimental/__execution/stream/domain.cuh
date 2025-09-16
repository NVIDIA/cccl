//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_DOMAIN
#define __CUDAX_EXECUTION_STREAM_DOMAIN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/get_stream.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__functional/compose.h>
#include <cuda/std/__type_traits/is_callable.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __stream
{
template <class _Tag>
struct __tag_t
{};

struct __no_tag_t
{};

_CCCL_API auto __tag_of(::cuda::std::__ignore_t) -> __no_tag_t;

template <class _Sndr>
_CCCL_API auto __tag_of(const _Sndr& __sndr) -> tag_of_t<_Sndr>;

template <class _Sndr>
using __tag_of_t = decltype(__stream::__tag_of(declval<_Sndr>()));

struct __adapted_t
{};

template <class _Sndr, class _GetStream>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

_CCCL_GLOBAL_CONSTANT auto __get_stream_from_attrs =
  __first_callable{get_stream, ::cuda::std::__compose(get_stream, get_completion_scheduler<set_value_t>)};

_CCCL_GLOBAL_CONSTANT auto __get_stream_from_env =
  __first_callable{get_stream, ::cuda::std::__compose(get_stream, get_scheduler)};

using __get_stream_from_attrs_t = decltype(__get_stream_from_attrs);
using __get_stream_from_env_t   = decltype(__get_stream_from_env);

// Get the stream either the sender's attributes or from the receiver's environment.
struct __get_stream_fn
{
  _CCCL_TEMPLATE(class _Sndr, class _Env)
  _CCCL_REQUIRES((__callable<__get_stream_from_attrs_t, env_of_t<_Sndr>, const _Env&>
                  || __callable<__get_stream_from_env_t, _Env>) )
  _CCCL_API constexpr auto operator()(const _Sndr& __sndr, const _Env& __env) const noexcept -> stream_ref
  {
    if constexpr (__callable<__get_stream_from_attrs_t, env_of_t<_Sndr>, const _Env&>)
    {
      // If the sender's attributes have a stream, use it.
      return __get_stream_from_attrs(execution::get_env(__sndr), __env);
    }
    else
    {
      // Otherwise, try to get the stream from the receiver's environment.
      return __get_stream_from_env(__env);
    }
  }
};

// Forward declaration of the __adapt function
template <class _Sndr, class _GetStream = __get_stream_fn>
_CCCL_API constexpr auto __adapt(_Sndr&& __sndr, _GetStream = {}) noexcept(__nothrow_decay_copyable<_Sndr>);
} // namespace __stream

_CCCL_GLOBAL_CONSTANT auto __get_stream = __stream::__get_stream_fn{};

//////////////////////////////////////////////////////////////////////////////////////////
// stream domain
struct stream_domain
{
  _CUDAX_SEMI_PRIVATE :
  struct __apply_adapt_t
  {
    // This is the default apply function that adapts a sender to a stream sender.
    // The constraint prevents this function from applying an adaptor to a sender
    // that has already been adapted. The __stream::__adapted_t query is present
    // only on receivers that come from an adapted sender.
    template <class _Sndr>
    _CCCL_API constexpr auto operator()(_Sndr&& __sndr, ::cuda::std::__ignore_t) const
      noexcept(__nothrow_decay_copyable<_Sndr>)
    {
      return __stream::__adapt(static_cast<_Sndr&&>(__sndr));
    }
  };

  struct __apply_passthru_t
  {
    template <class _Sndr>
    _CCCL_API constexpr auto operator()(_Sndr&& __sndr, ::cuda::std::__ignore_t) const
      noexcept(__nothrow_movable<_Sndr>) -> _Sndr
    {
      return static_cast<_Sndr&&>(__sndr);
    }
  };

  template <class _Tag>
  struct __apply_t : __apply_adapt_t
  {};

  template <class _Sndr, class _Env>
  _CCCL_API static constexpr auto __transform_strategy() noexcept
  {
    if constexpr (__queryable_with<_Env, __stream::__adapted_t>)
    {
      // The __stream::__adapted_t query is present only on receivers that come from an
      // adapted sender. Therefore, _Sndr has already been adapted. Pass it through as is.
      return __apply_passthru_t{};
    }
    else if constexpr (sender_for<_Sndr>)
    {
      // The sender has a tag type. Use the tag to determine the transformation to apply.
      return __apply_t<tag_of_t<_Sndr>>{};
    }
    else
    {
      // Otherwise, _Sndr is an unknown sender type that has not yet been adapted to
      // be a stream sender. Adapt it now.
      return __apply_adapt_t{};
    }
  }

  template <class _Sndr, class _Env>
  using __transform_strategy_t = decltype(__transform_strategy<_Sndr, _Env>());

public:
  _CCCL_TEMPLATE(class _Tag, class _Sndr, class... _Args)
  _CCCL_REQUIRES(__callable<__apply_t<_Tag>, _Sndr, _Args...>)
  _CCCL_NODEBUG_HOST_API static constexpr auto
  apply_sender(_Tag, _Sndr&& __sndr, _Args&&... __args) noexcept(__nothrow_callable<__apply_t<_Tag>, _Sndr, _Args...>)
    -> __call_result_t<__apply_t<_Tag>, _Sndr, _Args...>
  {
    return __apply_t<_Tag>{}(static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...);
  }

  _CCCL_TEMPLATE(class _Sndr, class _Env)
  _CCCL_REQUIRES(__callable<__transform_strategy_t<_Sndr, _Env>, _Sndr, const _Env&>)
  _CCCL_NODEBUG_API static constexpr auto transform_sender(_Sndr&& __sndr, const _Env& __env) noexcept(
    __nothrow_callable<__transform_strategy_t<_Sndr, _Env>, _Sndr, const _Env&>)
    -> __call_result_t<__transform_strategy_t<_Sndr, _Env>, _Sndr, const _Env&>
  {
    return __transform_strategy_t<_Sndr, _Env>{}(static_cast<_Sndr&&>(__sndr), __env);
  }
};

// If a sender has already been adapted to a stream sender, it will have a tag that is a specialization of
// __stream::__tag_t. In that case, we don't need to adapt it again, and we can just pass it through.
template <class _Tag>
struct stream_domain::__apply_t<__stream::__tag_t<_Tag>> : stream_domain::__apply_passthru_t
{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_DOMAIN
