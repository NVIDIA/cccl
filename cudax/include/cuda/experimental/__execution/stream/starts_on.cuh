//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_STARTS_ON
#define __CUDAX_EXECUTION_STREAM_STARTS_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/get_stream.h>
#include <cuda/std/__functional/compose.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/starts_on.cuh>
#include <cuda/experimental/__execution/stream/adaptor.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/write_env.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __stream
{
struct __starts_on_t
{
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __get_stream_fn
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_API auto operator()(const _Sndr& __sndr, ::cuda::std::__ignore_t) const
    {
      // __sndr is a write_env sender (see __mk_sndr_base below), which contains an
      // environment that contains the stream scheduler, from which we can obtain the
      // stream.
      auto& [__ign0, __env, __ign1] = __sndr;
      return cuda::get_stream(get_scheduler(__env));
    }
  };

  template <class _Sch, class _Sndr>
  [[nodiscard]] static _CCCL_API constexpr auto __mk_sndr_base(_Sch __sch, _Sndr&& __sndr)
  {
    // This is the implementation of the starts_on sender for the stream domain. _Sndr
    // here is the child of the starts_on sender, and _Sch is the stream scheduler. We use
    // write_env to let _Sndr and its children know that they are running on the stream
    // scheduler. We construct the adaptor with a __get_stream_fn that knows how to obtain
    // the stream from the write_env sender.
    return __stream::__adapt(write_env(static_cast<_Sndr&&>(__sndr), __mk_sch_env(__sch)), __get_stream_fn{});
  }

  template <class _Sch, class _Sndr>
  using __sndr_base_t = decltype(__starts_on_t::__mk_sndr_base(declval<_Sch>(), declval<_Sndr>()));

  template <class _Sch, class _Sndr>
  using __with_sch_t = __call_result_t<write_env_t, _Sndr, __call_result_t<__mk_sch_env_t, _Sch>>;

  // Wrap the sender returned from __mk_sndr_base in a type that hides the complexity of
  // the sender's type name. This results in more readable diagnostics.
  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __stream::__sndr_t<__with_sch_t<_Sch, _Sndr>, __get_stream_fn>
  {
    // BUGBUG NO this is a wrong use of __mk_sch_env. it needs to be passed an
    // environment. turn this into a transform_sender.
    _CCCL_API explicit constexpr __sndr_t(_Sch __sch, _Sndr __sndr)
        : __stream::__sndr_t<__with_sch_t<_Sch, _Sndr>, __get_stream_fn>{
            {}, {}, write_env(static_cast<_Sndr&&>(__sndr), __mk_sch_env(__sch))}
    {}
  };

  // The connect cpo calls transform_sender, which is directed here for starts_on senders.
  // It returns a custom sender that knows how to start the child sender on the specified
  // stream.
  template <class _Sndr>
  [[nodiscard]] _CCCL_API auto operator()(_Sndr&& __sndr, ::cuda::std::__ignore_t) const
  {
    auto& [__ign0, __sch, __child] = __sndr;
    return __sndr_t{__sch, ::cuda::std::forward_like<_Sndr>(__child)};
  }
};
} // namespace __stream

// Start work on the GPU
template <>
struct stream_domain::__apply_t<starts_on_t> : __stream::__starts_on_t
{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_STARTS_ON
