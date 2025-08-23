//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_SYNC_WAIT
#define __CUDAX_EXECUTION_STREAM_SYNC_WAIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/sync_wait.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __stream
{
/////////////////////////////////////////////////////////////////////////////////
// sync_wait: customization for the stream scheduler
struct __sync_wait_t : private sync_wait_t
{
  // TODO: calling sync_wait from device code is not supported yet.
  template <class _Sndr, class _Env>
  _CCCL_API auto operator()(_Sndr&& __sndr, _Env&& __env) const
  {
    // _Sndr is a sender that has not yet been transformed to run on the stream domain.
    // The transformation would happen in due course in the connect cpo, so why transform
    // it here? This transformation shuffles the sender into one that can provide a
    // stream_ref, which is needed by __host_apply.
    auto __new_sndr = execution::transform_sender(stream_domain{}, static_cast<_Sndr&&>(__sndr), __env);

    NV_IF_TARGET(NV_IS_HOST,
                 (return __host_apply(::cuda::std::move(__new_sndr), static_cast<_Env&&>(__env));),
                 (return __device_apply(::cuda::std::move(__new_sndr), static_cast<_Env&&>(__env));))
    _CCCL_UNREACHABLE();
  }

  template <class _Sndr>
  _CCCL_API auto operator()(_Sndr&& __sndr) const
  {
    return (*this)(static_cast<_Sndr&&>(__sndr), env{});
  }

private:
  template <class _Sndr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    using __partial_completions_t = completion_signatures_of_t<_Sndr, __env_t<_Env>>;
    using __all_nothrow_t =
      typename __partial_completions_t::template __transform_q<__nothrow_decay_copyable_t, ::cuda::std::_And>;

    using __completions_t =
      __concat_completion_signatures_t<__partial_completions_t, __eptr_completion_if_t<!__all_nothrow_t::value>>;

    using __values_t = __value_types<__completions_t, __decayed_tuple, ::cuda::std::__type_self_t>;
    using __errors_t = __error_types<__completions_t, __decayed_variant>;
    using __rcvr_t   = sync_wait_t::__rcvr_t<__values_t, __errors_t, _Env>;

    _CCCL_HOST_API explicit __state_t(_Sndr&& __sndr, _Env&& __env)
        : __result_{}
        , __state_{{{}, static_cast<_Env&&>(__env)}, &__result_, {}}
        , __opstate_{execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state_})}
    {}

    ::cuda::std::optional<__values_t> __result_;
    sync_wait_t::__state_t<__values_t, __errors_t, _Env> __state_;
    connect_result_t<_Sndr, __rcvr_t> __opstate_;
  };

  template <class _Sndr, class _Env>
  _CCCL_DEVICE_API static auto __device_apply(_Sndr&& __sndr, _Env&& __env)
  {
    return sync_wait.apply_sender(static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env));
  }

  template <class _Sndr, class _Env>
  _CCCL_HOST_API static auto __host_apply(_Sndr&& __sndr, _Env&& __env)
  {
    stream_ref __stream = __get_stream(__sndr, __env);

    // Launch the sender with a continuation that will fill in a variant
    using __box_t = __managed_box<__state_t<_Sndr, _Env>>;
    auto __box    = __box_t::__make_unique(static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env));
    execution::start(__box->__value.__opstate_);

    // The kernels have been launched, now we sync the stream to guarantee forward progress.
    __stream.sync();

    // While waiting for the variant to be filled in, process any work that may be
    // delegated to this thread.
    auto& __state = __box->__value.__state_;
    __state.__loop_.run();

    if (__state.__errors_.__index() != __npos)
    {
      __state.__errors_.__visit(sync_wait_t::__throw_error_fn{}, ::cuda::std::move(__state.__errors_));
    }

    return ::cuda::std::move(__box->__value.__result_);
  }
};
} // namespace __stream

template <>
struct stream_domain::__apply_t<sync_wait_t> : __stream::__sync_wait_t
{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_SYNC_WAIT
