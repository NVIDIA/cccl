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

#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/sync_wait.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Sndr, class _Rcvr>
struct __connect_emplace
{
  using type = connect_result_t<_Sndr, _Rcvr>;

  _CCCL_API operator type() && noexcept(__nothrow_connectable<_Sndr, _Rcvr>)
  {
    return execution::connect(static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr_));
  }

  _Sndr&& __sndr_;
  _Rcvr&& __rcvr_;
};

template <class _Sndr, class _Rcvr>
_CCCL_HOST_DEVICE __connect_emplace(_Sndr&& __sndr, _Rcvr&& __rcvr) -> __connect_emplace<_Sndr, _Rcvr>;

/////////////////////////////////////////////////////////////////////////////////
// sync_wait: customization for the stream scheduler
template <>
struct stream_domain::__apply_t<sync_wait_t>
{
  // TODO: calling sync_wait from device code is not supported yet.
  template <class _Sndr, class _Env>
  _CCCL_API auto operator()(_Sndr&& __sndr, _Env&& __env) const
  {
    NV_IF_TARGET(NV_IS_HOST,
                 (return __host_apply(static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env));),
                 (return __device_apply(static_cast<_Sndr&&>(__sndr), static_cast<_Env&&>(__env));))
    _CCCL_UNREACHABLE();
  }

  template <class _Sndr>
  _CCCL_API auto operator()(_Sndr&& __sndr) const
  {
    return (*this)(static_cast<_Sndr&&>(__sndr), env{});
  }

private:
  template <class _Sndr, class _Env>
  struct __state_t
  {
    using __values_t = typename sync_wait_t::__state_t<_Sndr, _Env>::__values_t;
    using __errors_t = typename sync_wait_t::__state_t<_Sndr, _Env>::__errors_t;
    using __rcvr_t   = sync_wait_t::__rcvr_t<_Sndr, _Env>;

    _CCCL_HOST_API explicit __state_t(_Sndr&& __sndr, _Env&& __env)
        : __result_{}
        , __state_{{{}, static_cast<_Env&&>(__env)}, &__result_, {}}
        , __opstate_{execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t{&__state_})}
    {}

    _CUDA_VSTD::optional<__values_t> __result_;
    sync_wait_t::__state_t<_Sndr, _Env> __state_;
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
    stream_ref __stream = get_stream(get_env(__sndr));

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
      __state.__errors_.__visit(sync_wait_t::__throw_error_fn{}, _CUDA_VSTD::move(__state.__errors_));
    }

    return _CUDA_VSTD::move(__box->__value.__result_);
  }
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_SYNC_WAIT
