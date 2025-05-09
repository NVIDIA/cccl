//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_STREAM_CONTEXT
#define __CUDAX__EXECUTION_STREAM_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__type_traits/is_callable.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/stream.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// A stream on which to schedule async deallocation operations
inline stream_ref __gc_stream() noexcept
{
  static stream str;
  return str;
}

template <class _Fn, class... _Args>
__global__ static void __stream_invoke(_Fn __fn, _Args... __args)
{
  static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...);
}

template <class _Fn, class... _Args>
__global__ static void __stream_invoke_r(_CUDA_VSTD::__call_result_t<_Fn, _Args...>* __return, _Fn __fn, _Args... __args)
{
  using _Return = _CUDA_VSTD::__call_result_t<_Fn, _Args...>;
  ::new (static_cast<void*>(__return)) _Return(static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...));
}

//////////////////////////////////////////////////////////////////////////////////////////
// stream_context
struct _CCCL_TYPE_VISIBILITY_DEFAULT stream_context : private __immovable
{
  stream_context() noexcept = default;

  _CCCL_HOST_API void sync() noexcept
  {
    __stream_.sync();
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler
  struct _CCCL_TYPE_VISIBILITY_DEFAULT scheduler
  {
    using scheduler_concept = scheduler_t;

    _CCCL_API bool operator==(const scheduler& __other) const noexcept
    {
      return __stream_ref_ == __other.__stream_ref_;
    }

    _CCCL_API static constexpr auto query(get_forward_progress_guarantee_t) noexcept
    {
      return forward_progress_guarantee::weakly_parallel;
    }

    _CCCL_API static constexpr auto query(get_domain_t) noexcept
    {
      return stream_domain{};
    }

    _CCCL_API constexpr auto query(get_stream_t) const noexcept
    {
      return __stream_ref_;
    }

    _CCCL_API auto schedule() const noexcept
    {
      return __sndr_t{{}, {__stream_ref_}};
    }

    stream_ref __stream_ref_;
  };

  _CCCL_TRIVIAL_HOST_API auto get_scheduler() noexcept -> scheduler
  {
    return scheduler{__stream_};
  }

  _CUDAX_SEMI_PRIVATE :
  ////////////////////////////////////////////////////////////////////////////////////////
  // environment of the stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    _CCCL_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept
    {
      return scheduler{__stream_ref_};
    }

    _CCCL_API constexpr auto query(get_stream_t) const noexcept
    {
      return __stream_ref_;
    }

    _CCCL_TRIVIAL_HOST_API static constexpr auto query(get_domain_t) noexcept
    {
      return stream_domain{};
    }

    stream_ref __stream_ref_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's operation state
  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _CCCL_API __opstate_t(_Rcvr __rcvr, stream_ref __stream_ref) noexcept
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __stream_ref_{__stream_ref}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API void start() & noexcept
    {
      NV_IF_TARGET(NV_IS_DEVICE, (__device_start();), (__host_start();));
    }

  private:
    _CCCL_HOST_API void __host_start() noexcept
    {
      __stream_invoke<<<1, 1, 0, __stream_ref_.get()>>>(set_value, __rcvr_ref{_CUDA_VSTD::addressof(__rcvr_)});
      if (auto __status = cudaGetLastError(); __status != cudaSuccess)
      {
        set_error(static_cast<_Rcvr&&>(__rcvr_), __status);
      }
    }

    _CCCL_DEVICE_API void __device_start() noexcept
    {
      // without the following, the kernel in __host_start will fail to launch with
      // cudaErrorInvalidDeviceFunction.
      [[maybe_unused]] auto __ignore = &__stream_invoke<set_value_t, __rcvr_ref<_Rcvr>>;
      set_value(static_cast<_Rcvr&&>(__rcvr_));
    }

    _Rcvr __rcvr_;
    stream_ref __stream_ref_;
  };

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __tag_t
  {};

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    template <class _Self>
    _CCCL_API static constexpr auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t(), set_error_t(cudaError_t)>{};
    }

    _CCCL_API constexpr auto get_env() const noexcept -> __env_t const&
    {
      return __env_;
    }

    template <class _Rcvr>
    _CCCL_API auto connect(_Rcvr __rcvr) const noexcept
    {
      return __opstate_t<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __env_.__stream_ref_};
    }

    _CCCL_NO_UNIQUE_ADDRESS __tag_t __tag_;
    __env_t __env_;
  };

  stream __stream_{};
};

using stream_scheduler = stream_context::scheduler;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_STREAM_CONTEXT
