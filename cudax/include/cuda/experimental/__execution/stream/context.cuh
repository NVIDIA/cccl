//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_CONTEXT_IMPL
#define __CUDAX_EXECUTION_STREAM_CONTEXT_IMPL

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/unique_ptr.h>
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

#include <new> // IWYU pragma: keep for placement new

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Tag, class _Rcvr, class... _Args>
__launch_bounds__(1) __global__ static void __stream_complete(_Tag, _Rcvr* __rcvr, _Args... __args)
{
  _Tag{}(static_cast<_Rcvr&&>(*__rcvr), static_cast<_Args&&>(__args)...);
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

    [[nodiscard]] _CCCL_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
    {
      return __stream_;
    }

    [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t) noexcept -> stream_domain
    {
      return {};
    }

    [[nodiscard]] _CCCL_API static constexpr auto query(get_forward_progress_guarantee_t) noexcept
      -> forward_progress_guarantee
    {
      return forward_progress_guarantee::weakly_parallel;
    }

    [[nodiscard]] _CCCL_API auto schedule() const noexcept
    {
      return __sndr_t{{}, {__stream_}};
    }

    [[nodiscard]] _CCCL_API friend bool operator==(const scheduler& __lhs, const scheduler& __rhs) noexcept
    {
      return __lhs.__stream_ == __rhs.__stream_;
    }

    [[nodiscard]] _CCCL_API friend bool operator!=(const scheduler& __lhs, const scheduler& __rhs) noexcept
    {
      return __lhs.__stream_ != __rhs.__stream_;
    }

    stream_ref __stream_;
  };

  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto get_scheduler() noexcept -> scheduler
  {
    return scheduler{__stream_};
  }

  _CUDAX_SEMI_PRIVATE :
  ////////////////////////////////////////////////////////////////////////////////////////
  // attributes of the stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
  {
    [[nodiscard]] _CCCL_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
    {
      return __stream_;
    }

    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> scheduler
    {
      return scheduler{__stream_};
    }

    [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(get_domain_t) noexcept -> stream_domain
    {
      return {};
    }

    [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(get_domain_late_t) noexcept -> stream_domain
    {
      return {};
    }

    stream_ref __stream_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's operation state
  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API explicit __opstate_t(_Rcvr __rcvr, stream_ref __stream_ref) noexcept(__nothrow_movable<_Rcvr>)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __stream_{__stream_ref}
    {
      _CCCL_ASSERT(execution::__get_pointer_attributes(this).type == ::cudaMemoryTypeManaged,
                   "stream scheduler's operation state must be allocated in managed memory");
    }

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API void start() noexcept
    {
      NV_IF_TARGET(NV_IS_HOST, (__host_start();), (__device_start();));
    }

  private:
    _CCCL_HOST_API void __host_start() noexcept
    {
      __stream_complete<<<1, 1, 0, __stream_.get()>>>(set_value, &__rcvr_);
      if (auto __status = cudaGetLastError(); __status != cudaSuccess)
      {
        execution::set_error(static_cast<_Rcvr&&>(__rcvr_), cudaError_t(__status));
      }
    }

    _CCCL_DEVICE_API void __device_start() noexcept
    {
      // without the following, the kernel in __host_start will fail to launch with
      // cudaErrorInvalidDeviceFunction.
      [[maybe_unused]] auto __ignore = &__stream_complete<set_value_t, _Rcvr>;
      execution::set_value(static_cast<_Rcvr&&>(__rcvr_));
    }

    _Rcvr __rcvr_;
    stream_ref __stream_;
  };

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __tag_t
  {};

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    template <class _Self>
    _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t(), set_error_t(cudaError_t)>{};
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> const __attrs_t&
    {
      return __env_;
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr>
    {
      return __opstate_t<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __env_.__stream_};
    }

    _CCCL_NO_UNIQUE_ADDRESS __tag_t __tag_;
    __attrs_t __env_;
  };

  stream __stream_{};
};

using stream_scheduler = stream_context::scheduler;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_CONTEXT_IMPL
