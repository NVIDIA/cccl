//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_SCHEDULER
#define __CUDAX_EXECUTION_STREAM_SCHEDULER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/get_stream.h>
#include <cuda/__utility/immovable.h>
#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental
{
namespace execution
{
template <int _BlockThreads, class _Tag, class _Rcvr, class... _Args>
_CCCL_VISIBILITY_HIDDEN __launch_bounds__(_BlockThreads) __global__
  void __stream_complete(_Tag, _Rcvr* __rcvr, _Args... __args)
{
  _Tag{}(static_cast<_Rcvr&&>(*__rcvr), static_cast<_Args&&>(__args)...);
}

////////////////////////////////////////////////////////////////////////////////////////
// stream scheduler
struct _CCCL_TYPE_VISIBILITY_DEFAULT stream_scheduler
{
  using scheduler_concept = scheduler_t;

  _CUDAX_SEMI_PRIVATE:
  ////////////////////////////////////////////////////////////////////////////////////////
  // attributes of the stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
  {
    [[nodiscard]] _CCCL_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
    {
      return __stream_;
    }

    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept
      -> stream_scheduler
    {
      return stream_scheduler{__stream_};
    }

    template <class _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<set_error_t>, _Env&& __env) const noexcept
      -> __scheduler_of_t<_Env&>
    {
      return execution::get_scheduler(__env);
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept
      -> stream_domain
    {
      return {};
    }

    template <class _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<set_error_t>, _Env&& __env) const noexcept
      -> __call_result_t<get_domain_t, _Env&>
    {
      return execution::get_domain(__env);
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(get_domain_override_t) const noexcept -> stream_domain
    {
      return {};
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(get_completion_behavior_t) const noexcept
    {
      return completion_behavior::asynchronous;
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
    _CCCL_API explicit __opstate_t(_Rcvr __rcvr, stream_ref __stream_ref) noexcept
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __stream_{__stream_ref}
    {
      _CCCL_ASSERT(execution::__get_pointer_attributes(this).type == cudaMemoryTypeManaged,
                   "stream scheduler's operation state must be allocated in managed memory");
    }

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API void start() noexcept
    {
      NV_IF_TARGET(NV_IS_HOST, (__host_start();), (__device_start();));
    }

  private:
    _CCCL_HOST_API void __host_start() noexcept
    {
      // Read the launch configuration passed to us by the parent operation. When we launch
      // the completion kernel, we will be completing the parent's receiver, so we must let
      // the receiver tell us how to launch the kernel.
      auto const __launch_dims      = get_launch_config(execution::get_env(__rcvr_)).dims;
      constexpr int __block_threads = decltype(__launch_dims)::static_count(experimental::thread, experimental::block);
      int const __grid_blocks       = __launch_dims.count(experimental::block, experimental::grid);
      static_assert(__block_threads != ::cuda::std::dynamic_extent);

      // printf("Launching completion kernel for stream_scheduler with %d block threads and %d grid blocks\n",
      //        __block_threads,
      //        __grid_blocks);

      // Launch the kernel that completes the receiver with the launch configuration from
      // the receiver.
      __stream_complete<__block_threads><<<__grid_blocks, __block_threads, 0, __stream_.get()>>>(set_value, &__rcvr_);

      if (auto __status = cudaGetLastError(); __status != cudaSuccess)
      {
        execution::set_error(static_cast<_Rcvr&&>(__rcvr_), cudaError_t(__status));
      }
    }

    // TODO: untested
    _CCCL_DEVICE_API void __device_start() noexcept
    {
      using __launch_dims_t         = decltype(get_launch_config(execution::get_env(__rcvr_)).dims);
      constexpr int __block_threads = __launch_dims_t::static_count(experimental::thread, experimental::block);

      // without the following, the kernel in __host_start will fail to launch with
      // cudaErrorInvalidDeviceFunction.
      ::__cccl_unused(&__stream_complete<__block_threads, set_value_t, _Rcvr>);
      execution::set_value(static_cast<_Rcvr&&>(__rcvr_));
    }

    _Rcvr __rcvr_;
    stream_ref __stream_;
  };

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __tag_t
  {};

public:
  _CCCL_API explicit constexpr stream_scheduler(stream_ref __stream) noexcept
      : __stream_{__stream}
  {}

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    _CCCL_API constexpr explicit __sndr_t(stream_ref __stream) noexcept
        : __attrs_{__stream}
    {}

    template <class _Self>
    _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t(), set_error_t(cudaError_t)>{};
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> const __attrs_t&
    {
      return __attrs_;
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr>
    {
      return __opstate_t<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __attrs_.__stream_};
    }

    _CCCL_NO_UNIQUE_ADDRESS __tag_t __tag_;
    __attrs_t __attrs_;
  };

  [[nodiscard]] _CCCL_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
  {
    return __stream_;
  }

  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept -> stream_domain
  {
    return {};
  }

  template <class _Env>
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<set_error_t>, _Env&&) const noexcept
    -> __call_result_t<get_domain_t, _Env&>
  {
    return {};
  }

  [[nodiscard]] _CCCL_API constexpr auto query(get_forward_progress_guarantee_t) const noexcept
    -> forward_progress_guarantee
  {
    return forward_progress_guarantee::weakly_parallel;
  }

  [[nodiscard]] _CCCL_API auto schedule() const noexcept -> __sndr_t
  {
    return __sndr_t{__stream_};
  }

  [[nodiscard]] _CCCL_API friend bool operator==(const stream_scheduler& __lhs, const stream_scheduler& __rhs) noexcept
  {
    return __lhs.__stream_ == __rhs.__stream_;
  }

  [[nodiscard]] _CCCL_API friend bool operator!=(const stream_scheduler& __lhs, const stream_scheduler& __rhs) noexcept
  {
    return __lhs.__stream_ != __rhs.__stream_;
  }

private:
  stream_ref __stream_;
};

// The stream_scheduler's sender does not need to be wrapped in a __stream::__sndr_t
// because it is already a stream sender. The following specialization ensures that
// no transform is applied to the stream_scheduler's sender.
template <>
struct stream_domain::__apply_t<stream_scheduler::__tag_t> : stream_domain::__apply_passthru_t
{};

} // namespace execution

_CCCL_HOST_API inline auto stream_ref::schedule() const noexcept
{
  return execution::schedule(execution::stream_scheduler{*this});
}

[[nodiscard]] _CCCL_API constexpr auto stream_ref::query(const execution::get_domain_t&) const noexcept
  -> execution::stream_domain
{
  return {};
}

} // namespace cuda::experimental

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_SCHEDULER
