//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_STREAM_SYNC_WAIT_CUH
#define __CUDAX__EXECUTION_STREAM_SYNC_WAIT_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/get_stream.h>
#include <cuda/std/__exception/cuda_error.h>

#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/run_loop.cuh>
#include <cuda/experimental/__execution/stream/context.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/stream/storage_registry.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
/////////////////////////////////////////////////////////////////////////////////
// sync_wait: customization for the stream scheduler
template <>
struct stream_domain::__apply_t<sync_wait_t>
{
  _CUDAX_SEMI_PRIVATE :
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    stream_ref __stream_;
    __storage_registry_context* __stg_context_; // lives on the host stack
    __storage_registry __stg_; // lives in managed memory ultimately
    cudaError_t __status_{cudaSuccess};
    run_loop __loop_{};
  };

  template <class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_ex_t : __state_t
  {
    _CUDA_VSTD::optional<_Values> __value_{};
  };

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    _CCCL_API auto query(get_storage_registry_t) const noexcept
    {
      return __state_->__stg_;
    }

    _CCCL_API auto query(get_storage_registry_context_t) const noexcept -> decltype(auto)
    {
      return (*__state_->__stg_context_);
    }

    _CCCL_API auto query(get_stream_t) const noexcept
    {
      return __state_->__stream_;
    }

    _CCCL_API auto query(get_scheduler_t) const noexcept
    {
      return stream_context::scheduler{__state_->__stream_};
    }

    _CCCL_API auto query(get_delegation_scheduler_t) const noexcept
    {
      return __state_->__loop_.get_scheduler();
    }

    __state_t* __state_{nullptr};
  };

  _CCCL_API static auto __to_cuda_error(cudaError_t __status) noexcept
  {
    _CCCL_ASSERT(__status != cudaSuccess, "cudaSuccess is not a valid error completion");
    return __status;
  }

  _CCCL_API static auto __to_cuda_error(__ignore) noexcept
  {
    return cudaErrorUnknown;
  }

  template <class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _Args>
    _CCCL_API void set_value(_Args&&... __args) && noexcept
    {
      auto& __state = *static_cast<__state_ex_t<_Values>*>(__env_.__state_);
      _CUDAX_TRY( //
        ({ //
          __state.__value_.emplace(static_cast<_Args&&>(__args)...);
        }),
        _CUDAX_CATCH(...) //
        ({ //
          __state.__status_ = cudaErrorUnknown;
        }))
      __state.__loop_.finish();
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __err) && noexcept
    {
      __env_.__state_->__status_ = __to_cuda_error(static_cast<_Error&&>(__err));
      __env_.__state_->__loop_.finish();
    }

    _CCCL_API void set_stopped() && noexcept
    {
      __env_.__state_->__loop_.finish();
    }

    _CCCL_TRIVIAL_HOST_API auto get_env() const noexcept
    {
      return __env_;
    }

    __env_t __env_{};
  };

  template <class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __mk_state
  {
    _CCCL_API auto
    operator()(stream_ref __stream, __storage_registry_context* __context, __storage_registry __stg) noexcept
      -> __state_ex_t<_Values>
    {
      return __state_ex_t<_Values>{{__stream, __context, __stg}};
    }
  };

public:
  template <class _Sndr, class... _Env>
  _CCCL_HOST_API auto operator()(_Sndr&& __sndr, _Env&&...) const
  {
    // TODO: use _Env
    using __sigs_t    = completion_signatures_of_t<_Sndr, __env_t>;
    using __values_t  = __value_types<__sigs_t, _CUDA_VSTD::tuple, _CUDA_VSTD::__type_self_t>;
    using __opstate_t = connect_result_t<_Sndr, __rcvr_t<__values_t>>;

    auto __stream = get_stream(get_completion_scheduler<set_value_t>(get_env(__sndr)));
    __storage_registry_context __context{__stream};

    const auto __state_id   = __context.__reserve_for<__state_ex_t<__values_t>>();
    const auto __opstate_id = __context.__reserve_for<__opstate_t>();
    // Allocate all the reserved temporary memory:
    auto __stg = __context.__finalize();

    // create the object to hold the result:
    auto& __state = __stg.__write_at_from(__state_id, __mk_state<__values_t>{}, __stream, &__context, __stg);

    // Put the operation state in the temp storage:
    auto& __opstate =
      __stg.__write_at_from(__opstate_id, connect, static_cast<_Sndr&&>(__sndr), __rcvr_t<__values_t>{{&__state}});
    start(__opstate); // Start the operation

    __state.__loop_.run(); // Drive the run_loop:
    __stream.sync(); // Wait for the stream to finish:

    // Throw if the stream execution failed:
    if (__state.__status_ != cudaSuccess)
    {
      cuda::__throw_cuda_error(__state.__status_, "stream execution failed");
    }

    // return the result from the temp storage:
    return static_cast<_CUDA_VSTD::optional<__values_t>&&>(__state.__value_);
  }
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_STREAM_SYNC_WAIT
