//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_STREAM_CONTEXT
#define __CUDAX_ASYNC_DETAIL_STREAM_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/optional>
#include <cuda/std/tuple>

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/domain.cuh>
#include <cuda/experimental/__async/sender/rcvr_ref.cuh>
#include <cuda/experimental/__async/sender/storage_registry.cuh>
#include <cuda/experimental/__async/sender/sync_wait.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/stream.cuh>

#include <new> // IWYU pragma: keep

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
//////////////////////////////////////////////////////////////////////////////////////////
// stream domain
struct stream_domain : default_domain
{
  template <class _Tag>
  static constexpr auto __apply(_Tag) noexcept
  {
    return default_domain::__apply(_Tag());
  }

  static constexpr auto __apply(sync_wait_t) noexcept;

  _CUDAX_SEMI_PRIVATE :
  struct __sync_wait_t;
};

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
__global__ static void __stream_invoke_r(__call_result_t<_Fn, _Args...>* __return, _Fn __fn, _Args... __args)
{
  using _Return = __call_result_t<_Fn, _Args...>;
  ::new (static_cast<void*>(__return)) _Return(static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...));
}

//////////////////////////////////////////////////////////////////////////////////////////
// queries regarding the stream context's storage registry
struct get_storage_registry_t
{
  template <class _Env>
  _CUDAX_API auto operator()(const _Env& __env) const noexcept -> __storage_registry
  {
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_storage_registry_t get_storage_registry{};

struct get_storage_registry_context_t
{
  template <class _Env>
  _CUDAX_API auto operator()(const _Env& __env) const noexcept -> const __storage_registry_context&
  {
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_storage_registry_context_t get_storage_registry_context{};

//////////////////////////////////////////////////////////////////////////////////////////
// stream_context
struct _CCCL_TYPE_VISIBILITY_DEFAULT stream_context : private __immovable
{
  stream_context() noexcept = default;

  _CUDAX_HOST_API void sync() noexcept
  {
    __stream_.sync();
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler
  struct _CCCL_TYPE_VISIBILITY_DEFAULT scheduler
  {
    using scheduler_concept = scheduler_t;

    _CUDAX_API bool operator==(const scheduler& __other) const noexcept
    {
      return __stream_ref_ == __other.__stream_ref_;
    }

    _CUDAX_API static constexpr auto query(get_forward_progress_guarantee_t) noexcept
    {
      return forward_progress_guarantee::weakly_parallel;
    }

    _CUDAX_API static constexpr auto query(get_domain_t) noexcept
    {
      return stream_domain{};
    }

    _CUDAX_API constexpr auto query(get_stream_t) const noexcept
    {
      return __stream_ref_;
    }

    _CUDAX_API auto schedule() const noexcept
    {
      return __sndr_t{{__stream_ref_}};
    }

    stream_ref __stream_ref_;
  };

  _CUDAX_TRIVIAL_HOST_API auto get_scheduler() noexcept -> scheduler
  {
    return scheduler{__stream_};
  }

  _CUDAX_SEMI_PRIVATE :
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __complete_fn
  {
    template <class _Tag, class _Rcvr, class... _Args>
    _CUDAX_TRIVIAL_API void operator()(_Tag, _Rcvr&& __rcvr, _Args&&... __args) const noexcept
    {
      _Tag{}(static_cast<_Rcvr&&>(__rcvr), static_cast<_Args&&>(__args)...);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // environment of the stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    _CUDAX_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept
    {
      return scheduler{__stream_ref_};
    }

    _CUDAX_API constexpr auto query(get_stream_t) const noexcept
    {
      return __stream_ref_;
    }

    _CUDAX_TRIVIAL_API static constexpr auto query(get_domain_t) noexcept
    {
      return stream_domain{};
    }

    stream_ref __stream_ref_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's operation state
  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : private __immovable
  {
    using operation_state_concept = operation_state_t;

    _CUDAX_API __opstate_t(_Rcvr __rcvr, stream_ref __stream_ref) noexcept
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __stream_ref_{__stream_ref}
    {}

    _CUDAX_API void start() & noexcept
    {
      NV_IF_TARGET(NV_IS_DEVICE, (__device_start();), (__host_start();));
    }

  private:
    _CUDAX_HOST_API void __host_start() noexcept
    {
      __stream_invoke<<<1, 1, 0, __stream_ref_.get()>>>(
        __complete_fn{}, set_value, __rcvr_ref{_CUDA_VSTD::addressof(__rcvr_)});
      if (auto __status = cudaGetLastError(); __status != cudaSuccess)
      {
        set_error(static_cast<_Rcvr&&>(__rcvr_), __status);
      }
    }

    _CUDAX_DEVICE_API void __device_start() noexcept
    {
      // without the following, the kernel in __host_start will fail to launch with
      // cudaErrorInvalidDeviceFunction.
      [[maybe_unused]] auto __ignore = &__stream_invoke<__complete_fn, set_value_t, __rcvr_ref<_Rcvr>>;
      set_value(static_cast<_Rcvr&&>(__rcvr_));
    }

    _Rcvr __rcvr_;
    stream_ref __stream_ref_;
  };

  ////////////////////////////////////////////////////////////////////////////////////////
  // stream scheduler's sender
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    template <class _Self>
    _CUDAX_API static constexpr auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t(), set_error_t(cudaError_t)>{};
    }

    _CUDAX_API constexpr auto get_env() const noexcept -> __env_t const&
    {
      return __env_;
    }

    template <class _Rcvr>
    _CUDAX_API auto connect(_Rcvr __rcvr) const noexcept
    {
      return __opstate_t<_Rcvr>{static_cast<_Rcvr&&>(__rcvr), __env_.__stream_ref_};
    }

    __env_t __env_;
  };

  stream __stream_{};
};

using stream_scheduler = stream_context::scheduler;

/////////////////////////////////////////////////////////////////////////////////
// wait: customization for the stream scheduler
struct stream_domain::__sync_wait_t
{
  _CUDAX_SEMI_PRIVATE :
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    stream_ref __stream_;
    __storage_registry_context* __stg_context_; // lives on the host stack
    __storage_registry __stg_;                  // lives in managed memory ultimately
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
    _CUDAX_API auto query(get_storage_registry_t) const noexcept
    {
      return __state_->__stg_;
    }

    _CUDAX_API auto query(get_storage_registry_context_t) const noexcept -> decltype(auto)
    {
      return (*__state_->__stg_context_);
    }

    _CUDAX_API auto query(get_stream_t) const noexcept
    {
      return __state_->__stream_;
    }

    _CUDAX_API auto query(get_scheduler_t) const noexcept
    {
      return stream_context::scheduler{__state_->__stream_};
    }

    _CUDAX_API auto query(get_delegation_scheduler_t) const noexcept
    {
      return __state_->__loop_.get_scheduler();
    }

    __state_t* __state_{nullptr};
  };

  _CUDAX_API static auto __to_cuda_error(cudaError_t __status) noexcept
  {
    _CCCL_ASSERT(__status != cudaSuccess, "cudaSuccess is not a valid error completion");
    return __status;
  }

  _CUDAX_API static auto __to_cuda_error(__ignore) noexcept
  {
    return cudaErrorUnknown;
  }

  template <class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _Args>
    _CUDAX_API void set_value(_Args&&... __args) && noexcept
    {
      auto& __state = *static_cast<__state_ex_t<_Values>*>(__env_.__state_);
      _CUDAX_TRY( //
        ({        //
          __state.__value_.emplace(static_cast<_Args&&>(__args)...);
        }),
        _CUDAX_CATCH(...) //
        ({                //
          __state.__status_ = cudaErrorUnknown;
        }))
      __state.__loop_.finish();
    }

    template <class _Error>
    _CUDAX_API void set_error(_Error&& __err) && noexcept
    {
      __env_.__state_->__status_ = __to_cuda_error(static_cast<_Error&&>(__err));
      __env_.__state_->__loop_.finish();
    }

    _CUDAX_API void set_stopped() && noexcept
    {
      __env_.__state_->__loop_.finish();
    }

    _CUDAX_TRIVIAL_API auto get_env() const noexcept
    {
      return __env_;
    }

    __env_t __env_{};
  };

  template <class _Values>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __mk_state
  {
    _CUDAX_API auto
    operator()(stream_ref __stream, __storage_registry_context* __context, __storage_registry __stg) noexcept
      -> __state_ex_t<_Values>
    {
      return __state_ex_t<_Values>{{__stream, __context, __stg}};
    }
  };

public:
  template <class _Sndr>
  _CUDAX_HOST_API auto operator()(_Sndr&& __sndr) const
  {
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
    __stream.sync();       // Wait for the stream to finish:

    // Throw if the stream execution failed:
    if (__state.__status_ != cudaSuccess)
    {
      throw cuda_error(__state.__status_, "stream execution failed");
    }

    // return the result from the temp storage:
    return static_cast<_CUDA_VSTD::optional<__values_t>&&>(__state.__value_);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////
// sync_wait, customized for the stream_context
constexpr auto stream_domain::__apply(sync_wait_t) noexcept
{
  return stream_domain::__sync_wait_t{};
}

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_STREAM_CONTEXT
