//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_ADAPTOR
#define __CUDAX_EXECUTION_STREAM_ADAPTOR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__launch/launch.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <nv/target>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

// This header provides a sender adaptor that adapts a non-stream sender to a stream
// sender. The adaptor does several things:
//
// 1. It ensures that the stream_domain is used for sender transformations.
// 2. It takes the launch configuration from the child sender and puts it into the
//    environment of the inner receiver used to connect the child sender.
// 3. It connects the child sender to the inner receiver, which will write the child's
//    results into a variant that is in managed memory.
// 4. It creates the child operation state in managed memory.
// 5. It starts the child operation on the host, which causes the predecessor kernels to
//    be launched in order.
// 6. It launches a completion kernel on the stream to read the results out of the variant
//    and send them to the outer receiver. The launch configuration is read from the outer
//    receiver's environment.

namespace cuda::experimental::execution
{
namespace __stream
{
template <class _Rcvr>
struct __completion_fn
{
  template <class _Tag, class... _Args>
  _CCCL_API void operator()(_Tag, _Args&&... __args) const noexcept
  {
    _Tag{}(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Args&&>(__args)...);
  }

  _Rcvr& __rcvr_;
};

template <class _Rcvr>
struct __results_visitor
{
  template <class _Tuple>
  _CCCL_API void operator()(_Tuple&& __tuple) const noexcept
  {
    ::cuda::std::__apply(__completion_fn<_Rcvr>{__rcvr_}, static_cast<_Tuple&&>(__tuple));
  }

  _Rcvr& __rcvr_;
};

// __state_t lives in managed memory. It stores everything the operation state needs,
// besides the child operation state.
template <class _Rcvr, class _Variant>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_base_t
{
  _Rcvr __rcvr_;
  _Variant __results_;
  bool __complete_inline_;
};

template <class _Rcvr, class _Config, class _Variant>
struct __state_t : __state_base_t<_Rcvr, _Variant>
{
  _Config __launch_config_;
};

// remove any exception_ptr error completion from the completion signatures, and replace it
// with a cudaError_t error completion.
template <class _Completions>
_CCCL_API constexpr auto __with_cuda_error(_Completions __completions) noexcept
{
  return __completions - __eptr_completion() + completion_signatures<set_error_t(cudaError_t)>{};
}

template <class _Config>
using __dims_of_t = decltype(_Config::dims);

// This kernel forwards the results from the child sender to the receiver of the parent
// sender. The receiver is where most algorithms do their work, so we want the receiver to
// tell us how to launch the kernel that completes it. Thus, the launch configuration is
// read from the outer receiver's environment.
template <int _BlockThreads, class _Rcvr, class _Variant>
_CCCL_VISIBILITY_HIDDEN __launch_bounds__(_BlockThreads) __global__
  void __completion_kernel(__state_base_t<_Rcvr, _Variant>* __state)
{
  _CCCL_ASSERT(__state->__results_.__index() != __npos, "__completion_kernel called with empty results");
  _Variant::__visit(__results_visitor<_Rcvr>{__state->__rcvr_}, __state->__results_);
}

// This is the environment of the inner receiver that is used to connect the child sender.
template <class _Env, class _Config>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
{
  _CCCL_TEMPLATE(class _Query, class... _As)
  _CCCL_REQUIRES(__queryable_with<_Env, _Query, _As...>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Query, _As&&... __args) const
    noexcept(__nothrow_queryable_with<_Env, _Query, _As...>) -> __query_result_t<_Env, _Query, _As...>
  {
    return __env_.query(_Query{}, static_cast<_As&&>(__args)...);
  }

  // This query is used to tell transform_sender that the child sender has been adapted to
  // the stream domain.
  [[nodiscard]] _CCCL_API constexpr auto query(__stream::__adapted_t) const noexcept -> ::cuda::std::__ignore_t
  {
    return {};
  }

  [[nodiscard]] _CCCL_API constexpr auto query(get_launch_config_t) const noexcept -> _Config
  {
    return __launch_config_;
  }

  _Env __env_;
  _CCCL_NO_UNIQUE_ADDRESS _Config __launch_config_;
};

// This is the inner receiver that is used to connect the child sender.
template <class _Rcvr, class _Config, class _Variant>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
{
  template <class _Tag, class... _Args>
  _CCCL_API void __complete(_Tag, _Args&&... __args) noexcept
  {
    if (__state_->__complete_inline_) // TODO: untested
    {
      _Tag{}(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Args&&>(__args)...);
    }
    else
    {
      using __tuple_t = ::cuda::std::__decayed_tuple<_Tag, _Args...>;
      __state_->__results_.template __emplace<__tuple_t>(_Tag{}, static_cast<_Args&&>(__args)...);
    }
  }

  template <class... _Args>
  _CCCL_NODEBUG_API constexpr void set_value(_Args&&... __args) noexcept
  {
    __complete(execution::set_value, static_cast<_Args&&>(__args)...);
  }

  template <class _Error>
  _CCCL_NODEBUG_API constexpr void set_error(_Error&& __err) noexcept
  {
    // Map any exception_ptr error completions to cudaErrorUnknown:
    if constexpr (__same_as<::cuda::std::remove_cvref_t<_Error>, ::std::exception_ptr>)
    {
      __complete(execution::set_error, cudaErrorUnknown);
    }
    else
    {
      __complete(execution::set_error, static_cast<_Error&&>(__err));
    }
  }

  _CCCL_NODEBUG_API constexpr void set_stopped() noexcept
  {
    __complete(execution::set_stopped);
  }

  _CCCL_API constexpr auto get_env() const noexcept -> __env_t<env_of_t<_Rcvr>, _Config>
  {
    return {execution::get_env(__state_->__rcvr_), __state_->__launch_config_};
  }

  __state_t<_Rcvr, _Config, _Variant>* __state_;
};

template <class _CvSndr, class _Rcvr, class _GetStream>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
{
  using operation_state_concept = operation_state_t;

  _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _GetStream __get_stream)
      : __stream_{__get_stream(__sndr, execution::get_env(__rcvr))}
  {
    NV_IF_TARGET(NV_IS_HOST,
                 (__host_make_state(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr));),
                 (__device_make_state(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr));));
  }

  _CCCL_IMMOVABLE(__opstate_t);

  _CCCL_API constexpr void start() noexcept
  {
    NV_IF_TARGET(NV_IS_HOST, (__host_start();), (__device_start();));
  }

  // This is called by the continues_on adaptor after it has sync'ed the stream.
  template <class _Rcvr2>
  _CCCL_HOST_API auto __set_results(_Rcvr2& __rcvr) noexcept
  {
    __results_t::__visit(__results_visitor<_Rcvr2&>{__rcvr}, __get_state().__state_.__results_);
  }

private:
  using __sndr_config_t       = __call_result_t<get_launch_config_t, env_of_t<_CvSndr>>;
  using __rcvr_config_t       = __call_result_t<get_launch_config_t, env_of_t<_Rcvr>>;
  using __env_t               = __stream::__env_t<env_of_t<_Rcvr>, __sndr_config_t>;
  using __child_completions_t = completion_signatures_of_t<_CvSndr, __env_t>;
  using __completions_t       = decltype(__stream::__with_cuda_error(__child_completions_t{}));
  using __results_t = typename __completions_t::template __transform_q<::cuda::std::__decayed_tuple, __variant>;
  using __rcvr_t    = __stream::__rcvr_t<_Rcvr, __sndr_config_t, __results_t>;

  _CCCL_HOST_API void __host_make_state(_CvSndr&& __sndr, _Rcvr __rcvr)
  {
    // If *this is already in device or managed memory, then we can avoid a separate
    // allocation.
    if (auto const __attrs = execution::__get_pointer_attributes(this); __attrs.type == ::cudaMemoryTypeManaged)
    {
      __state_.template __emplace<__state_t>(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr));
    }
    else
    {
      __state_.__emplace(
        __managed_box<__state_t>::__make_unique(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr)));
    }
  }

  _CCCL_DEVICE_API void __device_make_state(_CvSndr&& __sndr, _Rcvr __rcvr)
  {
    __state_.template __emplace<__state_t>(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr));
  }

  _CCCL_HOST_API void __host_start() noexcept
  {
    auto& __state = __get_state();

    // Read the launch configuration passed to us by the parent operation. When we launch
    // the completion kernel, we will be completing the parent's receiver, so we must let
    // the receiver tell us how to launch the kernel.
    auto const __launch_config    = get_launch_config(execution::get_env(__state.__state_.__rcvr_));
    using __launch_dims_t         = decltype(__launch_config.dims);
    constexpr int __block_threads = __launch_dims_t::static_count(experimental::thread, experimental::block);
    int const __grid_blocks       = __launch_config.dims.count(experimental::block, experimental::grid);
    static_assert(__block_threads != ::cuda::std::dynamic_extent);

    // Start the child operation state. This will launch kernels for all the predecessors
    // of this operation.
    execution::start(__state.__opstate_);

    // printf("Launching completion kernel for %s with %d block threads and %d grid blocks\n",
    //        __name,
    //        __block_threads,
    //        __grid_blocks);

    // launch a kernel to pass the results to the receiver.
    __completion_kernel<__block_threads><<<__grid_blocks, __block_threads, 0, __stream_.get()>>>(&__state.__state_);

    // Check for errors in the kernel launch.
    if (auto __status = cudaGetLastError(); __status != cudaSuccess)
    {
      execution::set_error(static_cast<_Rcvr&&>(__state.__state_.__rcvr_), cudaError_t(__status));
    }
  }

  // TODO: untested
  _CCCL_DEVICE_API void __device_start() noexcept
  {
    using __launch_dims_t         = __dims_of_t<__rcvr_config_t>;
    constexpr int __block_threads = __launch_dims_t::static_count(experimental::thread, experimental::block);
    auto& __state                 = __get_state();

    // without the following, the kernel in __host_start will fail to launch with
    // cudaErrorInvalidDeviceFunction.
    ::__cccl_unused(&__completion_kernel<__block_threads, _Rcvr, __results_t>);
    __state.__state_.__complete_inline_ = true;
    execution::start(__state.__opstate_);
  }

  // This is the part of the operation state that is stored in managed memory.
  struct __state_t
  {
    _CCCL_HOST_API constexpr explicit __state_t(_CvSndr&& __sndr, _Rcvr __rcvr)
        : __state_{{static_cast<_Rcvr&&>(__rcvr), {}, false}, get_launch_config(execution::get_env(__sndr))}
        , __opstate_(execution::connect(static_cast<_CvSndr&&>(__sndr), __rcvr_t{&__state_}))
    {}

    __stream::__state_t<_Rcvr, __sndr_config_t, __results_t> __state_;
    connect_result_t<_CvSndr, __rcvr_t> __opstate_;
  };

  // Return a reference to the state for this operation, whether it is stored in-situ or
  // in dyncamically-allocated managed memory.
  [[nodiscard]] _CCCL_API constexpr auto __get_state() noexcept -> __state_t&
  {
    return __state_.__index() == 0 ? __state_.template __get<0>() : __state_.template __get<1>()->__value;
  }

  stream_ref __stream_;
  __variant<__state_t, ::cuda::std::unique_ptr<__managed_box<__state_t>>> __state_{};
};

template <class _Sndr, class _GetStream>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

template <class _Sndr, class _GetStream>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
{
  // This makes sure that when `connect` calls `transform_sender`, it will use the stream
  // domain to find a customization.
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(get_domain_override_t) const noexcept -> stream_domain
  {
    return {};
  }

  // If the child sender knows how to provide a stream, make it available via the stream
  // adapter's attributes.
  _CCCL_TEMPLATE(class _GetStream2 = _GetStream)
  _CCCL_REQUIRES(__callable<_GetStream2, _Sndr, env<>>)
  [[nodiscard]] _CCCL_API constexpr auto query(get_stream_t) const noexcept -> stream_ref
  {
    return __sndr_.__get_stream_(__sndr_.__sndr_, env{});
  }

  // This sender executes asynchronously with respect to 'start()':
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t) const noexcept
  {
    return completion_behavior::asynchronous;
  }

  // This forwards even non-forwarding queries. A stream sender adaptor is not an ordinary
  // sender adaptor, like `then` or `let_value`. A stream sender adaptor is an
  // implementation detail that is not visible to the user. It should be as transparent as
  // possible.
  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__queryable_with<env_of_t<_Sndr>, _Query>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>)
    -> __query_result_t<env_of_t<_Sndr>, _Query>
  {
    return execution::get_env(__sndr_.__sndr_).query(_Query{});
  }

  const __sndr_t<_Sndr, _GetStream>& __sndr_;
};

// This is the sender adaptor that adapts a non-stream sender to a stream sender.
template <class _Sndr, class _GetStream>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    using __cv_sndr_t _CCCL_NODEBUG_ALIAS     = ::cuda::std::__copy_cvref_t<_Self, _Sndr>;
    using __sndr_config_t _CCCL_NODEBUG_ALIAS = __call_result_t<get_launch_config_t, env_of_t<_Sndr>>;
    using __env_t                             = __stream::__env_t<_Env, __sndr_config_t>;
    _CUDAX_LET_COMPLETIONS(auto(__completions) = execution::get_completion_signatures<__cv_sndr_t, __env_t>())
    {
      return __stream::__with_cuda_error(__completions);
    }
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Rcvr, _GetStream>
  {
    return __opstate_t<_Sndr, _Rcvr, _GetStream>(
      static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr), __get_stream_);
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Rcvr, _GetStream>
  {
    return __opstate_t<const _Sndr&, _Rcvr, _GetStream>(__sndr_, static_cast<_Rcvr&&>(__rcvr), __get_stream_);
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t<_Sndr, _GetStream>
  {
    return __attrs_t<_Sndr, _GetStream>{*this};
  }

  _CCCL_NO_UNIQUE_ADDRESS __tag_t<__stream::__tag_of_t<_Sndr>> __tag_;
  _CCCL_NO_UNIQUE_ADDRESS _GetStream __get_stream_;
  _Sndr __sndr_;
};

template <class _Sndr, class _GetStream>
_CCCL_API constexpr auto __adapt(_Sndr&& __sndr, _GetStream __get_stream) noexcept(__nothrow_decay_copyable<_Sndr>)
{
  return __sndr_t<_Sndr, _GetStream>{{}, __get_stream, static_cast<_Sndr&&>(__sndr)};
}
} // namespace __stream

template <class _Sndr, class _GetStream>
inline constexpr size_t structured_binding_size<__stream::__sndr_t<_Sndr, _GetStream>> = 3;

} // namespace cuda::experimental::execution

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_ADAPTOR
