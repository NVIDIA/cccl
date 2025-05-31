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

#include <cuda/atomic>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__launch/launch.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <nv/target>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::execution::__stream
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
    _CUDA_VSTD::__apply(__completion_fn<_Rcvr>{__rcvr_}, static_cast<_Tuple&&>(__tuple));
  }

  _Rcvr& __rcvr_;
};

template <class _Rcvr, class _Variant>
struct __state_t
{
  _Rcvr __rcvr_;
  _Variant __results_;
  stream_ref __stream_;
  bool __complete_inline_;
};

template <class _Completions>
_CCCL_API static constexpr auto __with_cuda_error(_Completions __completions) noexcept
{
  return __completions - __eptr_completion() + completion_signatures<set_error_t(cudaError_t)>{};
}

template <class _Rcvr, class _Variant>
__launch_bounds__(1) __global__ void __host_complete_fn(__state_t<_Rcvr, _Variant>* __state)
{
  _Variant::__visit(__results_visitor<_Rcvr>{__state->__rcvr_}, __state->__results_);
}

template <class _Env>
struct __env_t
{
  template <class _Query>
  [[nodiscard]] _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<_Env, _Query>)
    -> __query_result_t<_Env, _Query>
  {
    return __env_.query(_Query{});
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t) noexcept -> default_domain
  {
    return default_domain{};
  }

  _Env __env_;
};

template <class _Rcvr, class _Variant>
struct __rcvr_t
{
  template <class _Tag, class... _Args>
  _CCCL_API void __complete(_Tag, _Args&&... __args) noexcept
  {
    if (__state_->__complete_inline_)
    {
      _Tag{}(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Args&&>(__args)...);
    }
    else
    {
      using __tuple_t = _CUDA_VSTD::__decayed_tuple<_Tag, _Args...>;
      __state_->__results_.template __emplace<__tuple_t>(_Tag{}, static_cast<_Args&&>(__args)...);
    }
  }

  template <class... _Args>
  _CCCL_TRIVIAL_API void set_value(_Args&&... __args) noexcept
  {
    __complete(execution::set_value, static_cast<_Args&&>(__args)...);
  }

  template <class _Error>
  _CCCL_TRIVIAL_API void set_error(_Error&& __err) noexcept
  {
    if constexpr (_CUDA_VSTD::is_same_v<_CUDA_VSTD::remove_cvref_t<_Error>, ::std::exception_ptr>)
    {
      __complete(execution::set_error, cudaErrorUnknown);
    }
    else
    {
      __complete(execution::set_error, static_cast<_Error&&>(__err));
    }
  }

  _CCCL_TRIVIAL_API void set_stopped() noexcept
  {
    __complete(execution::set_stopped);
  }

  _CCCL_API auto get_env() const noexcept -> __env_t<env_of_t<_Rcvr>>
  {
    return {execution::get_env(__state_->__rcvr_)};
  }

  __state_t<_Rcvr, _Variant>* __state_;
};

template <class _Sndr>
_CCCL_HOST_API auto __bulk_launch_config(const _Sndr& __sndr) noexcept
{
  constexpr int __block             = 256;
  auto&& [__tag, __params, __child] = __sndr;
  auto&& [__shape, __fn]            = __params;
  const int __grid                  = (static_cast<int>(__shape) + __block - 1) / __block;
  return experimental::make_config(block_dims<__block>, grid_dims(__grid));
}

template <class _CvSndr, class _Rcvr>
struct __opstate_t
{
  using operation_state_concept = operation_state_t;

  _CCCL_API explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, stream_ref __stream)
      : __launch_config_{get_launch_config(get_env(__sndr))}
  {
    NV_IF_TARGET(NV_IS_HOST,
                 (__host_make_state(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __stream);),
                 (__device_make_state(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __stream);));
  }

  _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

  _CCCL_API void start() noexcept
  {
    NV_IF_TARGET(NV_IS_HOST, (__host_start();), (__device_start();));
  }

private:
  using __child_completions_t = completion_signatures_of_t<_CvSndr, __env_t<env_of_t<_Rcvr>>>;
  using __completions_t       = decltype(__stream::__with_cuda_error(__child_completions_t{}));
  using __results_t = typename __completions_t::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;

  _CCCL_HOST_API void __host_make_state(_CvSndr&& __sndr, _Rcvr __rcvr, stream_ref __stream)
  {
    // If *this is already in device or managed memory, then we can avoid a separate
    // allocation.
    if (auto const __attrs = execution::__get_pointer_attributes(this); __attrs.type == ::cudaMemoryTypeManaged)
    {
      __state_.template __emplace<__state_t>(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __stream);
    }
    else
    {
      __state_.__emplace(__managed_box<__state_t>::__make_unique(
        static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __stream));
    }
  }

  _CCCL_DEVICE_API void __device_make_state(_CvSndr&& __sndr, _Rcvr __rcvr, stream_ref __stream)
  {
    __state_.template __emplace<__state_t>(static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __stream);
  }

  _CCCL_HOST_API void __host_start() noexcept
  {
    auto& __state       = __get_state();
    auto const __stream = __state.__state_.__stream_.get();

    _CCCL_ASSERT(execution::__get_pointer_attributes(&__state.__state_).type == ::cudaMemoryTypeManaged,
                 "stream scheduler's operation state must be allocated in managed memory");
    // start the child operation state on the host and launch a kernel to pass the results
    // to the receiver.
    execution::start(__state.__opstate_);
    auto __status =
      __detail::launch_impl(__stream, __launch_config_, &__host_complete_fn<_Rcvr, __results_t>, &__state.__state_);
    if (__status != cudaSuccess)
    {
      execution::set_error(static_cast<_Rcvr&&>(__state.__state_.__rcvr_), __status);
    }
  }

  _CCCL_DEVICE_API void __device_start() noexcept
  {
    [[maybe_unused]] auto* const __complete_fn = &__host_complete_fn<_Rcvr, __results_t>;
    auto& __state                              = __get_state();
    __state.__state_.__complete_inline_        = true;
    execution::start(__state.__opstate_);
  }

  struct __state_t
  {
    _CCCL_HOST_API explicit __state_t(_CvSndr&& __sndr, _Rcvr __rcvr, stream_ref __stream)
        : __state_{static_cast<_Rcvr&&>(__rcvr), {}, __stream, false}
        , __opstate_(connect(static_cast<_CvSndr&&>(__sndr), __rcvr_t<_Rcvr, __results_t>{&__state_}))
    {}

    __stream::__state_t<_Rcvr, __results_t> __state_;
    connect_result_t<_CvSndr, __rcvr_t<_Rcvr, __results_t>> __opstate_;
  };

  // Return a reference to the state for this operation, whether it is stored in-situ or
  // in dyncamically-allocated managed memory.
  _CCCL_API auto __get_state() noexcept -> __state_t&
  {
    return __state_.__index() == 0 ? __state_.template __get<0>() : __state_.template __get<1>()->__value;
  }

  using __launch_config_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__call_result_t<get_launch_config_t, env_of_t<_CvSndr>>;
  __launch_config_t __launch_config_{};
  __variant<__state_t, _CUDA_VSTD::unique_ptr<__managed_box<__state_t>>> __state_{};
};

template <class _Sndr>
struct __attrs_t
{
  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(get_domain_late_t) noexcept -> stream_domain
  {
    return {};
  }

  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__queryable_with<env_of_t<_Sndr>, _Query>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>)
    -> __query_result_t<env_of_t<_Sndr>, _Query>
  {
    return execution::get_env(*__sndr_).query(_Query{});
  }

  const _Sndr* __sndr_;
};

template <class _Sndr>
struct __sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    using __cv_sndr_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__copy_cvref_t<_Self, _Sndr>;
    _CUDAX_LET_COMPLETIONS(auto(__completions) = execution::get_completion_signatures<__cv_sndr_t, __env_t<_Env>>())
    {
      return __with_cuda_error(__completions);
    }
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Rcvr>
  {
    return __opstate_t<_Sndr, _Rcvr>(
      static_cast<_Sndr&&>(__state_.__sndr_), static_cast<_Rcvr&&>(__rcvr), __state_.__stream_);
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Rcvr>
  {
    return __opstate_t<const _Sndr&, _Rcvr>(__state_.__sndr_, static_cast<_Rcvr&&>(__rcvr), __state_.__stream_);
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __attrs_t<_Sndr>
  {
    return __attrs_t<_Sndr>{&__state_.__sndr_};
  }

  // By having just one data member, this sender does not look like one that can be
  // introspected, transformed, or visited.
  struct __state_t
  {
    stream_ref __stream_;
    _Sndr __sndr_;
  } __state_;
};

template <class _Sndr>
_CCCL_API constexpr auto __adapt(_Sndr __sndr, stream_ref __stream) -> decltype(auto)
{
  return __sndr_t<_Sndr>{{__stream, static_cast<_Sndr&&>(__sndr)}};
}

template <class _Sndr>
_CCCL_API constexpr auto __adapt(_Sndr __sndr) -> decltype(auto)
{
  return __stream::__adapt(static_cast<_Sndr&&>(__sndr), get_stream(get_env(__sndr)));
}

template <class _Sndr>
_CCCL_API void __adapt(__sndr_t<_Sndr>, stream_ref) = delete;
} // namespace cuda::experimental::execution::__stream

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_ADAPTOR
