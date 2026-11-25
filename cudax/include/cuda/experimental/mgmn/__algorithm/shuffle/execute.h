// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_EXECUTE_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_EXECUTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

#include <cub/device/device_scan.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/__container/resizable_buffer.h>
#include <cuda/__container/simple_vector.h>
#include <cuda/__launch/configuration.h>
#include <cuda/__launch/launch.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cuda/experimental/mgmn/__algorithm/common.h>
#include <cuda/experimental/mgmn/__algorithm/shuffle/kernels.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__hashgraph_shuffle
{
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API ::cuda::__simple_vector<_Tp> __host_zeros(::cuda::std::size_t __size)
{
  ::cuda::__simple_vector<_Tp> __result{__size, ::cuda::no_init};
  for (::cuda::std::size_t __i = 0; __i < __size; ++__i)
  {
    __result.emplace_back(0);
  }
  return __result;
}

template <class _Buffer>
[[nodiscard]] _CCCL_HOST_API auto __read_metadata(const _Buffer& __buffer)
{
  // Buffer element and allocator types depend on the caller's execution environment.
  using _Tp   = typename _Buffer::value_type;
  auto __host = ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<_Tp>(__buffer.size());
  ::cuda::copy_bytes(
    __buffer.stream(),
    __buffer,
    __host,
    ::cuda::copy_configuration{
      __buffer.stream().device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});
  __buffer.stream().sync();
  return __host;
}

template <class _Buffer>
_CCCL_HOST_API void __clear(_Buffer& __buffer)
{
  __CUDAX_MULTI_GPU_DISPATCH(
    __buffer.stream(),
    cudaMemsetAsync,
    __buffer.data(),
    0,
    __buffer.size() * sizeof(typename _Buffer::value_type),
    __buffer.stream().get());
}

template <class _Kernel, class... _Args>
_CCCL_HOST_API void __launch(::cuda::stream_ref __stream, __index __count, _Kernel __kernel, _Args... __args)
{
  if (__count != 0)
  {
    const auto __blocks = static_cast<unsigned>(::cuda::std::min(__index{4096}, (__count - 1) / 256 + 1));
    const auto __config = ::cuda::make_config(::cuda::grid_dims(__blocks), ::cuda::block_dims<256>());
    ::cuda::launch(__stream, __config, __kernel, __args...);
  }
}

template <class _Tp, class _Env>
struct __rank_state
{
  using __resource = ::cuda::experimental::mgmn::__detail::__resource_type_for<_Env>;
  template <class _Up>
  using __buffer = typename __resource::default_queries::template rebind<::cuda::__resizable_buffer, _Up>;

  template <class _Up>
  [[nodiscard]] _CCCL_HOST_API static __buffer<_Up> __make(__index __size, const _Env& __env)
  {
    const auto __stream = ::cuda::get_stream(__env);
    return __buffer<_Up>{
      __stream,
      ::cuda::experimental::mgmn::__detail::__resource_from_env(__env, __stream.__logical_device()),
      __size,
      ::cuda::no_init,
      ::cuda::experimental::mgmn::__detail::__sanitize_buffer_env(__env)};
  }

  __buffer<__index> __sizes_;
  __buffer<__index> __coarse_;
  __buffer<__index> __global_coarse_;
  __buffer<__index> __splits_;
  __buffer<__index> __counts_;
  __buffer<__index> __offsets_;
  __buffer<__index> __cursors_;
  __buffer<__index> __destinations_;
  __buffer<__index> __recv_counts_;
  __buffer<__record<_Tp>> __input_;
  __buffer<__record<_Tp>> __packed_;
  __buffer<__record<_Tp>> __received_;
  __buffer<__record<_Tp>> __lrb_;
  __buffer<__record<_Tp>> __ordered_;
  ::cuda::__simple_vector<::cuda::std::size_t> __send_counts_;
  ::cuda::__simple_vector<::cuda::std::size_t> __send_offsets_;
  ::cuda::__simple_vector<::cuda::std::size_t> __receive_counts_;
  ::cuda::__simple_vector<::cuda::std::size_t> __receive_offsets_;
  ::cuda::__simple_vector<__index> __host_splits_;
  __index __start_ = 0;
  __index __end_   = 0;

  _CCCL_HOST_API __rank_state(__index __count, __index __ranks, __index __coarse_bins, const _Env& __env)
      : __sizes_(__make<__index>(__ranks, __env))
      , __coarse_(__make<__index>(__coarse_bins, __env))
      , __global_coarse_(__make<__index>(__coarse_bins, __env))
      , __splits_(__make<__index>(__ranks + 1, __env))
      , __counts_(__make<__index>(__ranks + 1, __env))
      , __offsets_(__make<__index>(__ranks + 1, __env))
      , __cursors_(__make<__index>(__ranks + 1, __env))
      , __destinations_(__make<__index>(__count, __env))
      , __recv_counts_(__make<__index>(__ranks, __env))
      , __input_(__make<__record<_Tp>>(__count, __env))
      , __packed_(__make<__record<_Tp>>(__count, __env))
      , __received_(__make<__record<_Tp>>(0, __env))
      , __lrb_(__make<__record<_Tp>>(0, __env))
      , __ordered_(__make<__record<_Tp>>(0, __env))
      , __send_counts_(
          ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<::cuda::std::size_t>(__ranks))
      , __send_offsets_(
          ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<::cuda::std::size_t>(__ranks))
      , __receive_counts_(
          ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<::cuda::std::size_t>(__ranks))
      , __receive_offsets_(
          ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<::cuda::std::size_t>(__ranks))
      , __host_splits_(::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<__index>(__ranks + 1))
  {}

  template <class _Bin>
  _CCCL_HOST_API void
  __build(const __buffer<__record<_Tp>>& __source,
          __buffer<__record<_Tp>>& __destination,
          __index __bins,
          _Bin __bin,
          const _Env& __env)
  {
    const auto __stream = __input_.stream();
    __counts_.resize_discard(__stream, __bins + 1, ::cuda::no_init);
    __offsets_.resize_discard(__stream, __bins + 1, ::cuda::no_init);
    __cursors_.resize_discard(__stream, __bins + 1, ::cuda::no_init);
    __destinations_.resize_discard(__stream, __source.size(), ::cuda::no_init);
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__clear(__counts_);
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__clear(__cursors_);
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__launch(
      __stream,
      __source.size(),
      __count<_Tp, _Bin>,
      __source.data(),
      __source.size(),
      __bin,
      __bins,
      __counts_.data(),
      __destinations_.data());
    __CUDAX_MULTI_GPU_DISPATCH(
      __stream, CUB_NS_QUALIFIER::DeviceScan::ExclusiveSum, __counts_.data(), __offsets_.data(), __bins + 1, __env);
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__launch(
      __stream,
      __source.size(),
      __pack<_Tp>,
      __source.data(),
      __destination.data(),
      __source.size(),
      __destinations_.data(),
      __offsets_.data(),
      __cursors_.data(),
      __bins);
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <class _CommRange, class _EnvRange, class _InputRange, class _SizeRange, class _OutputRange>
_CCCL_HOST_API void __execute(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __inputs,
  _SizeRange&& __sizes,
  _OutputRange&& __outputs,
  __index __seed)
{
  using _Tp    = ::cuda::std::iter_value_t<::cuda::std::ranges::range_value_t<_InputRange>>;
  using _Env   = ::cuda::std::ranges::range_value_t<_EnvRange>;
  using _State = __rank_state<_Tp, _Env>;
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>, "Shuffle payloads must be trivially copyable");
  static_assert(::cuda::std::is_integral_v<::cuda::std::ranges::range_value_t<_SizeRange>>,
                "Shuffle sizes must be integers");
  const auto __num_local = ::cuda::std::ranges::size(__comms);
  if (__num_local == 0)
  {
    return;
  }
  const __index __ranks       = ::cuda::std::ranges::begin(__comms)->size();
  const __index __coarse_bins = __ranks * 1024;
  ::cuda::__simple_vector<_State> __states{__num_local, ::cuda::no_init};
  for (auto&& [__env, __n] : ::cuda::std::ranges::views::zip(__envs, __sizes))
  {
    if constexpr (::cuda::std::is_signed_v<::cuda::std::remove_cvref_t<decltype(__n)>>)
    {
      _CCCL_VERIFY(__n >= 0, "Shuffle sizes must be nonnegative");
    }
    _CCCL_VERIFY(
      static_cast<__index>(__n) <= ::cuda::std::numeric_limits<::cuda::std::size_t>::max() / sizeof(__record<_Tp>),
      "Shuffle allocation size overflow");
    auto& __state = __states.emplace_back(static_cast<__index>(__n), __ranks, __coarse_bins, __env);
    // Initializing every slot avoids a host-to-device copy from a temporary scalar.
    __state.__sizes_ = typename _State::template __buffer<__index>{::cuda::make_buffer<__index>(
      __state.__sizes_.stream(),
      __state.__sizes_.memory_resource(),
      __ranks,
      static_cast<__index>(__n),
      ::cuda::experimental::mgmn::__detail::__sanitize_buffer_env(__env))};
  }
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    for (auto&& [__comm, __state] : ::cuda::std::ranges::views::zip(__comms, __states))
    {
      __comm.all_gather(
        __guard, __state.__sizes_.data() + __comm.rank(), __state.__sizes_.data(), 1, __state.__sizes_.stream());
    }
  }
  const auto __original_sizes =
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__read_metadata(__states[0].__sizes_);
  auto __desired = ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<__index>(__ranks + 1);
  for (__index __r = 0; __r < __ranks; ++__r)
  {
    _CCCL_VERIFY(__original_sizes[__r] <= ::cuda::std::numeric_limits<__index>::max() - __desired[__r],
                 "Shuffle global size overflow");
    __desired[__r + 1] = __desired[__r] + __original_sizes[__r];
  }
  const auto __total = __desired[__ranks];
  if (__total == 0)
  {
    return;
  }
  const auto __buckets      = ::cuda::std::max(__index{1}, ::cuda::std::min(__total / 8, __index{0xffffffffULL}));
  const auto __coarse_width = (__buckets - 1) / __coarse_bins + 1;
  for (auto&& [__comm, __state, __input] : ::cuda::std::ranges::views::zip(__comms, __states, __inputs))
  {
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__clear(__state.__coarse_);
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__launch(
      __state.__input_.stream(),
      __state.__input_.size(),
      __generate<::cuda::std::remove_cvref_t<decltype(__input)>, _Tp>,
      __input,
      __state.__input_.data(),
      __state.__input_.size(),
      __desired[__comm.rank()],
      __seed,
      __buckets,
      __coarse_width,
      __state.__coarse_.data());
  }
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    for (auto&& [__comm, __state] : ::cuda::std::ranges::views::zip(__comms, __states))
    {
      __comm.all_reduce(
        __guard,
        __state.__coarse_.data(),
        __state.__global_coarse_.data(),
        __coarse_bins,
        ::cuda::std::plus<>{},
        __state.__coarse_.stream());
    }
  }
  for (auto&& [__comm, __state, __env] : ::cuda::std::ranges::views::zip(__comms, __states, __envs))
  {
    const auto __histogram =
      ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__read_metadata(__state.__global_coarse_);
    auto __prefix = ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<__index>(__coarse_bins + 1);
    for (__index __i = 0; __i < __coarse_bins; ++__i)
    {
      __prefix[__i + 1] = __prefix[__i] + __histogram[__i];
    }
    auto& __splits = __state.__host_splits_;
    __index __bin  = 0;
    for (__index __r = 1; __r < __ranks; ++__r)
    {
      const auto __target = (__total / __ranks) * __r + ::cuda::std::min(__r, __total % __ranks);
      while (__bin < __coarse_bins && __prefix[__bin] < __target)
      {
        ++__bin;
      }
      __splits[__r] = __bin > __buckets / __coarse_width ? __buckets : __bin * __coarse_width;
    }
    __splits[__ranks] = __buckets;
    __state.__start_  = __splits[__comm.rank()];
    __state.__end_    = __splits[__comm.rank() + 1];
    ::cuda::copy_bytes(__state.__splits_.stream(),
                       __splits,
                       __state.__splits_,
                       ::cuda::copy_configuration{::cuda::host_memory_location, __state.__splits_.stream().device()});
    __state.__build(__state.__input_, __state.__packed_, __ranks, __owner{__state.__splits_.data(), __ranks}, __env);
  }
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    for (auto&& [__comm, __state] : ::cuda::std::ranges::views::zip(__comms, __states))
    {
      __comm.all_to_all(__guard, __state.__counts_.data(), __state.__recv_counts_.data(), 1, __state.__counts_.stream());
    }
  }
  for (auto&& [__state, __env] : ::cuda::std::ranges::views::zip(__states, __envs))
  {
    const auto __send = ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__read_metadata(__state.__counts_);
    const auto __recv =
      ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__read_metadata(__state.__recv_counts_);
    __index __received = 0;
    __index __sent     = 0;
    for (__index __r = 0; __r < __ranks; ++__r)
    {
      __state.__send_offsets_[__r]    = __sent;
      __state.__receive_offsets_[__r] = __received;
      __state.__send_counts_[__r]     = __send[__r];
      __state.__receive_counts_[__r]  = __recv[__r];
      __sent += __send[__r];
      __received += __recv[__r];
    }
    _CCCL_VERIFY(__received <= ::cuda::std::numeric_limits<::cuda::std::size_t>::max() / sizeof(__record<_Tp>),
                 "Shuffle receive allocation size overflow");
    const auto __stream = __state.__input_.stream();
    __state.__received_.resize_discard(__stream, __received, ::cuda::no_init);
    __state.__lrb_.resize_discard(__stream, __received, ::cuda::no_init);
    __state.__ordered_.resize_discard(__stream, __received, ::cuda::no_init);
  }
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    for (auto&& [__comm, __state] : ::cuda::std::ranges::views::zip(__comms, __states))
    {
      __comm.all_to_all_v(
        __guard,
        __state.__packed_.data(),
        __state.__send_counts_.data(),
        __state.__send_offsets_.data(),
        __state.__received_.data(),
        __state.__receive_counts_.data(),
        __state.__receive_offsets_.data(),
        __state.__input_.stream());
    }
  }
  for (auto&& [__state, __env] : ::cuda::std::ranges::views::zip(__states, __envs))
  {
    if (__state.__received_.size() != 0)
    {
      const auto __local_buckets = __state.__end_ - __state.__start_;
      _CCCL_VERIFY(__local_buckets != 0, "Received records for an empty hash range");
      const auto __lrb_bins = ::cuda::std::min(__index{16384}, __local_buckets);
      const auto __width    = (__local_buckets - 1) / __lrb_bins + 1;
      __state.__build(__state.__received_, __state.__lrb_, __lrb_bins, __local_bin{__state.__start_, __width}, __env);
      __state.__build(__state.__lrb_, __state.__ordered_, __local_buckets, __local_bin{__state.__start_, 1}, __env);
    }
    // Announce the actual CSR lengths before restoring the requested output layout.
    __state.__sizes_ = typename _State::template __buffer<__index>{::cuda::make_buffer<__index>(
      __state.__sizes_.stream(),
      __state.__sizes_.memory_resource(),
      __ranks,
      static_cast<__index>(__state.__received_.size()),
      ::cuda::experimental::mgmn::__detail::__sanitize_buffer_env(__env))};
  }
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    for (auto&& [__comm, __state] : ::cuda::std::ranges::views::zip(__comms, __states))
    {
      __comm.all_gather(
        __guard, __state.__sizes_.data() + __comm.rank(), __state.__sizes_.data(), 1, __state.__sizes_.stream());
    }
  }
  const auto __current_sizes =
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__read_metadata(__states[0].__sizes_);
  auto __current = ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__host_zeros<__index>(__ranks + 1);
  for (__index __r = 0; __r < __ranks; ++__r)
  {
    __current[__r + 1] = __current[__r] + __current_sizes[__r];
  }
  _CCCL_VERIFY(__current[__ranks] == __total, "HashGraph shuffle lost records");
  for (auto&& [__comm, __state] : ::cuda::std::ranges::views::zip(__comms, __states))
  {
    const auto __rank = __comm.rank();
    for (__index __peer = 0; __peer < __ranks; ++__peer)
    {
      const auto __send_begin            = ::cuda::std::max(__current[__rank], __desired[__peer]);
      const auto __send_end              = ::cuda::std::min(__current[__rank + 1], __desired[__peer + 1]);
      const auto __recv_begin            = ::cuda::std::max(__current[__peer], __desired[__rank]);
      const auto __recv_end              = ::cuda::std::min(__current[__peer + 1], __desired[__rank + 1]);
      __state.__send_counts_[__peer]     = __send_end > __send_begin ? __send_end - __send_begin : 0;
      __state.__receive_counts_[__peer]  = __recv_end > __recv_begin ? __recv_end - __recv_begin : 0;
      __state.__send_offsets_[__peer]    = __send_end > __send_begin ? __send_begin - __current[__rank] : 0;
      __state.__receive_offsets_[__peer] = __recv_end > __recv_begin ? __recv_begin - __desired[__rank] : 0;
    }
  }
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    for (auto&& [__comm, __state] : ::cuda::std::ranges::views::zip(__comms, __states))
    {
      // The caller's input is not overwritten while it is an in-flight send buffer.
      __comm.all_to_all_v(
        __guard,
        __state.__ordered_.data(),
        __state.__send_counts_.data(),
        __state.__send_offsets_.data(),
        __state.__input_.data(),
        __state.__receive_counts_.data(),
        __state.__receive_offsets_.data(),
        __state.__input_.stream());
    }
  }
  for (auto&& [__state, __output] : ::cuda::std::ranges::views::zip(__states, __outputs))
  {
    ::cuda::experimental::mgmn::__detail::__hashgraph_shuffle::__launch(
      __state.__input_.stream(),
      __state.__input_.size(),
      __extract<_Tp, ::cuda::std::remove_cvref_t<decltype(__output)>>,
      __state.__input_.data(),
      __output,
      __state.__input_.size());
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__hashgraph_shuffle
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_EXECUTE_H
