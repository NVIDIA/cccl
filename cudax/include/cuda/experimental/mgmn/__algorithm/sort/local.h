// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_LOCAL_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_LOCAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC)
#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/kernels.h>
#endif
#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cuda/__driver/driver_api.h>
#include <cuda/__execution/tune.h>
#include <cuda/__memory_resource/allocation_alignment.h>
#include <cuda/__stream/stream.h>
#include <cuda/launch>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/experimental/mgmn/__algorithm/common.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <class _Key, class _Compare, class _Env>
_CCCL_HOST_API void
__local_sort(_Key* __keys, _Key* __sorted_output, ::cuda::std::int64_t __count, _Compare __cmp, const _Env& __env)
{
  if (__count == 0)
  {
    return;
  }
  if constexpr (::cuda::std::is_arithmetic_v<_Key>
                && (::cuda::std::is_same_v<_Compare, ::cuda::std::less<>>
                    || ::cuda::std::is_same_v<_Compare, ::cuda::std::less<_Key>>) )
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      ::cuda::get_stream(__env),
      CUB_NS_QUALIFIER::DeviceRadixSort::SortKeys,
      __keys,
      __sorted_output,
      __count,
      0,
      sizeof(_Key) * 8,
      __env);
  }
  else if constexpr (::cuda::std::is_arithmetic_v<_Key>
                     && (::cuda::std::is_same_v<_Compare, ::cuda::std::greater<>>
                         || ::cuda::std::is_same_v<_Compare, ::cuda::std::greater<_Key>>) )
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      ::cuda::get_stream(__env),
      CUB_NS_QUALIFIER::DeviceRadixSort::SortKeysDescending,
      __keys,
      __sorted_output,
      __count,
      0,
      sizeof(_Key) * 8,
      __env);
  }
  else
  {
    ::cuda::__driver::__memcpyAsync(__sorted_output, __keys, __count * sizeof(_Key), ::cuda::get_stream(__env).get());
    __CUDAX_MULTI_GPU_DISPATCH(
      ::cuda::get_stream(__env), CUB_NS_QUALIFIER::DeviceMergeSort::SortKeys, __sorted_output, __count, __cmp, __env);
  }
}

template <class _Env>
class __scratch_allocator
{
  const _Env& __env_;
  ::cuda::stream_ref __stream_;
  ::cuda::experimental::mgmn::__detail::__resource_type_for<_Env> __resource_;

public:
  _CCCL_HOST_API __scratch_allocator(const _Env& __env, ::cuda::stream_ref __stream)
      : __env_(__env)
      , __stream_(__stream)
      , __resource_(::cuda::experimental::mgmn::__detail::__resource_from_env(__env, __stream.__logical_device()))
  {}
  template <class _Tp>
  [[nodiscard]] _CCCL_HOST_API auto operator()(::cuda::std::size_t __count) const
  {
    // Sixteen-key offsets preserve the transfer alignment requested here.
    const auto __requested = ::cuda::__call_or(::cuda::allocation_alignment, alignof(_Tp), __env_);
    ::cuda::__validate_allocation_alignment(__requested, alignof(_Tp));
    using _Alignment        = ::cuda::std::execution::prop<::cuda::allocation_alignment_t, ::cuda::std::size_t>;
    const auto __buffer_env = ::cuda::std::execution::env<_Alignment, const _Env&>{
      _Alignment{::cuda::allocation_alignment, ::cuda::std::max(::cuda::std::size_t{128}, __requested)}, __env_};
    return ::cuda::experimental::mgmn::__detail::__make_safe_uninitialized_buffer<_Tp>(
      __stream_, __resource_, __count, __buffer_env);
  }
};

struct __run
{
  ::cuda::std::int64_t __offset_, __count_;
};

// Limit both total work and average fragment size: a low-fan-in merge
// reaches the measured crossover sooner than a full-warp merge.
[[nodiscard]] _CCCL_HOST_API constexpr bool
__small_tiled_merge_size_supported(::cuda::std::size_t __runs, ::cuda::std::int64_t __size) noexcept
{
  constexpr ::cuda::std::int64_t __max_keys         = 64 * 1024 * 1024 / sizeof(int);
  constexpr ::cuda::std::int64_t __max_keys_per_run = 8 * 1024 * 1024 / sizeof(int);
  return __runs >= 3 && __runs <= 32 && __size > 0 && __size <= __max_keys
      && __size <= static_cast<::cuda::std::int64_t>(__runs) * __max_keys_per_run;
}

#if _CCCL_CUDA_COMPILER(NVCC)
template <class _Compare, class _Env>
_CCCL_HOST_API _CCCL_NOINLINE void __merge_small_tiled(
  int* __received,
  ::cuda::std::int64_t __size,
  const ::std::vector<__run>& __runs,
  int* __keys,
  _Compare __cmp,
  const _Env& __env,
  int __cc)
{
  using __policy     = ::cuda::experimental::mgmn::__detail::__sort::__multiway::static_mway_policy<512, 17>;
  using __count_type = ::cuda::std::int64_t;
  using __agent =
    ::cuda::experimental::mgmn::__detail::__sort::__multiway::agent_mway_merge_t<__policy, int, __count_type, _Compare>;
  const auto __stream = ::cuda::get_stream(__env);
  const __scratch_allocator<_Env> __allocate{__env, __stream};
  ::cuda::experimental::mgmn::__detail::__sort::__multiway::mway_input<int, __count_type> __input{};
  __input.k = int(__runs.size());
  for (::cuda::std::size_t __i = 0; __i < __runs.size(); ++__i)
  {
    __input.ptrs[__i]   = __received + __runs[__i].__offset_;
    __input.counts[__i] = __runs[__i].__count_;
  }
  const __count_type __tile      = __agent::items_per_tile;
  const __count_type __tiles     = (__size + __tile - 1) / __tile;
  const __count_type __diagonals = __tiles + 1;
  auto __boundaries              = __allocate.template operator()<__count_type>(__diagonals * 32);
  ::cuda::launch(
    __stream,
    ::cuda::make_config(::cuda::block_dims<256>(), ::cuda::grid_dims((__diagonals + 7) / 8)),
    ::cuda::experimental::mgmn::__detail::__sort::__multiway::mway_partition_kernel<int, __count_type, _Compare>,
    __input,
    __tile,
    __diagonals,
    __size,
    __boundaries.data(),
    __cmp);
  const auto __shared = __agent::temp_storage_bytes(__cc);
  ::cuda::launch(
    __stream,
    ::cuda::make_config(::cuda::block_dims<512>(),
                        ::cuda::grid_dims(__tiles),
                        ::cuda::dynamic_shared_memory<char[]>(__shared, ::cuda::non_portable)),
    ::cuda::experimental::mgmn::__detail::__sort::__multiway::mway_merge_kernel<__agent, int, __count_type, _Compare>,
    __input,
    __keys,
    __cmp,
    __boundaries.data());
}
#endif

template <class _Key, class _Compare, class _Env>
_CCCL_HOST_API void __merge_received(
  _Key* __received,
  ::cuda::std::int64_t __size,
  ::std::vector<__run> __runs,
  _Key* __keys,
  _Compare __cmp,
  const _Env& __env)
{
  const auto __stream = ::cuda::get_stream(__env);
  if (__runs.size() <= 1)
  {
    if (__size)
    {
      ::cuda::__driver::__memcpyAsync(__keys, __received + __runs[0].__offset_, __size * sizeof(_Key), __stream.get());
    }
    return;
  }
#if _CCCL_CUDA_COMPILER(NVCC)
  if constexpr (::cuda::std::is_same_v<_Key, int>
                && (::cuda::std::is_same_v<_Compare, ::cuda::std::less<>>
                    || ::cuda::std::is_same_v<_Compare, ::cuda::std::less<int>>
                    || ::cuda::std::is_same_v<_Compare, ::cuda::std::greater<>>
                    || ::cuda::std::is_same_v<_Compare, ::cuda::std::greater<int>>)
                && !::cuda::std::execution::__queryable_with<_Env, ::cuda::execution::__get_tuning_t>)
  {
    if (::cuda::experimental::mgmn::__detail::__sort::__small_tiled_merge_size_supported(__runs.size(), __size))
    {
      const auto __device = ::cuda::__driver::__deviceGet(__stream.device().get());
      const int __major =
        ::cuda::__driver::__deviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, __device);
      const int __minor =
        ::cuda::__driver::__deviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, __device);
      if (__major == 10 && __minor == 0)
      {
        ::cuda::experimental::mgmn::__detail::__sort::__merge_small_tiled(
          __received, __size, __runs, __keys, __cmp, __env, __major * 100 + __minor * 10);
        return;
      }
    }
  }
#endif

  if (__runs.size() == 2)
  {
    const auto __a = __runs[0], __b = __runs[1];
    __CUDAX_MULTI_GPU_DISPATCH(
      ::cuda::get_stream(__env),
      CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
      __received + __a.__offset_,
      __a.__count_,
      __received + __b.__offset_,
      __b.__count_,
      __keys,
      __cmp,
      __env);
    return;
  }
  const __scratch_allocator<_Env> __allocate{__env, __stream};
  auto __scratch       = __allocate.template operator()<_Key>(::cuda::std::max<::cuda::std::int64_t>(__size, 1));
  _Key* __merge_input  = __received;
  _Key* __merge_output = __scratch.data();
  if (__runs.size() > 1)
  {
    // The independent merge streams must wait for the caller-stream allocation.
    ::cuda::__driver::__streamSynchronize(__stream.get());
  }
  ::std::vector<::cuda::stream> __streams;
  __streams.reserve(__runs.size() / 2);
  for (::cuda::std::size_t __idx = 0; __idx < __runs.size() / 2; ++__idx)
  {
    __streams.emplace_back(__stream.__logical_device());
  }
  _CCCL_TRY
  {
    while (__runs.size() > 1)
    {
      // The final merge can write directly to the caller's disjoint output.
      // Earlier rounds alternate between the receive buffer and scratch.
      if (__runs.size() == 2)
      {
        __merge_output = __keys;
      }
      ::std::vector<__run> __next;
      ::cuda::std::int64_t __sorted_output_offset = 0;
      const ::cuda::std::size_t __pairs           = __runs.size() / 2;
      for (::cuda::std::size_t __pair = 0; __pair < __pairs; ++__pair)
      {
        const auto __a = __runs[2 * __pair], __b = __runs[2 * __pair + 1];
        // Override only the stream; preserve resource, tuning and all other
        // CUB environment queries. Each concurrent merge owns its scratch.
        auto __merge_env =
          ::cuda::std::execution::env<::cuda::stream_ref, const _Env&>{::cuda::stream_ref{__streams[__pair]}, __env};
        __CUDAX_MULTI_GPU_DISPATCH(
          ::cuda::get_stream(__merge_env),
          CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
          __merge_input + __a.__offset_,
          __a.__count_,
          __merge_input + __b.__offset_,
          __b.__count_,
          __merge_output + __sorted_output_offset,
          __cmp,
          __merge_env);
        __next.push_back({__sorted_output_offset, __a.__count_ + __b.__count_});
        __sorted_output_offset += __a.__count_ + __b.__count_;
      }
      if (__runs.size() % 2)
      {
        const auto __tail = __runs.back();
        ::cuda::__driver::__memcpyAsync(
          __merge_output + __sorted_output_offset,
          __merge_input + __tail.__offset_,
          __tail.__count_ * sizeof(_Key),
          __streams[0].get());
        __next.push_back({__sorted_output_offset, __tail.__count_});
      }
      for (::cuda::std::size_t __idx = 0; __idx < __pairs; ++__idx)
      {
        __streams[__idx].sync();
      }
      __runs.swap(__next);
      ::cuda::std::swap(__merge_input, __merge_output);
    }
  }
  _CCCL_CATCH_ALL
  {
    for (auto& __merge_stream : __streams)
    {
      _CCCL_ASSERT_DRIVER_API(
        ::cuda::__driver::__streamSynchronizeNoThrow, "waiting for failed sort merge", __merge_stream.get());
    }
    _CCCL_RETHROW;
  }
  // Every merge has completed before the temporary inputs are released.
  // The final round already wrote the caller's output, so no copy is needed.
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)
#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_LOCAL_H
