// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// keep checks at the top so compilation of discarded variants fails really fast
#include <cub/device/dispatch/dispatch_transform.cuh>
#if !TUNE_BASE && TUNE_ALGORITHM == 1
#  if _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
#    error "When tuning, this benchmark does not support being compiled for multiple architectures"
#  endif
#  if (__CUDA_ARCH_LIST__) < 900
#    error "Cannot compile algorithm 4 (ublkcp) below sm90"
#  endif
#  ifndef _CUB_HAS_TRANSFORM_UBLKCP
#    error "Cannot tune for ublkcp algorithm, which is not provided by CUB (old CTK?)"
#  endif
#endif

#include <cub/util_namespace.cuh>

#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

template <typename... RandomAccessIteratorsIn>
#if TUNE_BASE
using policy_hub_t = cub::detail::transform::policy_hub<false, ::cuda::std::tuple<RandomAccessIteratorsIn...>>;
#else
struct policy_hub_t
{
  struct max_policy : cub::ChainedPolicy<500, max_policy, max_policy>
  {
    static constexpr int min_bif    = cub::detail::transform::arch_to_min_bytes_in_flight(__CUDA_ARCH_LIST__);
    static constexpr auto algorithm = static_cast<cub::detail::transform::Algorithm>(TUNE_ALGORITHM);
    using algo_policy =
      ::cuda::std::_If<algorithm == cub::detail::transform::Algorithm::prefetch,
                       cub::detail::transform::prefetch_policy_t<TUNE_THREADS>,
                       cub::detail::transform::async_copy_policy_t<TUNE_THREADS, __CUDA_ARCH_LIST__ == 900 ? 128 : 16>>;
  };
};
#endif

template <typename OffsetT,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename ExecTag = decltype(nvbench::exec_tag::no_batch)>
void bench_transform(
  nvbench::state& state,
  ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
  RandomAccessIteratorOut output,
  OffsetT num_items,
  TransformOp transform_op,
  ExecTag exec_tag = nvbench::exec_tag::no_batch)
{
  state.exec(nvbench::exec_tag::gpu | exec_tag, [&](const nvbench::launch& launch) {
    cub::detail::transform::dispatch_t<
      cub::detail::transform::requires_stable_address::no,
      OffsetT,
      ::cuda::std::tuple<RandomAccessIteratorsIn...>,
      RandomAccessIteratorOut,
      TransformOp,
      policy_hub_t<RandomAccessIteratorsIn...>>::dispatch(inputs, output, num_items, transform_op, launch.get_stream());
  });
}

// TODO(bgruber): we should put those somewhere into libcu++:
// from C++ GSL
struct narrowing_error : std::runtime_error
{
  narrowing_error()
      : std::runtime_error("Narrowing error")
  {}
};

// from C++ GSL
// implementation insipired by: https://github.com/microsoft/GSL/blob/main/include/gsl/narrow
template <typename DstT, typename SrcT, ::cuda::std::enable_if_t<::cuda::std::is_arithmetic_v<SrcT>, int> = 0>
constexpr DstT narrow(SrcT value)
{
  constexpr bool is_different_signedness = ::cuda::std::is_signed_v<SrcT> != ::cuda::std::is_signed_v<DstT>;
  const auto converted                   = static_cast<DstT>(value);
  if (static_cast<SrcT>(converted) != value || (is_different_signedness && ((converted < DstT{}) != (value < SrcT{}))))
  {
    throw narrowing_error{};
  }
  return converted;
}
