// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/detail/it_traits.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>

template <typename InputIt, typename OutputIt, typename InitValueT, typename BinaryOp>
void compute_exclusive_scan_reference(InputIt first, InputIt last, OutputIt result, InitValueT init, BinaryOp op)
{
  using value_t  = cub::detail::it_value_t<InputIt>;
  using accum_t  = ::cuda::std::__accumulator_t<BinaryOp, value_t, InitValueT>;
  using output_t = cub::detail::it_value_t<OutputIt>;
  accum_t acc    = static_cast<accum_t>(init);
  for (; first != last; ++first)
  {
    auto v    = *first;
    *result++ = static_cast<output_t>(acc);
    acc       = op(acc, v);
  }
}

template <typename InputIt, typename OutputIt, typename BinaryOp, typename InitValueT>
void compute_inclusive_scan_reference(InputIt first, InputIt last, OutputIt result, BinaryOp op, InitValueT init)
{
  using value_t  = cub::detail::it_value_t<InputIt>;
  using accum_t  = ::cuda::std::__accumulator_t<BinaryOp, value_t, InitValueT>;
  using output_t = cub::detail::it_value_t<OutputIt>;
  accum_t acc    = static_cast<accum_t>(init);
  for (; first != last; ++first)
  {
    acc       = op(acc, *first);
    *result++ = static_cast<output_t>(acc);
  }
}
