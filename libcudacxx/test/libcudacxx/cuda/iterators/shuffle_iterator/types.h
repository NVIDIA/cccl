//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_ITERATOR_SHUFFLE_ITERATOR_H
#define TEST_CUDA_ITERATOR_SHUFFLE_ITERATOR_H

#include <cuda/std/cstdint>

#include "test_macros.h"

struct fake_rng
{
  using result_type = uint32_t;

  constexpr fake_rng() = default;

  [[nodiscard]] __host__ __device__ constexpr result_type operator()() noexcept
  {
    return __random_indices[__start++ % 5];
  }

  // Needed for uniform_int_distribution
  [[nodiscard]] __host__ __device__ static constexpr result_type min() noexcept
  {
    return 0;
  }

  [[nodiscard]] __host__ __device__ static constexpr result_type max() noexcept
  {
    return 5;
  }

  uint32_t __start{0};
  uint32_t __random_indices[5] = {4, 1, 2, 0, 3};
};
static_assert(cuda::std::__cccl_random_is_valid_urng<fake_rng>);

template <bool HasConstructor = true, bool HasNothrowCallOperator = true>
struct fake_bijection
{
  using index_type = uint32_t;

  constexpr fake_bijection() = default;

  _CCCL_TEMPLATE(class RNG, bool HasConstructor2 = HasConstructor)
  _CCCL_REQUIRES(HasConstructor2)
  __host__ __device__ constexpr fake_bijection(index_type, RNG&&) noexcept {}

  [[nodiscard]] __host__ __device__ constexpr index_type size() const noexcept(HasNothrowCallOperator)
  {
    return 5;
  }

  [[nodiscard]] __host__ __device__ constexpr index_type operator()(index_type n) const noexcept(HasNothrowCallOperator)
  {
    return __random_indices[n];
  }

  uint32_t __random_indices[5] = {4, 1, 2, 0, 3};
};

#endif // TEST_CUDA_ITERATOR_SHUFFLE_ITERATOR_H
