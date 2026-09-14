// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/detail/fast_modulo_division.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "c2h/utility.h"
#include "c2h/vector.h"
#include "cub_test_macros.h"

using uint_types = c2h::type_list<::cuda::std::uint32_t, ::cuda::std::uint64_t>;

template <typename UInt>
[[nodiscard]] UInt reference_divide(UInt numerator, UInt divisor)
{
  return divisor == UInt{0} ? numerator : numerator / divisor;
}

template <typename UInt>
[[nodiscard]] constexpr auto divisors()
{
  constexpr int bits       = static_cast<int>(sizeof(UInt) * 8);
  constexpr UInt max_value = ::cuda::std::numeric_limits<UInt>::max();
  constexpr UInt high_bit  = UInt{1} << (bits - 1);

  return ::cuda::std::array<UInt, 15>{
    UInt{0},
    UInt{1},
    UInt{2},
    UInt{4},
    UInt{8},
    high_bit,
    UInt{3},
    UInt{5},
    UInt{7},
    UInt{10},
    UInt{31},
    high_bit - UInt{1},
    high_bit + UInt{1},
    max_value - UInt{1},
    max_value};
}

template <typename UInt>
[[nodiscard]] c2h::host_vector<UInt> make_numerators()
{
  constexpr int bits       = static_cast<int>(sizeof(UInt) * 8);
  constexpr UInt max_value = ::cuda::std::numeric_limits<UInt>::max();
  constexpr UInt high_bit  = UInt{1} << (bits - 1);

  c2h::host_vector<UInt> numerators{
    UInt{0}, UInt{1}, UInt{2}, UInt{3}, high_bit - UInt{1}, high_bit, high_bit + UInt{1}, max_value - UInt{1}, max_value};

  for (const UInt divisor : divisors<UInt>())
  {
    if (divisor != UInt{0})
    {
      numerators.push_back(divisor - UInt{1});
      numerators.push_back(divisor);
      if (divisor != max_value)
      {
        numerators.push_back(divisor + UInt{1});
      }
      const UInt largest_multiple = max_value - (max_value % divisor);
      numerators.push_back(largest_multiple);
      if (largest_multiple != UInt{0})
      {
        numerators.push_back(largest_multiple - UInt{1});
      }
    }
  }

  UInt value = static_cast<UInt>(0x9e3779b9U);
  for (int i = 0; i < 4096; ++i)
  {
    value = value * static_cast<UInt>(6364136223846793005ULL) + static_cast<UInt>(1442695040888963407ULL);
    numerators.push_back(value);
  }
  return numerators;
}

template <typename UInt>
__global__ void
fast_divide_by_constant_kernel(const UInt* numerators, UInt* quotients, ::cuda::std::size_t count, UInt divisor)
{
  const ::cuda::std::size_t index = static_cast<::cuda::std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
  {
    const cub::detail::fast_divide_by_constant<UInt> divider{divisor};
    quotients[index] = divider.divide(numerators[index]);
  }
}

CUB_TEST("fast_divide_by_constant agrees with division on the host", "[util][division]", CUB_SMALL, uint_types)
{
  using uint_t = c2h::get<0, TestType>;

  const c2h::host_vector<uint_t> numerators = make_numerators<uint_t>();
  cub::detail::fast_divide_by_constant<uint_t> divider;
  for (const uint_t divisor : divisors<uint_t>())
  {
    divider.init(divisor);
    CAPTURE(c2h::type_name<uint_t>(), divisor);
    for (const uint_t numerator : numerators)
    {
      CAPTURE(numerator);
      REQUIRE(divider.divide(numerator) == reference_divide(numerator, divisor));
    }
  }
}

CUB_TEST("fast_divide_by_constant agrees with division on the device", "[util][division]", CUB_SMALL, uint_types)
{
  using uint_t = c2h::get<0, TestType>;

  const c2h::host_vector<uint_t> numerators = make_numerators<uint_t>();
  const c2h::device_vector<uint_t> d_numerators(numerators);
  c2h::device_vector<uint_t> d_quotients(numerators.size());

  constexpr int block_threads = 256;
  const int blocks            = static_cast<int>((numerators.size() + block_threads - 1) / block_threads);
  for (const uint_t divisor : divisors<uint_t>())
  {
    fast_divide_by_constant_kernel<<<blocks, block_threads>>>(
      thrust::raw_pointer_cast(d_numerators.data()),
      thrust::raw_pointer_cast(d_quotients.data()),
      numerators.size(),
      divisor);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    const c2h::host_vector<uint_t> quotients = d_quotients;
    CAPTURE(c2h::type_name<uint_t>(), divisor);
    for (::cuda::std::size_t index = 0; index < numerators.size(); ++index)
    {
      CAPTURE(index, numerators[index]);
      REQUIRE(quotients[index] == reference_divide(numerators[index], divisor));
    }
  }
}
