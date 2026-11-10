//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/coop/tile.cuh>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct   = cuda::tiles;
namespace coop = cuda::coop::tile;

inline constexpr size_t rows                     = 8;
inline constexpr size_t columns                  = 128;
inline constexpr size_t count                    = rows * columns;
inline constexpr size_t minimum_offset           = rows;
inline constexpr size_t maximum_offset           = minimum_offset + rows;
inline constexpr size_t columns_offset           = maximum_offset + rows;
inline constexpr size_t broadcast_offset         = columns_offset + columns;
inline constexpr size_t broadcast_minimum_offset = broadcast_offset + columns;
inline constexpr size_t scalar_offset            = broadcast_minimum_offset + columns;
inline constexpr size_t dependent_offset         = scalar_offset + 1;
inline constexpr size_t raking_offset            = dependent_offset + 1;
inline constexpr size_t output_count             = raking_offset + 1;

template <typename T, typename Shape>
__tile__ void store_flat(ct::tile<T, Shape> value, T* output)
{
  using Flat = ct::shape<ct::tile_size_v<ct::tile<T, Shape>>>;
  ct::partition_view{ct::tensor_span{output, Flat{}}, Flat{}}.store(ct::reshape(value, Flat{}), 0);
}

template <typename T>
__tile__ void reduction_values(T* input, T* output)
{
  const auto values   = ct::partition_view{ct::tensor_span{input, ct::shape<count>{}}, ct::shape<count>{}}.load(0);
  const auto lanes    = ct::partition_view{ct::tensor_span{input, ct::shape<columns>{}}, ct::shape<columns>{}}.load(0);
  const auto matrix   = ct::reshape(values, ct::shape<rows, columns>{});
  const auto row_sums = coop::sum<1>(matrix);
  const auto column_sums = coop::sum<0>(matrix);
  const auto minimum     = coop::reduce<1>(matrix, cuda::minimum<>{});
  const auto maximum     = coop::reduce<1>(matrix, cuda::maximum<>{});
  const auto scalar      = coop::sum<0>(lanes);
  const auto dependent   = coop::reduce<0>(row_sums, cuda::std::plus<>{});
  const auto raking      = coop::sum<0>(row_sums, coop::reduce_policy<coop::reduce_algorithm::raking>{});
  static_assert(cuda::std::is_same_v<decltype(row_sums), const ct::tile<T, ct::shape<rows, 1>>>);
  static_assert(cuda::std::is_same_v<decltype(column_sums), const ct::tile<T, ct::shape<1, columns>>>);
  static_assert(cuda::std::is_same_v<decltype(scalar), const ct::tile<T, ct::shape<1>>>);
  static_assert(cuda::std::is_same_v<decltype(dependent), const ct::tile<T, ct::shape<1, 1>>>);
  store_flat(row_sums, output);
  store_flat(minimum, output + minimum_offset);
  store_flat(maximum, output + maximum_offset);
  store_flat(column_sums, output + columns_offset);
  store_flat(coop::sum(lanes), output + broadcast_offset);
  store_flat(coop::reduce(lanes, cuda::minimum<>{}), output + broadcast_minimum_offset);
  store_flat(scalar, output + scalar_offset);
  store_flat(dependent, output + dependent_offset);
  store_flat(raking, output + raking_offset);
  store_flat(values, input);
}

template <typename T, size_t Warps>
[[cutile::hint(0, num_worker_warps_per_cta = Warps)]]
__tile_global__ void reduce_explicit(T* input, T* output)
{
  reduction_values(input, output);
}

template <typename T>
__tile_global__ void reduce_automatic(T* input, T* output)
{
  reduction_values(input, output);
}

bool check(cudaError_t status)
{
  if (status == cudaSuccess)
  {
    return true;
  }
  std::cerr << cudaGetErrorString(status) << '\n';
  return false;
}

template <typename T, size_t Warps>
bool run_case()
{
  std::vector<T> input(count), preserved(count), output(output_count), expected(output_count);
  for (size_t i = 0; i < count; ++i)
  {
    input[i] = static_cast<T>(static_cast<int>((i * 17) % 41) - 20) / T{8};
  }
  T total = 0;
  for (size_t row = 0; row < rows; ++row)
  {
    T sum     = 0;
    T minimum = input[row * columns];
    T maximum = minimum;
    for (size_t column = 0; column < columns; ++column)
    {
      const T value = input[row * columns + column];
      sum += value;
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      expected[columns_offset + column] += value;
    }
    expected[row]                  = sum;
    expected[minimum_offset + row] = minimum;
    expected[maximum_offset + row] = maximum;
    total += sum;
  }
  for (size_t column = 0; column < columns; ++column)
  {
    expected[broadcast_offset + column]         = expected[0];
    expected[broadcast_minimum_offset + column] = expected[minimum_offset];
  }
  expected[scalar_offset]    = expected[0];
  expected[dependent_offset] = total;
  expected[raking_offset]    = total;

  T* device_input  = nullptr;
  T* device_output = nullptr;
  if (!check(cudaMalloc(&device_input, count * sizeof(T)))
      || !check(cudaMalloc(&device_output, output_count * sizeof(T)))
      || !check(cudaMemcpy(device_input, input.data(), count * sizeof(T), cudaMemcpyHostToDevice)))
  {
    return false;
  }
  if constexpr (Warps == 0)
  {
    reduce_automatic<T><<<1, 1>>>(device_input, device_output);
  }
  else
  {
    reduce_explicit<T, Warps><<<1, 1>>>(device_input, device_output);
  }
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, output_count * sizeof(T), cudaMemcpyDeviceToHost))
      || !check(cudaMemcpy(preserved.data(), device_input, count * sizeof(T), cudaMemcpyDeviceToHost))
      || !check(cudaFree(device_output)) || !check(cudaFree(device_input)))
  {
    return false;
  }
  for (size_t i = 0; i < output_count; ++i)
  {
    const double error     = std::abs(static_cast<double>(output[i]) - static_cast<double>(expected[i]));
    const double tolerance = 1e-6 * std::max(1.0, std::abs(static_cast<double>(expected[i])));
    if (!(error <= tolerance))
    {
      std::cerr << "reduction mismatch at " << i << " with " << Warps << " worker warps\n";
      return false;
    }
  }
  return input == preserved;
}

int main()
{
  if (!run_case<double, 0>() || !run_case<double, 4>() || !run_case<double, 8>() || !run_case<double, 16>()
      || !run_case<float, 8>() || !run_case<int, 16>())
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: C++ axis reductions and legacy broadcasts at automatic/4/8/16 worker warps\n";
  return EXIT_SUCCESS;
}
