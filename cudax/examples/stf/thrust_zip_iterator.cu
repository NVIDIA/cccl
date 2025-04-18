//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief This example illustrates how we can convert Thrust iterators to
 *        logical data, and how to create thrust iterators from data instances in a
 *        task.
 */

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cuda/experimental/stf.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

// Functor to apply the transformation
struct my_transform_functor
{
  __host__ __device__ int operator()(const thrust::tuple<int, char>& t) const
  {
    int a  = thrust::get<0>(t);
    char b = thrust::get<1>(t);
    return a + static_cast<int>(b); // Example operation
  }
};

/*
 * How to use CUDASTF to manipulate data originally created using Thrust
 */
template <typename ZippedIt, typename OutIt>
void thrust_algorithm(context& ctx, ZippedIt& first, ZippedIt& last, OutIt& output, data_place data_location)
{
  /*
   * Interpret Thrust data structures as logical data
   */
  size_t num_elements = cuda::std::distance(first, last);

  // Extract underlying iterators from the zip iterator
  auto itA = thrust::get<0>(first.get_iterator_tuple());
  int* A   = thrust::raw_pointer_cast(&(*itA));

  auto itB = thrust::get<1>(first.get_iterator_tuple());
  char* B  = thrust::raw_pointer_cast(&(*itB));

  int* C = thrust::raw_pointer_cast(output.data());

  auto lA = ctx.logical_data(make_slice(A, num_elements), data_location);
  auto lB = ctx.logical_data(make_slice(B, num_elements), data_location);
  auto lC = ctx.logical_data(make_slice(C, num_elements), data_location);

  /* Important : result C will only be valid once we finalize the context or introduce a task fence ! */
  ctx.task(lA.read(), lB.read(), lC.write())->*[](cudaStream_t stream, auto dA, auto dB, auto dC) {
    // Reconstruct a zipped iterator from the data instances passed to the lambda function
    size_t num_elements = dA.size();
    auto dfirst         = thrust::make_zip_iterator(thrust::make_tuple(dA.data_handle(), dB.data_handle()));
    auto dlast          = dfirst + num_elements;

    // Create a device pointer from the raw pointer
    thrust::device_ptr<int> dout = thrust::device_pointer_cast(dC.data_handle());

    thrust::transform(thrust::cuda::par_nosync.on(stream), dfirst, dlast, dout, my_transform_functor());
  };
}

int main()
{
  context ctx;

  /*
   * First create device vectors and zipped them
   */
  thrust::device_vector<int> A(3);
  thrust::device_vector<char> B(3);
  thrust::device_vector<int> C(3);

  A[0] = 10;
  A[1] = 20;
  A[2] = 30;
  B[0] = 'x';
  B[1] = 'y';
  B[2] = 'z';

  auto first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));
  auto last  = thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end()));

  thrust_algorithm(ctx, first, last, C, data_place::current_device());

  /*
   * Use host data, and rely on CUDASTF for transfers
   */

  thrust::host_vector<int> hA(3);
  thrust::host_vector<char> hB(3);
  thrust::host_vector<int> hC(3);

  hA[0] = 10;
  hA[1] = 20;
  hA[2] = 30;
  hB[0] = 'x';
  hB[1] = 'y';
  hB[2] = 'z';

  auto hfirst = thrust::make_zip_iterator(thrust::make_tuple(hA.begin(), hB.begin()));
  auto hlast  = thrust::make_zip_iterator(thrust::make_tuple(hA.end(), hB.end()));

  thrust_algorithm(ctx, hfirst, hlast, hC, data_place::host());

  /* Before this, we cannot assume that the Thrust algorithms have been
   * performed and/or that the results have been written back to their original
   * location. */
  ctx.finalize();

  // Check results
  for (size_t i = 0; i < 3; i++)
  {
    EXPECT(C[i] == (A[i] + static_cast<int>(B[i])));
    EXPECT(hC[i] == (hA[i] + static_cast<int>(hB[i])));
  }

  return 0;
}
