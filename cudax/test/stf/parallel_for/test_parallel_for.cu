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
 * @brief This test ensures that the parallel_for API works properly with its
 *        different overloads (with places, partitioners, etc...)
 */

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

__host__ __device__ double x0(size_t ind)
{
  return sin((double) ind);
}

__host__ __device__ double y0(size_t ind)
{
  return cos((double) ind);
}

int main()
{
  for (int style = 0; style < 7; ++style)
  {
    stream_ctx ctx;

    const int N = 16;
    double X[2 * N], Y[N];

    for (size_t ind = 0; ind < 2 * N; ind++)
    {
      X[ind] = x0(ind);
    }

    for (size_t ind = 0; ind < N; ind++)
    {
      Y[ind] = y0(ind);
    }

    auto lx = ctx.logical_data(X);
    auto ly = ctx.logical_data(Y);

    switch (style)
    {
      case 0:
        // Basic usage
        ctx.parallel_for(ly.shape(), lx.read(), ly.rw())->*[=] _CCCL_DEVICE(size_t pos, auto sx, auto sy) {
          sy(pos) += 0.5 * (sx(2 * pos) + sx(2 * pos + 1));
        };
        break;
      case 1:
        // This throws because you're attempting to run a device lambda on the host
        try
        {
          ctx.parallel_for(exec_place::host, ly.shape(), lx.read(), ly.rw())
              ->*[=] _CCCL_DEVICE(size_t pos, auto sx, auto sy) {
                    sy(pos) += 0.5 * (sx(2 * pos) + sx(2 * pos + 1));
                  };
        }
        catch (std::exception&)
        {
          // All good, but don't check the result
          ctx.finalize();
          continue;
        }
        assert(0);
        break;
      case 2:
        // This throws because you're attempting to run a host lambda on a device
        try
        {
          ctx.parallel_for(ly.shape(), lx.read(), ly.rw())->*[=] __host__(size_t pos, auto sx, auto sy) {
            sy(pos) += 0.5 * (sx(2 * pos) + sx(2 * pos + 1));
          };
        }
        catch (std::exception&)
        {
          // All good, but don't check the result
          ctx.finalize();
          continue;
        }
        assert(0);
        break;
      case 3:
        // This works because it dynamically selects the dual function to run on the host
        ctx.parallel_for(exec_place::host, ly.shape(), lx.read(), ly.rw())
            ->*[=] __host__ __device__(size_t pos, slice<double> sx, slice<double> sy) {
                  sy(pos) += 0.5 * (sx(2 * pos) + sx(2 * pos + 1));
                };
        break;
      case 4:
        // This works because it dynamically selects the dual function to run on the device
        ctx.parallel_for(exec_place::current_device(), ly.shape(), lx.read(), ly.rw())
            ->*[=] __host__ __device__(size_t pos, slice<double> sx, slice<double> sy) {
                  sy(pos) += 0.5 * (sx(2 * pos) + sx(2 * pos + 1));
                };
        break;
      case 5:
        // This works because it dynamically selects the dual function to run on the current device
        ctx.parallel_for(ly.shape(), lx.read(), ly.rw())
            ->*[=] __host__ __device__(size_t pos, slice<double> sx, slice<double> sy) {
                  sy(pos) += 0.5 * (sx(2 * pos) + sx(2 * pos + 1));
                };
        break;
      case 6:
        // This works because it dispatches on all devices
        ctx.parallel_for(blocked_partition(), exec_place::all_devices(), ly.shape(), lx.read(), ly.rw())
            ->*[=] __host__ __device__(size_t pos, slice<double> sx, slice<double> sy) {
                  sy(pos) += 0.5 * (sx(2 * pos) + sx(2 * pos + 1));
                };
        break;
      default:
        assert(0);
    }

    /* Check the result on the host */
    ctx.parallel_for(exec_place::host, ly.shape(), lx.read(), ly.read())->*[=](size_t pos, auto sx, auto sy) {
      EXPECT(fabs(sx(2 * pos) - x0(2 * pos)) < 0.0001);
      EXPECT(fabs(sx(2 * pos + 1) - x0(2 * pos + 1)) < 0.0001);
      EXPECT(fabs(sy(pos) - (y0(pos) + 0.5 * (x0(2 * pos) + x0(2 * pos + 1)))) < 0.0001);
    };
    ctx.finalize();
  }
}
