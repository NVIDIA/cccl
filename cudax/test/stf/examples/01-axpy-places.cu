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
 *
 * @brief This example illustrates how to use the task construct with grids of
 *        places and composite data places
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void axpy(size_t start, size_t cnt, T a, const T* x, T* y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < cnt; ind += nthreads)
  {
    y[ind + start] += a * x[ind + start];
  }
}

double X0(size_t i)
{
  return sin((double) i);
}

double Y0(size_t i)
{
  return cos((double) i);
}

template <typename Ctx>
void run()
{
  Ctx ctx;

  const int N = 1024 * 1024 * 32;
  double *X, *Y;

  X = new double[N];
  Y = new double[N];
  SCOPE(exit)
  {
    delete[] X;
    delete[] Y;
  };

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = X0(ind);
    Y[ind] = Y0(ind);
  }

  //    std::shared_ptr<execution_grid> all_devs = exec_place::all_devices();
  // use grid [ 0 0 0 0 ] for debugging purpose
  auto all_devs = exec_place::repeat(exec_place::device(0), 4);

  // 512k doubles = 4MB (2 pages)
  // A 1D blocking strategy over all devices with a block size of 32 and a round robin distribution of blocks accross
  // devices
  //    data_place cdp = data_place(exec_place::all_devices().as_grid().get_grid(),
  //            [](dim4 grid_dim, pos4 index_pos) { return pos4((index_pos.x / (512 * 1024ULL)) % grid_dim.x); });

  data_place cdp = data_place::composite(tiled_partition<512 * 1024ULL>(), all_devs);

  auto handle_X = ctx.logical_data(X, {N});
  auto handle_Y = ctx.logical_data(Y, {N});

  double alpha = 3.14;

  /* Compute Y = Y + alpha X */
  auto t = ctx.task(all_devs, handle_X.read(cdp), handle_Y.rw(cdp));
  t->*[&](auto stream, auto sX, auto sY) {
    // should be nullptr as we did not set a place
    // fprintf(stderr, "t.get_stream() = %p\n", t.get_stream());
    size_t grid_size = t.grid_dims().size();

    assert(N % grid_size == 0);

    for (size_t i = 0; i < grid_size; i++)
    {
      t.set_current_place(pos4(i));
      // fprintf(stderr, "t.get_stream(%ld) = %p\n", i, t.get_stream());
      axpy<<<16, 128, 0, stream>>>(i * N / grid_size, N / grid_size, alpha, sX.data_handle(), sY.data_handle());
      t.unset_current_place();
    }
  };

  /* Check the result on the host */
  ctx.host_launch(handle_X.read(), handle_Y.read())->*[&](auto sX, auto sY) {
    for (size_t ind = 0; ind < N; ind++)
    {
      // Y should be Y0 + alpha X0
      EXPECT(fabs(sY(ind) - (Y0(ind) + alpha * X0(ind))) < 0.0001);

      // X should be X0
      EXPECT(fabs(sX(ind) - X0(ind)) < 0.0001);
    }
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  // Disabled until composite data places are implemented with graphs
  // run<graph_ctx>();
}
