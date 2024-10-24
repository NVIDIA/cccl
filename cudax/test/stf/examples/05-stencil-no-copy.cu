//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

/*
 * DATA BLOCKS
 *   | GHOSTS | DATA | GHOSTS |
 */
template <typename T>
class data_block
{
public:
  data_block(stream_ctx& ctx, size_t beg, size_t end, size_t GHOST_SIZE)
      : beg(beg)
      , end(end)
      , block_size(end - beg)
      , ghost_size(GHOST_SIZE)
      , array(std::vector<T>(block_size + 2 * ghost_size))
      , handle(ctx.logical_data(&array[0], block_size + 2 * ghost_size))
  {}

public:
  size_t beg;
  size_t end;
  size_t block_size;
  size_t ghost_size;
  int dev_id;

private:
  std::vector<T> array;

public:
  // HANDLE = whole data + boundaries
  logical_data<slice<T>> handle;
};

template <typename T>
T check_sum(stream_ctx& ctx, data_block<T>& bn)
{
  T sum = 0.0;

  auto t = ctx.task(exec_place::host, bn.handle.read());
  t->*[&](cudaStream_t stream, auto h_center) {
    cuda_safe_call(cudaStreamSynchronize(stream));
    for (size_t offset = bn.ghost_size; offset < bn.ghost_size + bn.block_size; offset++)
    {
      sum += h_center.data_handle()[offset];
    }
  };

  return sum;
}

// array and array1 have a size of (cnt + 2*ghost_size)
template <typename T>
__global__ void stencil_kernel(size_t cnt, size_t ghost_size, T* array, const T* array1)
{
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < cnt; idx += blockDim.x * gridDim.x)
  {
    size_t idx2 = idx + ghost_size;
    array[idx2] = 0.9 * array1[idx2] + 0.05 * array1[idx2 - 1] + 0.05 * array1[idx2 + 1];
  }
}

template <typename T>
void stencil(stream_ctx& ctx, data_block<T>& bn, data_block<T>& bn1)
{
  int dev = bn.dev_id;

  auto t = ctx.task(exec_place::device(dev), bn.handle.rw(), bn1.handle.read());
  t->*[&](cudaStream_t stream, auto bn_array, auto bn1_array) {
    stencil_kernel<T>
      <<<32, 64, 0, stream>>>(bn.block_size, bn.ghost_size, bn_array.data_handle(), bn1_array.data_handle());
  };
}

template <typename T>
__global__ void copy_kernel(size_t cnt, T* dst, const T* src)
{
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < cnt; idx += blockDim.x * gridDim.x)
  {
    dst[idx] = src[idx];
  }
}

template <typename T>
void copy_task(
  stream_ctx& ctx,
  size_t cnt,
  logical_data<slice<T>>& dst,
  size_t offset_dst,
  int dst_dev,
  logical_data<slice<T>>& src,
  size_t offset_src,
  int src_dev)
{
  auto t = ctx.task(exec_place::device(dst_dev), dst.rw(), src.read(data_place::device(src_dev)));
  t->*[&](cudaStream_t stream, auto dst_array, auto src_array) {
    int nblocks = (cnt > 64) ? 32 : 1;
    copy_kernel<T>
      <<<nblocks, 64, 0, stream>>>(cnt, dst_array.data_handle() + offset_dst, src_array.data_handle() + offset_src);
  };
}

// Copy left/right handles from neighbours to the array
template <typename T>
void update_halo(stream_ctx& ctx, data_block<T>& bn, data_block<T>& left, data_block<T>& right)
{
  size_t gs = bn.ghost_size;
  size_t bs = bn.block_size;

  // Copy the bn.ghost_size last computed items in "left" (outside the halo)
  copy_task<T>(ctx, gs, bn.handle, 0, bn.dev_id, left.handle, bs, left.dev_id);

  // Copy the bn.ghost_size first computed items (outside the halo)
  copy_task<T>(ctx, gs, bn.handle, gs + bs, bn.dev_id, right.handle, gs, right.dev_id);
}

// Copy inner part of bn into bn1
template <typename T>
void copy_inner(stream_ctx& ctx, data_block<T>& bn1, data_block<T>& bn)
{
  size_t gs = bn.ghost_size;
  size_t bs = bn.block_size;

  int dev_id = bn.dev_id;

  // Copy the bn.ghost_size last computed items in "left" (outside the halo)
  copy_task<T>(ctx, bs, bn1.handle, gs, dev_id, bn.handle, gs, dev_id);
}

int main(int argc, char** argv)
{
  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  stream_ctx ctx;

  int NITER         = 500;
  size_t NBLOCKS    = 4 * ndevs;
  size_t BLOCK_SIZE = 2048 * 1024;

  if (argc > 1)
  {
    NITER = atoi(argv[1]);
  }

  if (argc > 2)
  {
    NBLOCKS = atoi(argv[2]);
  }

  const size_t GHOST_SIZE = 1;

  size_t TOTAL_SIZE = NBLOCKS * BLOCK_SIZE;

  double* U0 = new double[NBLOCKS * BLOCK_SIZE];
  for (size_t idx = 0; idx < NBLOCKS * BLOCK_SIZE; idx++)
  {
    U0[idx] = (idx == 0) ? 1.0 : 0.0;
  }

  std::vector<data_block<double>> Un;
  std::vector<data_block<double>> Un1;

  // Create blocks and allocates host data
  for (size_t b = 0; b < NBLOCKS; b++)
  {
    size_t beg = b * BLOCK_SIZE;
    size_t end = (b + 1) * BLOCK_SIZE;

    Un.emplace_back(ctx, beg, end, 1ull);
    Un1.emplace_back(ctx, beg, end, 1ull);
  }

  for (size_t b = 0; b < NBLOCKS; b++)
  {
    Un[b].dev_id  = b % ndevs;
    Un1[b].dev_id = b % ndevs;
  }

  // Fill blocks with initial values. For the sake of simplicity, we are
  // using a synchronization primitive and host code, but this could have
  // been written asynchronously using host callbacks.
  for (size_t b = 0; b < NBLOCKS; b++)
  {
    size_t beg = b * BLOCK_SIZE;

    auto t = ctx.task(exec_place::host, Un[b].handle.rw(), Un1[b].handle.rw());
    t->*[&](cudaStream_t stream, auto Un_vals, auto Un1_vals) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      for (size_t local_idx = 0; local_idx < BLOCK_SIZE; local_idx++)
      {
        double val                                     = U0[(beg + local_idx + TOTAL_SIZE) % TOTAL_SIZE];
        Un1_vals.data_handle()[local_idx + GHOST_SIZE] = val;
        Un_vals.data_handle()[local_idx + GHOST_SIZE]  = val;
      }
    };
  }

  for (int iter = 0; iter < NITER; iter++)
  {
    for (size_t b = 0; b < NBLOCKS; b++)
    {
      update_halo(ctx, Un1[b], Un[(b - 1 + NBLOCKS) % NBLOCKS], Un[(b + 1) % NBLOCKS]);
    }

    // UPDATE Un from Un1
    for (size_t b = 0; b < NBLOCKS; b++)
    {
      stencil(ctx, Un[b], Un1[b]);
    }

#if 0
    // We make sure that the total sum of elements remains constant
    if (iter % 250 == 0)
    {
      double sum = 0.0;
      for (size_t b = 0; b < NBLOCKS; b++)
      {
        sum += check_sum(ctx, Un[b]);
      }

      // fprintf(stderr, "iter %d : CHECK SUM = %e\n", iter, sum);
    }
#endif

    for (size_t b = 0; b < NBLOCKS; b++)
    {
      // Copy inner part of Un into Un1
      copy_inner(ctx, Un[b], Un1[b]);
    }
  }

  // In this stencil, the sum of the elements is supposed to be a constant
  double sum = 0.0;
  for (size_t b = 0; b < NBLOCKS; b++)
  {
    sum += check_sum(ctx, Un[b]);
  }

  double err = fabs(sum - 1.0);
  EXPECT(err < 0.0001);

  ctx.finalize();
}
