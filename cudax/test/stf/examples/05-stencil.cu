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

static stream_ctx ctx;

/*
 * DATA BLOCKS
 *   | GHOSTS | DATA | GHOSTS |
 */
template <typename T>
class data_block
{
public:
  data_block(size_t beg, size_t end, size_t GHOST_SIZE)
      : beg(beg)
      , end(end)
      , block_size(end - beg)
      , ghost_size(GHOST_SIZE)
      , array(std::vector<T>(block_size + 2 * ghost_size))
      , left_interface(std::vector<T>(ghost_size))
      , right_interface(std::vector<T>(ghost_size))
      , handle(ctx.logical_data(&array[0], block_size + 2 * ghost_size))
      , left_handle(ctx.logical_data(&left_interface[0], ghost_size))
      , right_handle(ctx.logical_data(&right_interface[0], ghost_size))
  {}

  T check_sum()
  {
    T sum = 0.0;

    ctx.task(exec_place::host, handle.read())->*[&](cudaStream_t stream, auto sn) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      const T* h_center = sn.data_handle();
      for (size_t offset = ghost_size; offset < ghost_size + block_size; offset++)
      {
        sum += h_center[offset];
      }
    };

    return sum;
  }

public:
  size_t beg;
  size_t end;
  size_t block_size;
  size_t ghost_size;
  int preferred_device;

private:
  std::vector<T> array;
  std::vector<T> left_interface;
  std::vector<T> right_interface;

public:
  // HANDLE = whole data + boundaries
  logical_data<slice<T>> handle;
  // A piece of data to store the left part of the block
  logical_data<slice<T>> left_handle;
  // A piece of data to store the right part of the block
  logical_data<slice<T>> right_handle;
};

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

// bn1.array = bn.array
template <typename T>
void stencil(data_block<T>& bn, data_block<T>& bn1)
{
  int dev = bn.preferred_device;

  ctx.task(exec_place::device(dev), bn.handle.rw(), bn1.handle.read())->*[&](cudaStream_t stream, auto sN, auto sN1) {
    stencil_kernel<T><<<32, 64, 0, stream>>>(bn.block_size, bn.ghost_size, sN.data_handle(), sN1.data_handle());
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
  size_t cnt, logical_data<slice<T>>& dst, size_t offset_dst, logical_data<slice<T>>& src, size_t offset_src, int dev)
{
  ctx.task(exec_place::device(dev), dst.rw(), src.read())->*[&](cudaStream_t stream, auto dstS, auto srcS) {
    int nblocks = (cnt > 64) ? 32 : 1;
    copy_kernel<T><<<nblocks, 64, 0, stream>>>(cnt, dstS.data_handle() + offset_dst, srcS.data_handle() + offset_src);
  };
}

template <typename T>
void update_inner_interfaces(data_block<T>& bn)
{
  // LEFT
  copy_task<T>(bn.ghost_size, bn.left_handle, 0, bn.handle, bn.ghost_size, bn.preferred_device);

  // RIGHT
  copy_task<T>(bn.ghost_size, bn.right_handle, 0, bn.handle, bn.block_size, bn.preferred_device);
}

// Copy left/right handles from neighbours to the array
template <typename T>
void update_outer_interfaces(data_block<T>& bn, data_block<T>& left, data_block<T>& right)
{
  // update_outer_interface_left
  copy_task<T>(bn.ghost_size, bn.handle, 0, left.right_handle, 0, bn.preferred_device);

  // update_outer_interface_right
  copy_task<T>(bn.ghost_size, bn.handle, bn.ghost_size + bn.block_size, right.left_handle, 0, bn.preferred_device);
}

// bn1.array = bn.array
template <typename T>
void copy_array(data_block<T>& bn, data_block<T>& bn1)
{
  assert(bn.preferred_device == bn1.preferred_device);
  copy_task<T>(bn.block_size + 2 * bn.ghost_size, bn1.handle, 0, bn.handle, 0, bn.preferred_device);
}

int main(int argc, char** argv)
{
  int NITER         = 500;
  size_t NBLOCKS    = 4;
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

  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  // fprintf(stderr, "GOT %d devices\n", ndevs);

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

    Un.emplace_back(beg, end, 1ull);
    Un1.emplace_back(beg, end, 1ull);
  }

  for (size_t b = 0; b < NBLOCKS; b++)
  {
    Un[b].preferred_device  = b % ndevs;
    Un1[b].preferred_device = b % ndevs;
  }

  // Fill blocks with initial values. For the sake of simplicity, we are
  // using a synchronization primitive and host code, but this could have
  // been written asynchronously using host callbacks.
  for (size_t b = 0; b < NBLOCKS; b++)
  {
    size_t beg = b * BLOCK_SIZE;

    ctx.task(exec_place::host, Un1[b].handle.rw())->*[&](cudaStream_t stream, auto sUn1) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      double* Un1_vals = sUn1.data_handle();

      for (size_t local_idx = 0; local_idx < BLOCK_SIZE; local_idx++)
      {
        Un1_vals[local_idx + GHOST_SIZE] = U0[(beg + local_idx + TOTAL_SIZE) % TOTAL_SIZE];
      }
    };
  }

  for (int iter = 0; iter < NITER; iter++)
  {
    for (size_t b = 0; b < NBLOCKS; b++)
    {
      // Update the internal copies of the left and right boundaries
      update_inner_interfaces(Un1[b]);
    }

    for (size_t b = 0; b < NBLOCKS; b++)
    {
      // Apply ghost cells from neighbours to put then in the "center" array
      update_outer_interfaces(Un1[b], Un1[(b - 1 + NBLOCKS) % NBLOCKS], Un1[(b + 1) % NBLOCKS]);
    }

    // UPDATE Un from Un1
    for (size_t b = 0; b < NBLOCKS; b++)
    {
      stencil(Un[b], Un1[b]);
    }

    for (size_t b = 0; b < NBLOCKS; b++)
    {
      // Save Un into Un1
      copy_array(Un[b], Un1[b]);
    }

#if 0
    // We make sure that the total sum of elements remains constant
    if (iter % 250 == 0)
    {
      double check_sum = 0.0;
      for (size_t b = 0; b < NBLOCKS; b++)
      {
        check_sum += Un[b].check_sum();
      }

      // fprintf(stderr, "iter %d : CHECK SUM = %e\n", iter, check_sum);
    }
#endif
  }

  // In this stencil, the sum of the elements is supposed to be a constant
  double check_sum = 0.0;
  for (size_t b = 0; b < NBLOCKS; b++)
  {
    check_sum += Un[b].check_sum();
  }

  double err = fabs(check_sum - 1.0);
  EXPECT(err < 0.0001);

  ctx.finalize();
}
