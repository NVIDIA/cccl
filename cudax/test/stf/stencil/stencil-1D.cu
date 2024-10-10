//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

static graph_ctx ctx;

/*
 * DATA BLOCKS
 *   | GHOSTS | DATA | GHOSTS |
 */
template <typename T>
class data_block
{
public:
  data_block(size_t beg, size_t end, size_t ghost_size)
      : ghost_size(ghost_size)
      , block_size(end - beg + 1)
      , array(new T[block_size + 2 * ghost_size])
      , left_interface(new T[ghost_size])
      , right_interface(new T[ghost_size])
      , handle(ctx.logical_data(array.get(), block_size + 2 * ghost_size))
      , left_handle(ctx.logical_data(left_interface.get(), ghost_size))
      , right_handle(ctx.logical_data(right_interface.get(), ghost_size))
  {}

  T* get_array_in_task()
  {
    return handle.instance().data_handle();
  }

  T* get_array()
  {
    return array.get();
  }

public:
  size_t ghost_size;
  size_t block_size;
  std::unique_ptr<T[]> array;
  std::unique_ptr<T[]> left_interface;
  std::unique_ptr<T[]> right_interface;

  // HANDLE = whole data + boundaries
  logical_data<slice<T>> handle;

  // A piece of data to store the left part of the block
  logical_data<slice<T>> left_handle;

  // A piece of data to store the right part of the block
  logical_data<slice<T>> right_handle;
};

template <typename T>
__global__ void copy_kernel(size_t cnt, T* dst, T* src)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < cnt; idx += blockDim.x * gridDim.x)
  {
    dst[idx] = src[idx];
  }
}

template <typename T>
__global__ void stencil_kernel(size_t cnt, size_t ghost_size, T* array, T* array1)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < cnt; idx += blockDim.x * gridDim.x)
  {
    int idx2    = idx + ghost_size;
    array[idx2] = 0.9 * array1[idx2] + 0.05 * array1[idx2 - 1] + 0.05 * array1[idx2 + 1];
  }
}

// bn1.array = bn.array
template <typename T>
void stencil(data_block<T>& bn, data_block<T>& bn1)
{
  ctx.task(bn.handle.rw(), bn1.handle.read())
      ->*[bs = bn.block_size, gs = bn.ghost_size](cudaStream_t stream, auto s1, auto s2) {
            stencil_kernel<<<256, 64, 0, stream>>>(bs, gs, s1.data_handle(), s2.data_handle());
          };
}

template <typename T>
void update_inner_interfaces(data_block<T>& bn)
{
  // LEFT
  ctx.task(bn.handle.read(), bn.left_handle.rw())->*[gs = bn.ghost_size](cudaStream_t stream, auto s1, auto s2) {
    copy_kernel<<<(gs > 64 ? 256 : 1), 64, 0, stream>>>(gs, s2.data_handle(), s1.data_handle() + gs);
  };

  // RIGHT
  ctx.task(bn.handle.read(), bn.right_handle.rw())
      ->*[bs = bn.block_size, gs = bn.ghost_size](cudaStream_t stream, auto s1, auto s2) {
            copy_kernel<<<(gs > 64 ? 256 : 1), 64, 0, stream>>>(gs, s2.data_handle(), s1.data_handle() + bs);
          };
}

template <typename T>
void update_outer_interfaces(data_block<T>& bn, data_block<T>& left, data_block<T>& right)
{
  ctx.task(bn.handle.rw(), left.right_handle.read())->*[gs = bn.ghost_size](cudaStream_t stream, auto s1, auto s2) {
    copy_kernel<<<(gs > 64 ? 256 : 1), 64, 0, stream>>>(gs, s1.data_handle(), s2.data_handle());
  };
  ctx.task(bn.handle.rw(), right.left_handle.read())
      ->*[bs = bn.block_size, gs = bn.ghost_size](cudaStream_t stream, auto s1, auto s2) {
            copy_kernel<<<(gs > 64 ? 256 : 1), 64, 0, stream>>>(gs, s1.data_handle() + gs + bs, s2.data_handle());
          };
}

// bn1.array = bn.array
template <typename T>
void copy_array(data_block<T>& bn, data_block<T>& bn1)
{
  ctx.task(bn1.handle.rw(), bn.handle.read())
      ->*[sz = bn.block_size + 2 * bn.ghost_size](cudaStream_t stream, auto s1, auto s2) {
            copy_kernel<<<256, 64, 0, stream>>>(sz, s1.data_handle(), s2.data_handle());
          };
}

int main(int argc, char** argv)
{
  size_t NBLOCKS    = 2;
  size_t BLOCK_SIZE = 1024 * 64;

  size_t TOTAL_SIZE = NBLOCKS * BLOCK_SIZE;

  std::vector<double> U0(NBLOCKS * BLOCK_SIZE);
  for (size_t idx = 0; idx < NBLOCKS * BLOCK_SIZE; idx++)
  {
    U0[idx] = (idx == 0) ? 1.0 : 0.0;
  }

  std::vector<data_block<double>> Un;
  std::vector<data_block<double>> Un1;

  // Create blocks and allocates host data
  for (size_t b = 0; b < NBLOCKS; b++)
  {
    int beg = b * BLOCK_SIZE;
    int end = (b + 1) * BLOCK_SIZE;

    Un.push_back(data_block<double>(beg, end, 1));
    Un1.push_back(data_block<double>(beg, end, 1));
  }

  // Fill blocks with initial values
  for (size_t b = 0; b < NBLOCKS; b++)
  {
    size_t beg = b * BLOCK_SIZE;
    // int end = (b+1)*BLOCK_SIZE;

    double* Un_vals  = Un[b].get_array();
    double* Un1_vals = Un1[b].get_array();
    // Attention, unusual loop: index goes all through BLOCK_SIZE inclusive.
    for (size_t local_idx = 0; local_idx <= BLOCK_SIZE; local_idx++)
    {
      Un_vals[local_idx]  = U0[(beg + local_idx - 1) % TOTAL_SIZE];
      Un1_vals[local_idx] = U0[(beg + local_idx - 1) % TOTAL_SIZE];
    }
  }

  // Create the graph - it starts out empty

  int NITER = 400;
  for (int iter = 0; iter < NITER; iter++)
  {
    // UPDATE Un from Un1
    for (size_t b = 0; b < NBLOCKS; b++)
    {
      stencil(Un[b], Un1[b]);
    }

    for (size_t b = 0; b < NBLOCKS; b++)
    {
      // Update the internal copies of the left and right boundaries
      update_inner_interfaces(Un[b]);
    }

    for (size_t b = 0; b < NBLOCKS; b++)
    {
      update_outer_interfaces(Un[b], Un[(b - 1 + NBLOCKS) % NBLOCKS], Un[(b + 1) % NBLOCKS]);
    }

    for (size_t b = 0; b < NBLOCKS; b++)
    {
      copy_array(Un[b], Un1[b]);
    }
  }

  ctx.submit();

  if (argc > 1)
  {
    std::cout << "Generating DOT output in " << argv[1] << std::endl;
    ctx.print_to_dot(argv[1]);
  }

  ctx.finalize();
}
