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
 * @brief A parallel scan algorithm using CUB kernels
 *
 */

#include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__host__ __device__ double X0(int i)
{
  return sin((double) i);
}

/**
 * @brief Performs an inclusive scan on a logical data slice using CUB.
 *
 * This function determines the temporary device storage requirements for a scan, allocates
 * temporary storage, and then performs the scan using the CUB library. The scan is performed
 * in place, modifying the input `logical_data` slice.
 *
 * @tparam Ctx The context type for data management and task execution.
 * @tparam T The data type of the elements in the `logical_data` slice.
 *
 * @param ctx Reference to the context object.
 * @param ld Reference to the `logical_data` object containing the data slice.
 * @param dp The `data_place` enum specifying where the data should reside (e.g., CPU, GPU).
 */
template <typename Ctx, typename T>
void scan(Ctx& ctx, logical_data<slice<T>>& ld, data_place dp)
{
  // Determine temporary device storage requirements
  auto num_items  = int(ld.shape().size());
  size_t tmp_size = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp_size, (T*) nullptr, (T*) nullptr, num_items);

  // fprintf(stderr, "SCAN %ld items TMP = %ld bytes\n", num_items, tmp_size);

  logical_data<slice<char>> ltmp = ctx.logical_data(shape_of<slice<char>>(tmp_size)).set_symbol("tmp");

  ctx.task(ld.rw(mv(dp)), ltmp.write()).set_symbol("scan " + ld.get_symbol())
      ->*[=](cudaStream_t stream, auto d, auto tmp) mutable {
            T* buffer = d.data_handle();
            cub::DeviceScan::InclusiveSum(tmp.data_handle(), tmp_size, buffer, buffer, num_items, stream);
          };
}

int main(int argc, char** argv)
{
  stream_ctx ctx;
  // graph_ctx ctx;

  // const size_t N = 128ULL*1024ULL*1024ULL;
  size_t nmb = 128;
  if (argc > 1)
  {
    nmb = atoi(argv[1]);
  }

  int check = 0;
  if (argc > 2)
  {
    check = atoi(argv[2]);
  }

  const size_t N = nmb * 1024ULL * 1024ULL;

  const int ndevs      = cuda_try<cudaGetDeviceCount>();
  const size_t NBLOCKS = 2 * ndevs;

  size_t BLOCK_SIZE = (N + NBLOCKS - 1) / NBLOCKS;

  auto fixed_alloc = block_allocator<fixed_size_allocator>(ctx, BLOCK_SIZE * sizeof(double));
  ctx.set_allocator(fixed_alloc);

  // dummy task to initialize the allocator XXX
  {
    auto ldummy = ctx.logical_data(shape_of<slice<double>>(NBLOCKS)).set_symbol("dummy");
    ctx.task(ldummy.write(data_place::managed))->*[](cudaStream_t, auto) {};
  }

  std::vector<double> X(N);
  std::vector<logical_data<slice<double>>> lX(NBLOCKS);
  logical_data<slice<double>> laux;

  // If we were to register each part one by one, there could be pages which
  // cross multiple parts, and the pinning operation would fail.
  cuda_safe_call(cudaHostRegister(&X[0], N * sizeof(double), cudaHostRegisterPortable));

  for (size_t b = 0; b < NBLOCKS; b++)
  {
    size_t start = b * BLOCK_SIZE;
    size_t end   = std::min(start + BLOCK_SIZE, N);
    lX[b]        = ctx.logical_data(&X[start], {end - start}).set_symbol("X_" + std::to_string(b));

    // No need to move this back to the host if we do not check the result
    if (!check)
    {
      lX[b].set_write_back(false);
    }
  }

  for (size_t b = 0; b < NBLOCKS; b++)
  {
    cuda_safe_call(cudaSetDevice(b % ndevs));
    size_t start = b * BLOCK_SIZE;
    ctx.parallel_for(lX[b].shape(), lX[b].write())->*[=] _CCCL_DEVICE(size_t i, auto lx) {
      lx(i) = X0(i + start);
    };
  }

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));
  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  for (size_t k = 0; k < 100; k++)
  {
    // Create an auxiliary temporary buffer and blank it
    laux = ctx.logical_data(shape_of<slice<double>>(NBLOCKS)).set_symbol("aux");
    ctx.parallel_for(laux.shape(), laux.write(data_place::managed)).set_symbol("init_aux")
        ->*[] _CCCL_DEVICE(size_t i, auto aux) {
              aux(i) = 0.0;
            };

    // Scan each block
    for (size_t b = 0; b < NBLOCKS; b++)
    {
      cuda_safe_call(cudaSetDevice(b % ndevs));
      scan(ctx, lX[b], data_place::device(b % ndevs));
    }

    for (size_t b = 0; b < NBLOCKS; b++)
    {
      //        cuda_safe_call(cudaSetDevice(b % ndevs));

      ctx.parallel_for(exec_place::device(0),
                       box({b, b + 1}),
                       lX[b].read(data_place::device(b % ndevs)),
                       laux.rw(data_place::managed))
          .set_symbol("store sum X_" + std::to_string(b))
          ->*[] _CCCL_DEVICE(size_t ind, auto Xb, auto aux) {
                aux(ind) = Xb(Xb.extent(0) - 1);
              };
    }

    // Prefix sum of the per-block sums
    scan(ctx, laux, data_place::managed);

    // Add partial sum of Xi to X(i+1)
    for (size_t b = 1; b < NBLOCKS; b++)
    {
      cuda_safe_call(cudaSetDevice(b % ndevs));
      ctx.parallel_for(lX[b].shape(), lX[b].rw(), laux.read(data_place::managed))
          .set_symbol("add X_" + std::to_string(b))
          ->*[=] _CCCL_DEVICE(size_t i, auto Xb, auto aux) {
                Xb(i) += aux(b - 1);
              };
    }
  }

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  ctx.finalize();

  float ms = 0;
  cuda_safe_call(cudaEventElapsedTime(&ms, start, stop));

  fprintf(stdout, "%zu %f ms\n", N / 1024 / 1024, ms);

  if (check)
  {
#if 0
    for (size_t i = 0; i < N; i++) {
        EXPECT(fabs(X[i] - expected_result[i]) < 0.00001);
    }
#endif

#if 1
    fprintf(stderr, "Checking result ...\n");
    EXPECT(fabs(X[0] - X0(0)) < 0.00001);
    for (size_t i = 0; i < N; i++)
    {
      EXPECT(fabs(X[i] - X[i - 1] - X0(i)) < 0.00001);
    }
  }
#endif
}
