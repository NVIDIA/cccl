/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cuda.h>

#include <string>

#include <c2h/catch2_test_helper.h>
#include <nvrtc.h>
#include <nvrtc_args.h>

TEST_CASE("Test nvrtc", "[test][nvrtc]")
{
  nvrtcProgram prog{};

  const char* src  = R"asdf(
    #include <cub/warp/warp_reduce.cuh>
    #include <cub/warp/warp_scan.cuh>
    #include <cub/warp/warp_exchange.cuh>
    #include <cub/warp/warp_load.cuh>
    #include <cub/warp/warp_store.cuh>
    #include <cub/warp/warp_merge_sort.cuh>
    #include <cub/block/block_adjacent_difference.cuh>
    #include <cub/block/block_discontinuity.cuh>
    #include <cub/block/block_exchange.cuh>
    #include <cub/block/block_histogram.cuh>
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_merge_sort.cuh>
    #include <cub/block/block_radix_rank.cuh>
    #include <cub/block/block_radix_sort.cuh>
    #include <cub/block/block_reduce.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/device/dispatch/kernels/reduce.cuh>
    #include <cub/device/dispatch/kernels/for_each.cuh>
    #include <cub/device/dispatch/kernels/scan.cuh>
    #include <cub/device/dispatch/kernels/merge_sort.cuh>

    extern "C" __global__ void kernel(int *ptr, int *errors)
    {
      constexpr int items_per_thread = 4;
      constexpr int threads_per_block = 128;
      using warp_load_t = cub::WarpLoad<int, items_per_thread>;
      using warp_load_storage_t = warp_load_t::TempStorage;

      using warp_exchange_t = cub::WarpExchange<int, items_per_thread>;
      using warp_exchange_storage_t = warp_exchange_t::TempStorage;

      using warp_reduce_t = cub::WarpReduce<int>;
      using warp_reduce_storage_t = warp_reduce_t::TempStorage;

      using warp_merge_sort_t = cub::WarpMergeSort<int, items_per_thread>;
      using warp_merge_sort_storage_t = warp_merge_sort_t::TempStorage;

      using warp_scan_t = cub::WarpScan<int>;
      using warp_scan_storage_t = warp_scan_t::TempStorage;

      using warp_store_t = cub::WarpStore<int, items_per_thread>;
      using warp_store_storage_t = warp_store_t::TempStorage;

      __shared__ warp_load_storage_t warp_load_storage;
      __shared__ warp_exchange_storage_t warp_exchange_storage;
      __shared__ warp_reduce_storage_t warp_reduce_storage;
      __shared__ warp_merge_sort_storage_t warp_merge_sort_storage;
      __shared__ warp_scan_storage_t warp_scan_storage;
      __shared__ warp_store_storage_t warp_store_storage;

      int items[items_per_thread];
      if (threadIdx.x < 32)
      {
        // Test warp load
        warp_load_t(warp_load_storage).Load(ptr, items);

        for (int i = 0; i < items_per_thread; i++)
        {
          if (items[i] != (i + threadIdx.x * items_per_thread))
          {
            atomicAdd(errors, 1);
          }
        }

        // Test warp exchange
        warp_exchange_t(warp_exchange_storage).BlockedToStriped(items, items);

        for (int i = 0; i < items_per_thread; i++)
        {
          if (items[i] != (i * 32 + threadIdx.x))
          {
            atomicAdd(errors, 1);
          }
        }

        // Test warp reduce
        const int sum = warp_reduce_t(warp_reduce_storage).Sum(items[0]);
        if (threadIdx.x == 0)
        {
          if (sum != (32 * (32 - 1) / 2))
          {
            atomicAdd(errors, 1);
          }
        }

        // Test warp scan
        int prefix_sum{};
        warp_scan_t(warp_scan_storage).InclusiveSum(items[0], prefix_sum);
        if (prefix_sum != (threadIdx.x * (threadIdx.x + 1) / 2))
        {
          atomicAdd(errors, 1);
        }

        // Test warp merge sort
        warp_merge_sort_t(warp_merge_sort_storage).Sort(
          items,
          [](int a, int b) { return a < b; });

        for (int i = 0; i < items_per_thread; i++)
        {
          if (items[i] != (i + threadIdx.x * items_per_thread))
          {
            atomicAdd(errors, 1);
          }
        }

        // Test warp store
        warp_store_t(warp_store_storage).Store(ptr, items);
      }
      __syncthreads();

      using block_load_t = cub::BlockLoad<int, threads_per_block, items_per_thread>;
      using block_load_storage_t = block_load_t::TempStorage;

      using block_exchange_t = cub::BlockExchange<int, threads_per_block, items_per_thread>;
      using block_exchange_storage_t = block_exchange_t::TempStorage;

      using block_reduce_t = cub::BlockReduce<int, threads_per_block>;
      using block_reduce_storage_t = block_reduce_t::TempStorage;

      using block_scan_t = cub::BlockScan<int, threads_per_block>;
      using block_scan_storage_t = block_scan_t::TempStorage;

      using block_radix_sort_t = cub::BlockRadixSort<int, threads_per_block, items_per_thread>;
      using block_radix_sort_storage_t = block_radix_sort_t::TempStorage;

      using block_store_t = cub::BlockStore<int, threads_per_block, items_per_thread>;
      using block_store_storage_t = block_store_t::TempStorage;

      __shared__ block_load_storage_t block_load_storage;
      __shared__ block_exchange_storage_t block_exchange_storage;
      __shared__ block_reduce_storage_t block_reduce_storage;
      __shared__ block_scan_storage_t block_scan_storage;
      __shared__ block_radix_sort_storage_t block_radix_sort_storage;
      __shared__ block_store_storage_t block_store_storage;

      // Test block load
      block_load_t(block_load_storage).Load(ptr, items);

      for (int i = 0; i < items_per_thread; i++)
      {
        if (items[i] != (i + threadIdx.x * items_per_thread))
        {
          atomicAdd(errors, 1);
        }
      }

      // Test block exchange
      block_exchange_t(block_exchange_storage).BlockedToStriped(items, items);

      for (int i = 0; i < items_per_thread; i++)
      {
        if (items[i] != (i * threads_per_block + threadIdx.x))
        {
          atomicAdd(errors, 1);
        }
      }

      // Test block reduce
      const int sum = block_reduce_t(block_reduce_storage).Sum(items[0]);
      if (threadIdx.x == 0)
      {
        if (sum != (threads_per_block * (threads_per_block - 1) / 2))
        {
          atomicAdd(errors, 1);
        }
      }

      // Test block scan
      int prefix_sum{};
      block_scan_t(block_scan_storage).InclusiveSum(items[0], prefix_sum);
      if (prefix_sum != (threadIdx.x * (threadIdx.x + 1) / 2))
      {
        atomicAdd(errors, 1);
      }

      // Test block radix sort
      block_radix_sort_t(block_radix_sort_storage).SortDescending(items);

      // Test block store
      block_store_t(block_store_storage).Store(ptr, items);
    }
)asdf";
  const char* name = "test";

  REQUIRE(NVRTC_SUCCESS == nvrtcCreateProgram(&prog, src, name, 0, nullptr, nullptr));

  int ptx_version{};
  cub::PtxVersion(ptx_version);
  const std::string arch = std::string("-arch=sm_") + std::to_string(ptx_version / 10);
  const std::string std  = std::string("-std=c++") + std::to_string(_CCCL_STD_VER - 2000);

  constexpr int num_includes         = 6;
  const char* includes[num_includes] = {
    NVRTC_CUB_PATH, NVRTC_THRUST_PATH, NVRTC_LIBCUDACXX_PATH, NVRTC_CTK_PATH, arch.c_str(), std.c_str()};

  std::size_t log_size{};
  nvrtcResult compile_result = nvrtcCompileProgram(prog, num_includes, includes);

  REQUIRE(NVRTC_SUCCESS == nvrtcGetProgramLogSize(prog, &log_size));

  std::unique_ptr<char[]> log{new char[log_size]};
  REQUIRE(NVRTC_SUCCESS == nvrtcGetProgramLog(prog, log.get()));
  INFO("nvrtc log = " << log.get());
  REQUIRE(NVRTC_SUCCESS == compile_result);

  std::size_t code_size{};
  REQUIRE(NVRTC_SUCCESS == nvrtcGetCUBINSize(prog, &code_size));

  std::unique_ptr<char[]> code{new char[code_size]};
  REQUIRE(NVRTC_SUCCESS == nvrtcGetCUBIN(prog, code.get()));
  REQUIRE(NVRTC_SUCCESS == nvrtcDestroyProgram(&prog));

  CUcontext context{};
  CUdevice device{};
  CUmodule module{};
  CUfunction kernel{};

  REQUIRE(CUDA_SUCCESS == cuInit(0));
  REQUIRE(CUDA_SUCCESS == cuDeviceGet(&device, 0));
  REQUIRE(CUDA_SUCCESS == cuCtxCreate(&context, 0, device));
  REQUIRE(CUDA_SUCCESS == cuModuleLoadDataEx(&module, code.get(), 0, 0, 0));
  REQUIRE(CUDA_SUCCESS == cuModuleGetFunction(&kernel, module, "kernel"));

  // Generate input for execution, and create output buffers.
  constexpr int threads_in_block = 128;
  constexpr int items_per_thread = 4;
  constexpr int tile_size        = threads_in_block * items_per_thread;

  CUdeviceptr d_ptr{};
  REQUIRE(CUDA_SUCCESS == cuMemAlloc(&d_ptr, tile_size * sizeof(int)));

  CUdeviceptr d_err{};
  REQUIRE(CUDA_SUCCESS == cuMemAlloc(&d_err, sizeof(int)));

  int h_ptr[tile_size];
  for (int i = 0; i < tile_size; i++)
  {
    h_ptr[i] = i;
  }
  REQUIRE(CUDA_SUCCESS == cuMemcpyHtoD(d_ptr, h_ptr, tile_size * sizeof(int)));

  int h_err{0};
  REQUIRE(CUDA_SUCCESS == cuMemcpyHtoD(d_err, &h_err, sizeof(int)));

  void* args[] = {&d_ptr, &d_err};

  REQUIRE(CUDA_SUCCESS == cuLaunchKernel(kernel, 1, 1, 1, threads_in_block, 1, 1, 0, nullptr, args, 0));
  REQUIRE(CUDA_SUCCESS == cuCtxSynchronize());
  REQUIRE(CUDA_SUCCESS == cuMemcpyDtoH(h_ptr, d_ptr, tile_size * sizeof(int)));
  REQUIRE(CUDA_SUCCESS == cuMemcpyDtoH(&h_err, d_err, sizeof(int)));

  REQUIRE(h_err == 0);
  for (int i = 0; i < tile_size; i++)
  {
    const int actual   = h_ptr[i];
    const int expected = tile_size - i - 1;
    REQUIRE(actual == expected);
  }

  REQUIRE(CUDA_SUCCESS == cuMemFree(d_ptr));
  REQUIRE(CUDA_SUCCESS == cuMemFree(d_err));
  REQUIRE(CUDA_SUCCESS == cuModuleUnload(module));
  REQUIRE(CUDA_SUCCESS == cuCtxDestroy(context));
}
