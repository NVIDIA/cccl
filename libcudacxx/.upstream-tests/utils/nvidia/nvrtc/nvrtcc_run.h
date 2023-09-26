//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "nvrtcc_common.h"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

static void list_devices()
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices found: %d\n", device_count);

    int selected_device;
    cudaGetDevice(&selected_device);

    for (int dev = 0; dev < device_count; ++dev)
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);

        printf("Device %d: \"%s\", ", dev, device_prop.name);
        if(dev == selected_device)
            printf("Selected, ");
        else
            printf("Unused, ");

        printf("SM%d%d, %zu [bytes]\n",
            device_prop.major, device_prop.minor,
            device_prop.totalGlobalMem);
    }
}

static void load_and_run_gpu_code(const std::string input_file) {
    std::ifstream istr(input_file, std::ios::binary);
    std::vector<char> code(
        std::istreambuf_iterator<char>{istr},
        std::istreambuf_iterator<char>{} );
    istr.close();

    unsigned int thread_count = 1;
    unsigned int shmem_size = 1;

    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, code.data(), 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "main_kernel"));
    CUDA_SAFE_CALL(cuLaunchKernel(kernel,
        1, 1, 1,
        thread_count, 1, 1,
        shmem_size,
        NULL,
        NULL, 0));

    CUDA_API_CALL(cudaGetLastError());
    CUDA_API_CALL(cudaDeviceSynchronize());

    return;
}
