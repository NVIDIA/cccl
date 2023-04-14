//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_ARCH__
    #include <thread>
#else
    #if __CUDA_ARCH__ < 350
        #error "This test requires CUDA dynamic parallelism to work."
    #endif
#endif

template<typename... Fs>
__host__ __device__
void concurrent_agents_launch(Fs ...fs)
{
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,(
        assert(blockDim.x == sizeof...(Fs));
        using fptr = void (*)(void *);

        fptr device_threads[] = {
            [](void * data) {
                (*reinterpret_cast<Fs *>(data))();
            }...
        };

        void * device_thread_data[] = {
            reinterpret_cast<void *>(&fs)...
        };

        __syncthreads();

        device_threads[threadIdx.x](device_thread_data[threadIdx.x]);

        __syncthreads();
    ),(
        std::thread threads[]{
            std::thread{ std::forward<Fs>(fs) }...
        };

        for (auto && thread : threads)
        {
            thread.join();
        }
    ))
}

