//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: c++98, c++03

#include <cuda/annotated_ptr>
#include <cooperative_groups.h>

#ifndef __CUDA_ARCH__
#include <array>
#endif

#if defined(DEBUG)
    #define DPRINTF(...) { printf(__VA_ARGS__); }
#else
    #define DPRINTF(...) do {} while (false)
#endif

#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(disable: 4505)
#endif

template <typename ... T>
__host__ __device__ constexpr bool unused(T...) {return true;}

// ************************ Device code ***************************************
static __device__
void shared_mem_test_dev() {
    __shared__ int smem[42];
    smem[10] = 42;
    assert(smem[10] == 42);

    cuda::annotated_ptr<int, cuda::access_property::shared> p{smem + 10};

    assert(*p == 42);
}

static __global__
void shared_mem_test() {
    shared_mem_test_dev();
};

static __device__
void annotated_ptr_timing_dev(int * in, int * out) {
    cuda::access_property ap(cuda::access_property::persisting{});
    // Retrieve global id
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    cuda::annotated_ptr<int, cuda::access_property> in_ann{in, ap};
    cuda::annotated_ptr<int, cuda::access_property> out_ann{out, ap};

    DPRINTF("&out[i]:%p = &in[i]:%p for i = %d\n", &out[i], &in[i], i);
    DPRINTF("&out[i]:%p = &in_ann[i]:%p for i = %d\n", &out_ann[i], &in_ann[i], i);

    out_ann[i] = in_ann[i];
};

static __global__
void annotated_ptr_timing(int * in, int * out) {
    annotated_ptr_timing_dev(in, out);
}

static __device__
void ptr_timing_dev(int * in, int * out) {
    // Retrieve global id
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    DPRINTF("&out[i]:%p = &in[i]:%p for i = %d\n", &out[i], &in[i], i);
    out[i] = in[i];
};

static __global__
void ptr_timing(int * in, int * out) {
    ptr_timing_dev(in, out);
};

// ************************ Host/device code ***************************************
__device__ __host__
void assert_rt_wrap(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
#ifndef __CUDACC_RTC__
        printf("assert: %s %s %d\n", cudaGetErrorString(code), file, line);
#endif
        assert(code == cudaSuccess);
    }
}
#define assert_rt(ret) { assert_rt_wrap((ret), __FILE__, __LINE__); }

__device__ __host__ __noinline__
void test_access_property_interleave() {
    (void)cuda::access_property::shared{};
    (void)cuda::access_property::global{};
    assert(cuda::access_property::persisting{} == cudaAccessPropertyPersisting);
    assert(cuda::access_property::streaming{} == cudaAccessPropertyStreaming);
    assert(cuda::access_property::normal{} == cudaAccessPropertyNormal);

    const uint64_t INTERLEAVE_NORMAL           = uint64_t{0x10F0000000000000};
    const uint64_t INTERLEAVE_NORMAL_DEMOTE    = uint64_t{0x16F0000000000000};
    const uint64_t INTERLEAVE_PERSISTING       = uint64_t{0x14F0000000000000};
    const uint64_t INTERLEAVE_STREAMING        = uint64_t{0x12F0000000000000};
    cuda::access_property ap(cuda::access_property::persisting{});
    cuda::access_property ap2;

    assert(INTERLEAVE_PERSISTING == static_cast<uint64_t>(ap));
    assert(static_cast<uint64_t>(ap2) == INTERLEAVE_NORMAL);

    ap = cuda::access_property(cuda::access_property::normal());
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL_DEMOTE);

    ap = cuda::access_property(cuda::access_property::streaming());
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_STREAMING);

    ap = cuda::access_property(cuda::access_property::normal(), 2.0f);
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL_DEMOTE);
}

__device__ __host__ __noinline__
void test_access_property_block() {
    //assuming ptr address is 0;
    const size_t TOTAL_BYTES = 0xFFFFFFFF;
    const size_t HIT_BYTES = 0xFFFFFFFF;
    const size_t BLOCK_0ADDR_PERSISTHIT_STREAMISS_MAXBYTES = size_t{0x1DD00FE000000000};
    const uint64_t INTERLEAVE_NORMAL = uint64_t{0x10F0000000000000};

    cuda::access_property ap(0x0, HIT_BYTES, TOTAL_BYTES, cuda::access_property::persisting{}, cuda::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == BLOCK_0ADDR_PERSISTHIT_STREAMISS_MAXBYTES);

    ap = cuda::access_property(0x0, 0xFFFFFFFF, 0xFFFFFFFFF, cuda::access_property::persisting{}, cuda::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL);

    ap = cuda::access_property(0x0, 0xFFFFFFFFF, 0xFFFFFFFF, cuda::access_property::persisting{}, cuda::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL);

    ap = cuda::access_property(0x0, 0, 0, cuda::access_property::persisting{}, cuda::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL);

    for (size_t ptr = 1; ptr < size_t{0xFFFFFFFF}; ptr <<= 1) {
        for (size_t hit = 1; hit < size_t{0xFFFFFFFF}; hit <<= 1) {
            ap = cuda::access_property((void*)ptr, hit, hit, cuda::access_property::persisting{}, cuda::access_property::streaming{});
            DPRINTF("Block encoding PTR:%p, hit:%p, block encoding:%p\n", ptr, hit, static_cast<uint64_t>(ap));
        }
    }
}

__device__ __host__ __noinline__
void test_access_property_functions() {
    size_t ARR_SZ = 1 << 10;
    int* arr0 = nullptr;
    int* arr1 = nullptr;
    cuda::access_property ap(cuda::access_property::persisting{});
    cuda::access_property a;
    unused(a);
    cuda::access_property as(cuda::access_property::streaming{});

#ifdef __CUDA_ARCH__
    arr0 = (int*)malloc(ARR_SZ * sizeof(int));
    arr1 = (int*)malloc(ARR_SZ * sizeof(int));
#else
    assert_rt(cudaMallocManaged((void**) &arr0, ARR_SZ * sizeof(int)));
    assert_rt(cudaMallocManaged((void**) &arr1, ARR_SZ * sizeof(int)));
    assert_rt(cudaDeviceSynchronize());
#endif

    cuda::discard_memory(arr0, ARR_SZ);
    arr0 = cuda::associate_access_property(arr0, ap);
    arr1 = cuda::associate_access_property(arr1, as);
    cuda::apply_access_property(arr0, ARR_SZ, cuda::access_property::persisting{});
    cuda::apply_access_property(arr1, ARR_SZ, cuda::access_property::normal{});

#ifdef __CUDA_ARCH__
    free(arr0);
    free(arr1);
#else
    assert_rt(cudaFree(arr0));
    assert_rt(cudaFree(arr1));
#endif

}

__device__ __host__ __noinline__
void test_annotated_ptr_basic() {
    cuda::access_property ap(cuda::access_property::persisting{});
    static const size_t ARR_SZ = 1 << 10;
    int* array0 = new int[ARR_SZ];
    int* array1 = new int[ARR_SZ];
    cuda::annotated_ptr<int, cuda::access_property> array_anno_ptr{array0, ap};
    cuda::annotated_ptr<int, cuda::access_property> array0_anno_ptr0{array0, ap};
    cuda::annotated_ptr<int, cuda::access_property> array0_anno_ptr1 = array0_anno_ptr0;
    cuda::annotated_ptr<int, cuda::access_property> array0_anno_ptr2{array0_anno_ptr0};
    cuda::annotated_ptr<int, cuda::access_property> array1_anno_ptr{array1, ap};
#ifndef __CUDA_ARCH__
    cuda::annotated_ptr<int, cuda::access_property::shared> shared_ptr1;
    cuda::annotated_ptr<int, cuda::access_property::shared> shared_ptr2;

    shared_ptr1 = shared_ptr2;
    unused(shared_ptr1);

    //Check on host the arrays through annotated_ptr ops
    std::array<int, 3> a1{3, 2, 1};
    cuda::annotated_ptr<std::array<int, 3>, cuda::access_property> anno_ptr{&a1, ap};
    assert(anno_ptr->at(0) == 3);
#endif

    //Fill the arrays
    for (size_t i = 0; i < ARR_SZ; ++i) {
        array0[i] = static_cast<int>(i);
        array1[i] = static_cast<int>(ARR_SZ - i);
    }

    assert((bool)array0_anno_ptr0 == true);
    assert(array0_anno_ptr0.get() == array0);

    for (size_t i = 0; i < ARR_SZ; ++i) {
        assert(array0_anno_ptr0[i] == static_cast<int>(i));
        assert(array0_anno_ptr2[i] == static_cast<int>(i));
        assert(&array0[i] == &array0_anno_ptr0[i]);
        assert(&array0[i] == &array0_anno_ptr1[i]);
    }

    for (size_t i = 0; i < ARR_SZ; ++i) {
        assert(array1_anno_ptr[i] == array1[i]);
    }

    delete[] array0;
    delete[] array1;
}

__device__ __host__ __noinline__
void test_annotated_ptr_launch_kernel() {
#ifndef __CUDA_ARCH__
    static const size_t ARR_SZ          = 1 << 22;
    static const size_t THREAD_CNT      = 128;
    static const size_t BLOCK_CNT       = ARR_SZ / THREAD_CNT;
    const dim3 threads(THREAD_CNT, 1, 1), blocks(BLOCK_CNT, 1, 1);
    cudaEvent_t start, stop;
#else
    static const size_t ARR_SZ     = 1 << 10;
#endif
    int* arr0 = nullptr;
    int* arr1 = nullptr;
    float annotated_time = 0.f, pointer_time = 0.f;

#ifdef __CUDA_ARCH__
    arr0 = (int*)malloc(ARR_SZ * sizeof(int));
    arr1 = (int*)malloc(ARR_SZ * sizeof(int));
#else
    assert_rt(cudaMallocManaged((void**) &arr0, ARR_SZ * sizeof(int)));
    assert_rt(cudaMallocManaged((void**) &arr1, ARR_SZ * sizeof(int)));
    assert_rt(cudaDeviceSynchronize());
#endif

#ifdef __CUDA_ARCH__
    shared_mem_test_dev();
#else
    shared_mem_test<<<1, 1, 0, 0>>>();
    assert_rt(cudaStreamSynchronize(0));
#endif


#ifdef __CUDA_ARCH__
    ptr_timing_dev(arr0, arr1);
#else
    ptr_timing<<<blocks,threads>>>(arr0, arr1);
    assert_rt(cudaDeviceSynchronize());
#endif

    for (size_t i = 0; i < ARR_SZ; ++i) {
        arr0[i] = static_cast<int>(i);
        arr1[i] = 0;
    }

#ifdef __CUDA_ARCH__
    ptr_timing_dev(arr0, arr1);
#else
    assert_rt(cudaDeviceSynchronize());
    assert_rt(cudaEventCreate(&start));
    assert_rt(cudaEventCreate(&stop));
    assert_rt(cudaEventRecord(start));
    ptr_timing<<<blocks,threads>>>(arr0, arr1);
    assert_rt(cudaEventRecord(stop));
    assert_rt(cudaEventSynchronize(stop));
    assert_rt(cudaEventElapsedTime(&pointer_time, start, stop));
    assert_rt(cudaEventDestroy(start));
    assert_rt(cudaEventDestroy(stop));
    assert_rt(cudaDeviceSynchronize());

    for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int)i) {
            DPRINTF("arr1[%d] == %d, should be:%d\n", i, arr1[i], i);
            assert(arr1[i] == static_cast<int>(i));
        }

        arr1[i] = 0;
    }
#endif

#ifdef __CUDA_ARCH__
    annotated_ptr_timing_dev(arr0, arr1);
#else
    assert_rt(cudaDeviceSynchronize());
    annotated_ptr_timing<<<blocks,threads>>>(arr0, arr1);
    assert_rt(cudaDeviceSynchronize());
#endif

    for (size_t i = 0; i < ARR_SZ; ++i) {
        arr0[i] = static_cast<int>(i);
        arr1[i] = 0;
    }

#ifdef __CUDA_ARCH__
    annotated_ptr_timing_dev(arr0, arr1);
#else
    assert_rt(cudaDeviceSynchronize());
    assert_rt(cudaEventCreate(&start));
    assert_rt(cudaEventCreate(&stop));
    assert_rt(cudaEventRecord(start));
    annotated_ptr_timing<<<blocks,threads>>>(arr0, arr1);
    assert_rt(cudaEventRecord(stop));
    assert_rt(cudaEventSynchronize(stop));
    assert_rt(cudaEventElapsedTime(&annotated_time, start, stop));
    assert_rt(cudaEventDestroy(start));
    assert_rt(cudaEventDestroy(stop));
    assert_rt(cudaDeviceSynchronize());

    for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int)i) {
            DPRINTF("arr1[%d] == %d, should be:%d\n", i, arr1[i], i);
            assert(arr1[i] == static_cast<int>(i));
        }

        arr1[i] = 0;
    }
#endif

#ifdef __CUDA_ARCH__
    free(arr0);
    free(arr1);
#else
    assert_rt(cudaFree(arr0));
    assert_rt(cudaFree(arr1));
#endif
    printf("array(ms):%f, arrotated_ptr(ms):%f\n",
            pointer_time, annotated_time);
}

__device__ __host__ __noinline__
void test_annotated_ptr_functions() {
    size_t ARR_SZ = 1 << 10;
    int* arr0 = nullptr;
    int* arr1 = nullptr;
    cuda::access_property ap(cuda::access_property::persisting{});
    cuda::barrier<cuda::thread_scope_system> bar0, bar1, bar2, bar3;
    init(&bar0, 1);
    init(&bar1, 1);
    init(&bar2, 1);
    init(&bar3, 1);

#ifdef __CUDA_ARCH__
    arr0 = (int*)malloc(ARR_SZ * sizeof(int));
    arr1 = (int*)malloc(ARR_SZ * sizeof(int));

    auto group = cooperative_groups::this_thread_block();
#else
    assert_rt(cudaMallocManaged((void**) &arr0, ARR_SZ * sizeof(int)));
    assert_rt(cudaMallocManaged((void**) &arr1, ARR_SZ * sizeof(int)));
    assert_rt(cudaDeviceSynchronize());
#endif

    cuda::annotated_ptr<int, cuda::access_property> ann0{arr0, ap};
    cuda::annotated_ptr<int, cuda::access_property> ann1{arr1, ap};

    for (size_t i = 0; i < ARR_SZ; ++i) {
        arr0[i] = static_cast<int>(i);
        arr1[i] = 0;
    }

    cuda::memcpy_async(ann1, ann0, ARR_SZ * sizeof(int), bar0);
    bar0.arrive_and_wait();

    for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int)i) {
            DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
            assert(arr1[i] == static_cast<int>(i));
        }

        arr1[i] = 0;
    }

    cuda::memcpy_async(arr1, ann0, ARR_SZ * sizeof(int), bar1);
    bar1.arrive_and_wait();

    for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int)i) {
            DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
            assert(arr1[i] == static_cast<int>(i));
        }

        arr1[i] = 0;
    }

#ifdef __CUDA_ARCH__
    cuda::memcpy_async(group, ann1, ann0, ARR_SZ * sizeof(int), bar2);
    bar2.arrive_and_wait();

    for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int)i) {
            DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
            assert(arr1[i] == i);
        }

        arr1[i] = 0;
    }

    cuda::memcpy_async(group, arr1, ann0, ARR_SZ * sizeof(int), bar3);
    bar3.arrive_and_wait();

    for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int)i) {
            DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
            assert(arr1[i] == i);
        }

        arr1[i] = 0;
    }
#endif

#ifdef __CUDA_ARCH__
    free(arr0);
    free(arr1);
#else
    assert_rt(cudaFree(arr0));
    assert_rt(cudaFree(arr1));
#endif
}
