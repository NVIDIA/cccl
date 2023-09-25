/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/******************************************************************************
 * Test of iterator utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <stdio.h>
#include <typeinfo>

#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/tex_obj_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


template <typename T>
struct TransformOp
{
    // Increment transform
    __host__ __device__ __forceinline__ T operator()(T input) const
    {
        T addend;
        InitValue(INTEGER_SEED, addend, 1);
        return input + addend;
    }
};

struct SelectOp
{
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(T input)
    {
        return true;
    }
};


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Test random access input iterator
 */
template <
    typename InputIteratorT,
    typename T>
__global__ void Kernel(
    InputIteratorT    d_in,
    T                 *d_out,
    InputIteratorT    *d_itrs)
{
    d_out[0] = *d_in;               // Value at offset 0
    d_out[1] = d_in[100];           // Value at offset 100
    d_out[2] = *(d_in + 1000);      // Value at offset 1000
    d_out[3] = *(d_in + 10000);     // Value at offset 10000

    d_in++;
    d_out[4] = d_in[0];             // Value at offset 1

    d_in += 20;
    d_out[5] = d_in[0];             // Value at offset 21
    d_itrs[0] = d_in;               // Iterator at offset 21

    d_in -= 10;
    d_out[6] = d_in[0];             // Value at offset 11;

    d_in -= 11;
    d_out[7] = d_in[0];             // Value at offset 0
    d_itrs[1] = d_in;               // Iterator at offset 0
}



//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------


/**
 * Run iterator test on device
 */
template <
    typename        InputIteratorT,
    typename        T,
    int             TEST_VALUES>
void Test(
    InputIteratorT  d_in,
    T               (&h_reference)[TEST_VALUES])
{
    // Allocate device arrays
    T                 *d_out    = NULL;
    InputIteratorT    *d_itrs   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out,     sizeof(T) * TEST_VALUES));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_itrs,    sizeof(InputIteratorT) * 2));

    int compare;

    // Run unguarded kernel
    Kernel<<<1, 1>>>(d_in, d_out, d_itrs);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Check results
    compare = CompareDeviceResults(h_reference, d_out, TEST_VALUES, g_verbose, g_verbose);
    printf("\tValues: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check iterator at offset 21
    InputIteratorT h_itr = d_in + 21;
    compare = CompareDeviceResults(&h_itr, d_itrs, 1, g_verbose, g_verbose);
    printf("\tIterators: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Check iterator at offset 0
    compare = CompareDeviceResults(&d_in, d_itrs + 1, 1, g_verbose, g_verbose);
    printf("\tIterators: %s\n", (compare) ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    // Cleanup
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_itrs) CubDebugExit(g_allocator.DeviceFree(d_itrs));
}


/**
 * Test constant iterator
 */
template <typename T>
void TestConstant(T base)
{
    printf("\nTesting constant iterator on type %s (base: %lld)\n", typeid(T).name(), (unsigned long long) (base)); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    T h_reference[8] = {base, base, base, base, base, base, base, base};
    ConstantInputIterator<T> d_itr(base);
    Test(d_itr, h_reference);
}


/**
 * Test counting iterator
 */
template <typename T>
void TestCounting(T base)
{
    printf("\nTesting counting iterator on type %s (base: %d) \n", typeid(T).name(), int(base)); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = static_cast<T>(base + 0);          // Value at offset 0
    h_reference[1] = static_cast<T>(base + 100);        // Value at offset 100
    h_reference[2] = static_cast<T>(base + 1000);       // Value at offset 1000
    h_reference[3] = static_cast<T>(base + 10000);      // Value at offset 10000
    h_reference[4] = static_cast<T>(base + 1);          // Value at offset 1
    h_reference[5] = static_cast<T>(base + 21);         // Value at offset 21
    h_reference[6] = static_cast<T>(base + 11);         // Value at offset 11
    h_reference[7] = static_cast<T>(base + 0);          // Value at offset 0;

    CountingInputIterator<T> d_itr(base);
    Test(d_itr, h_reference);
}


/**
 * Test modified iterator
 */
template <typename T, typename CastT>
void TestModified()
{
    printf("\nTesting cache-modified iterator on type %s\n", typeid(T).name()); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    constexpr int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        RandomBits(h_data[i]);
    }

    // Allocate device arrays
    T *d_data = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = h_data[0];          // Value at offset 0
    h_reference[1] = h_data[100];        // Value at offset 100
    h_reference[2] = h_data[1000];       // Value at offset 1000
    h_reference[3] = h_data[10000];      // Value at offset 10000
    h_reference[4] = h_data[1];          // Value at offset 1
    h_reference[5] = h_data[21];         // Value at offset 21
    h_reference[6] = h_data[11];         // Value at offset 11
    h_reference[7] = h_data[0];          // Value at offset 0;

    Test(CacheModifiedInputIterator<LOAD_DEFAULT, T>((CastT*) d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CA, T>((CastT*) d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CG, T>((CastT*) d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CS, T>((CastT*) d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_CV, T>((CastT*) d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_LDG, T>((CastT*) d_data), h_reference);
    Test(CacheModifiedInputIterator<LOAD_VOLATILE, T>((CastT*) d_data), h_reference);

    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}


/**
 * Test transform iterator
 */
template <typename T, typename CastT>
void TestTransform()
{
    printf("\nTesting transform iterator on type %s\n", typeid(T).name()); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    constexpr int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        InitValue(INTEGER_SEED, h_data[i], i);
    }

    // Allocate device arrays
    T *d_data = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    TransformOp<T> op;

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = op(h_data[0]);          // Value at offset 0
    h_reference[1] = op(h_data[100]);        // Value at offset 100
    h_reference[2] = op(h_data[1000]);       // Value at offset 1000
    h_reference[3] = op(h_data[10000]);      // Value at offset 10000
    h_reference[4] = op(h_data[1]);          // Value at offset 1
    h_reference[5] = op(h_data[21]);         // Value at offset 21
    h_reference[6] = op(h_data[11]);         // Value at offset 11
    h_reference[7] = op(h_data[0]);          // Value at offset 0;

    TransformInputIterator<T, TransformOp<T>, CastT*> d_itr((CastT*) d_data, op);
    Test(d_itr, h_reference);

    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}


/**
 * Test tex-obj texture iterator
 */
template <typename T, typename CastT>
void TestTexObj()
{
    printf("\nTesting tex-obj iterator on type %s\n", typeid(T).name()); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    constexpr unsigned int TEST_VALUES          = 11000;
    constexpr unsigned int DUMMY_OFFSET         = 500;
    constexpr unsigned int DUMMY_TEST_VALUES    = TEST_VALUES - DUMMY_OFFSET;

    T *h_data = new T[TEST_VALUES];
    for (unsigned int i = 0; i < TEST_VALUES; ++i)
    {
        RandomBits(h_data[i]);
    }

    // Allocate device arrays
    T *d_data   = NULL;
    T *d_dummy  = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dummy, sizeof(T) * DUMMY_TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_dummy, h_data + DUMMY_OFFSET, sizeof(T) * DUMMY_TEST_VALUES, cudaMemcpyHostToDevice));

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = h_data[0];          // Value at offset 0
    h_reference[1] = h_data[100];        // Value at offset 100
    h_reference[2] = h_data[1000];       // Value at offset 1000
    h_reference[3] = h_data[10000];      // Value at offset 10000
    h_reference[4] = h_data[1];          // Value at offset 1
    h_reference[5] = h_data[21];         // Value at offset 21
    h_reference[6] = h_data[11];         // Value at offset 11
    h_reference[7] = h_data[0];          // Value at offset 0;

    // Create and bind obj-based test iterator
    TexObjInputIterator<T> d_obj_itr;
    CubDebugExit(d_obj_itr.BindTexture((CastT*) d_data, sizeof(T) * TEST_VALUES));

    Test(d_obj_itr, h_reference);

    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
    if (d_dummy) CubDebugExit(g_allocator.DeviceFree(d_dummy));
}

/**
 * Test texture transform iterator
 */
template <typename T, typename CastT>
void TestTexTransform()
{
    printf("\nTesting tex-transform iterator on type %s\n", typeid(T).name()); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    constexpr int TEST_VALUES = 11000;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
    {
        InitValue(INTEGER_SEED, h_data[i], i);
    }

    // Allocate device arrays
    T *d_data = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_data, sizeof(T) * TEST_VALUES));
    CubDebugExit(cudaMemcpy(d_data, h_data, sizeof(T) * TEST_VALUES, cudaMemcpyHostToDevice));

    TransformOp<T> op;

    // Initialize reference data
    T h_reference[8];
    h_reference[0] = op(h_data[0]);          // Value at offset 0
    h_reference[1] = op(h_data[100]);        // Value at offset 100
    h_reference[2] = op(h_data[1000]);       // Value at offset 1000
    h_reference[3] = op(h_data[10000]);      // Value at offset 10000
    h_reference[4] = op(h_data[1]);          // Value at offset 1
    h_reference[5] = op(h_data[21]);         // Value at offset 21
    h_reference[6] = op(h_data[11]);         // Value at offset 11
    h_reference[7] = op(h_data[0]);          // Value at offset 0;

    // Create and bind texture iterator
    typedef TexObjInputIterator<T> TextureIterator;

    TextureIterator d_tex_itr;
    CubDebugExit(d_tex_itr.BindTexture((CastT*) d_data, sizeof(T) * TEST_VALUES));

    // Create transform iterator
    TransformInputIterator<T, TransformOp<T>, TextureIterator> xform_itr(d_tex_itr, op);

    Test(xform_itr, h_reference);

    CubDebugExit(d_tex_itr.UnbindTexture());
    if (h_data) delete[] h_data;
    if (d_data) CubDebugExit(g_allocator.DeviceFree(d_data));
}

/**
 * Run non-integer tests
 */
template <typename T, typename CastT>
void Test(Int2Type<false> /* is_integer */)
{
    TestModified<T, CastT>();
    TestTransform<T, CastT>();
    TestTexObj<T, CastT>();
    TestTexTransform<T, CastT>();
}

/**
 * Run integer tests
 */
template <typename T, typename CastT>
void Test(Int2Type<true> /* is_integer */)
{
    TestConstant<T>(0);
    TestConstant<T>(99);

    TestCounting<T>(0);
    TestCounting<T>(99);

    // Run non-integer tests
    Test<T, CastT>(Int2Type<false>());
}

/**
 * Run tests
 */
template <typename T>
void Test()
{
    enum {
        IS_INTEGER = (Traits<T>::CATEGORY == SIGNED_INTEGER) || (Traits<T>::CATEGORY == UNSIGNED_INTEGER)
    };

    // Test non-const type
    Test<T, T>(Int2Type<IS_INTEGER>());

    // Test non-const type
    Test<T, const T>(Int2Type<IS_INTEGER>());
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Get ptx version
    int ptx_version = 0;
    CubDebugExit(PtxVersion(ptx_version));

    // Evaluate different data types
    Test<signed char>();
    Test<short>();
    Test<int>();
    Test<long>();
    Test<long long>();
    Test<float>();
    Test<double>();

    Test<char2>();
    Test<short2>();
    Test<int2>();
    Test<long2>();
    Test<longlong2>();
    Test<float2>();
    Test<double2>();

    Test<char3>();
    Test<short3>();
    Test<int3>();
    Test<long3>();
    Test<longlong3>();
    Test<float3>();
    Test<double3>();

    Test<char4>();
    Test<short4>();
    Test<int4>();
    Test<long4>();
    Test<longlong4>();
    Test<float4>();
    Test<double4>();

    Test<TestFoo>();
    Test<TestBar>();

    printf("\nTest complete\n"); fflush(stdout);

    return 0;
}



