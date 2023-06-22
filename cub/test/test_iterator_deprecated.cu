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

// This file tests deprecated CUB APIs. Silence deprecation warnings:
#define CUB_IGNORE_DEPRECATED_API

#include <cub/iterator/tex_ref_input_iterator.cuh>
#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>

#include <iterator>
#include <cstdio>
#include <typeinfo>

#include "test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;
CachingDeviceAllocator  g_allocator(true);

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
    if (d_out)
    {
        CubDebugExit(g_allocator.DeviceFree(d_out));
    }
    if (d_itrs)
    {
        CubDebugExit(g_allocator.DeviceFree(d_itrs));
    }
}

/**
 * Test tex-ref texture iterator
 */
template <typename T, typename CastT>
void TestTexRef()
{
    printf("\nTesting tex-ref iterator on type %s\n", typeid(T).name()); fflush(stdout);

    //
    // Test iterator manipulation in kernel
    //

    constexpr int TEST_VALUES                   = 11000;
    constexpr unsigned int DUMMY_OFFSET         = 500;
    constexpr unsigned int DUMMY_TEST_VALUES    = TEST_VALUES - DUMMY_OFFSET;

    T *h_data = new T[TEST_VALUES];
    for (int i = 0; i < TEST_VALUES; ++i)
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

    // Create and bind ref-based test iterator
    TexRefInputIterator<T, __LINE__> d_ref_itr;
    CubDebugExit(d_ref_itr.BindTexture((CastT*) d_data, sizeof(T) * TEST_VALUES));

    // Create and bind dummy iterator of same type to check with interferance
    TexRefInputIterator<T, __LINE__> d_ref_itr2;
    CubDebugExit(d_ref_itr2.BindTexture((CastT*) d_dummy, sizeof(T) * DUMMY_TEST_VALUES));

    Test(d_ref_itr, h_reference);

    CubDebugExit(d_ref_itr.UnbindTexture());
    CubDebugExit(d_ref_itr2.UnbindTexture());

    if (h_data)
    {
        delete[] h_data;
    }
    if (d_data)
    {
        CubDebugExit(g_allocator.DeviceFree(d_data));
    }
    if (d_dummy)
    {
        CubDebugExit(g_allocator.DeviceFree(d_dummy));
    }
}

/**
 * Run non-integer tests
 */
template <typename T, typename CastT>
void Test()
{
    TestTexRef<T, CastT>();
}

/**
 * Run tests
 */
template <typename T>
void Test()
{
    // Test non-const type
    Test<T, T>();

    // Test non-const type
    Test<T, const T>();
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

    printf("\nTest complete\n");
    fflush(stdout);

    return 0;
}
