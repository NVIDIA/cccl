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

/******************************************************************************
 * Test of DeviceMergeSort utilities using large user types (i.e., with vsmem utilities)
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cstdio>
#include <new> // for std::bad_alloc

#include "c2h/huge_type.cuh"
#include "test_device_merge_sort.cuh"

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  using DataType = int64_t;

  thrust::default_random_engine rng;
  for (unsigned int pow2 = 9; pow2 < 22; pow2 += 2)
  {
    try
    {
      const unsigned int num_items = 1 << pow2;
      // Testing vsmem facility with a fallback policy
      TestHelper<true>::AllocateAndTest<c2h::detail::huge_data_type_t<128>, DataType>(rng, num_items);
      // Testing vsmem facility with virtual shared memory
      TestHelper<true>::AllocateAndTest<c2h::detail::huge_data_type_t<256>, DataType>(rng, num_items);
    }
    catch (std::bad_alloc& e)
    {
      if (pow2 > 20)
      { // Some cards don't have enough memory for large allocations, these
        // can be skipped.
        printf("Skipping large memory test. (num_items=2^%u): %s\n", pow2, e.what());
      }
      else
      { // For smaller problem sizes, treat as an error:
        printf("Error (num_items=2^%u): %s", pow2, e.what());
        throw;
      }
    }
  }

  return 0;
}
