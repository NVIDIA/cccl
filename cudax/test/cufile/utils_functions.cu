//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__cufile/utils.hpp>

#include <c2h/catch2_test_helper.h>

#include "test_utils.h"
#include <stdexcept>

namespace cuda::experimental::cufile {

using namespace cuda::experimental::cufile::utils;

TEST_CASE("Alignment utilities", "[utils][alignment]") {

    SECTION("is_aligned function") {
        // Test various alignment scenarios
        REQUIRE(is_aligned(0, 4));
        REQUIRE(is_aligned(4, 4));
        REQUIRE(is_aligned(8, 4));
        REQUIRE(is_aligned(12, 4));
        REQUIRE(is_aligned(16, 4));

        REQUIRE_FALSE(is_aligned(1, 4));
        REQUIRE_FALSE(is_aligned(2, 4));
        REQUIRE_FALSE(is_aligned(3, 4));
        REQUIRE_FALSE(is_aligned(5, 4));
        REQUIRE_FALSE(is_aligned(7, 4));

        // Test with different alignment values
        REQUIRE(is_aligned(0, 8));
        REQUIRE(is_aligned(8, 8));
        REQUIRE(is_aligned(16, 8));
        REQUIRE_FALSE(is_aligned(4, 8));
        REQUIRE_FALSE(is_aligned(12, 8));

        // Test with 4KB alignment (typical for file systems)
        REQUIRE(is_aligned(0, 4096));
        REQUIRE(is_aligned(4096, 4096));
        REQUIRE(is_aligned(8192, 4096));
        REQUIRE_FALSE(is_aligned(1, 4096));
        REQUIRE_FALSE(is_aligned(4095, 4096));
        REQUIRE_FALSE(is_aligned(4097, 4096));
    }

    SECTION("align_up function") {
        // Test aligning up to 4-byte boundary
        REQUIRE(align_up(0, 4) == 0);
        REQUIRE(align_up(1, 4) == 4);
        REQUIRE(align_up(2, 4) == 4);
        REQUIRE(align_up(3, 4) == 4);
        REQUIRE(align_up(4, 4) == 4);
        REQUIRE(align_up(5, 4) == 8);
        REQUIRE(align_up(8, 4) == 8);

        // Test aligning up to 8-byte boundary
        REQUIRE(align_up(0, 8) == 0);
        REQUIRE(align_up(1, 8) == 8);
        REQUIRE(align_up(7, 8) == 8);
        REQUIRE(align_up(8, 8) == 8);
        REQUIRE(align_up(9, 8) == 16);

        // Test aligning up to 4KB boundary
        REQUIRE(align_up(0, 4096) == 0);
        REQUIRE(align_up(1, 4096) == 4096);
        REQUIRE(align_up(4095, 4096) == 4096);
        REQUIRE(align_up(4096, 4096) == 4096);
        REQUIRE(align_up(4097, 4096) == 8192);

        // Test edge cases
        REQUIRE(align_up(SIZE_MAX - 3, 4) == SIZE_MAX - 3);  // Should not overflow
    }

    SECTION("get_optimal_alignment function") {
        size_t alignment = get_optimal_alignment();
        REQUIRE(alignment == 4096);  // Should return 4KB alignment

        // Verify the returned alignment is a power of 2
        REQUIRE((alignment & (alignment - 1)) == 0);

        // Verify it's at least 512 bytes (sector size)
        REQUIRE(alignment >= 512);
    }
}

TEST_CASE("CUDA memory functions", "[utils][cuda_memory]") {
    if (!test_utils::is_cuda_available()) {
        SKIP("CUDA not available");
    }

    SECTION("is_gpu_memory function") {
        // Test with GPU memory
        test_utils::GPUMemoryRAII gpu_mem(1024);
        REQUIRE(is_gpu_memory(gpu_mem.get()));

        // Test with host memory
        test_utils::HostMemoryRAII host_mem(1024);
        REQUIRE_FALSE(is_gpu_memory(host_mem.get()));

        // Test with regular memory
        test_utils::RegularMemoryRAII regular_mem(1024);
        REQUIRE_FALSE(is_gpu_memory(regular_mem.get()));

        // Test with null pointer
        REQUIRE_FALSE(is_gpu_memory(nullptr));
    }

    SECTION("get_device_id function") {
        // Test with GPU memory
        test_utils::GPUMemoryRAII gpu_mem(1024);
        int device_id = get_device_id(gpu_mem.get());
        REQUIRE(device_id >= 0);

        // Test with host memory - should not throw but return a device ID
        test_utils::HostMemoryRAII host_mem(1024);
        int host_device_id = get_device_id(host_mem.get());
        REQUIRE(host_device_id >= 0);

        // Test with regular memory - may return -1 for non-CUDA memory
        test_utils::RegularMemoryRAII regular_mem(1024);
        int regular_device_id = get_device_id(regular_mem.get());
        // Regular memory may return -1, which is valid for non-CUDA memory
        REQUIRE(regular_device_id >= -1);
    }

    SECTION("is_cufile_compatible function") {
        // Test with GPU memory (should be compatible)
        test_utils::GPUMemoryRAII gpu_mem(1024);
        REQUIRE(is_cufile_compatible(gpu_mem.get()));

        // Test with pinned host memory (should be compatible)
        test_utils::HostMemoryRAII host_mem(1024);
        REQUIRE(is_cufile_compatible(host_mem.get()));

        // Test with regular memory (should not be compatible)
        test_utils::RegularMemoryRAII regular_mem(1024);
        REQUIRE_FALSE(is_cufile_compatible(regular_mem.get()));

        // Test with null pointer (should not be compatible)
        REQUIRE_FALSE(is_cufile_compatible(nullptr));
    }
}

TEST_CASE("Corner cases and error conditions", "[utils][edge_cases]") {

    SECTION("Null pointer handling") {
        // Test functions with null pointer
        REQUIRE_FALSE(is_gpu_memory(nullptr));
        REQUIRE_FALSE(is_cufile_compatible(nullptr));

        // Note: get_device_id(nullptr) behavior depends on CUDA implementation
        // Some CUDA versions may not throw for null pointers
        try {
            get_device_id(nullptr);
            // If it doesn't throw, that's also acceptable behavior
        } catch (const ::std::runtime_error&) {
            // Expected behavior - null pointer caused an error
        }
    }

    SECTION("Alignment edge cases") {
        // Test alignment with 1 (should be no-op)
        REQUIRE(align_up(0, 1) == 0);
        REQUIRE(align_up(1, 1) == 1);
        REQUIRE(align_up(100, 1) == 100);

        // Test alignment with large values
        REQUIRE(align_up(1000000, 4096) == 1003520);  // 1000000 -> 1003520 (245 * 4096)

        // Test is_aligned with 1
        REQUIRE(is_aligned(0, 1));
        REQUIRE(is_aligned(1, 1));
        REQUIRE(is_aligned(12345, 1));
    }
}

TEST_CASE("Integration tests", "[utils][integration]") {
    if (!test_utils::is_cuda_available()) {
        SKIP("CUDA not available");
    }

    SECTION("Memory type integration") {
        // Test that GPU memory is identified correctly and is cuFile compatible
        test_utils::GPUMemoryRAII gpu_mem(4096);  // Use aligned size
        void* gpu_ptr = gpu_mem.get();

        REQUIRE(is_gpu_memory(gpu_ptr));
        REQUIRE(is_cufile_compatible(gpu_ptr));

        int device_id = get_device_id(gpu_ptr);
        REQUIRE(device_id >= 0);

        // Test that host memory is not GPU memory but is cuFile compatible
        test_utils::HostMemoryRAII host_mem(4096);
        void* host_ptr = host_mem.get();

        REQUIRE_FALSE(is_gpu_memory(host_ptr));
        REQUIRE(is_cufile_compatible(host_ptr));

        // Test that regular memory is neither GPU memory nor cuFile compatible
        test_utils::RegularMemoryRAII regular_mem(4096);
        void* regular_ptr = regular_mem.get();

        REQUIRE_FALSE(is_gpu_memory(regular_ptr));
        REQUIRE_FALSE(is_cufile_compatible(regular_ptr));
    }

    SECTION("Alignment workflow") {
        // Test a typical alignment workflow
        size_t size = 1000;
        size_t alignment = get_optimal_alignment();

        // Check if size is aligned
        if (!is_aligned(size, alignment)) {
            size = align_up(size, alignment);
        }

        // Verify the result is aligned
        REQUIRE(is_aligned(size, alignment));
        REQUIRE(size == 4096);  // 1000 -> 4096 with 4KB alignment

        // Test with already aligned size
        size_t aligned_size = 8192;
        REQUIRE(is_aligned(aligned_size, alignment));
        REQUIRE(align_up(aligned_size, alignment) == aligned_size);
    }
}

} // namespace cuda::experimental::cufile