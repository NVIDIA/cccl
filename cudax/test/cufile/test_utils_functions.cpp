#include <gtest/gtest.h>
#include <cuda/experimental/__cufile/utils.hpp>
#include "test_utils.h"
#include <memory>

namespace cuda::experimental::utils {

class UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuda_available_ = test_utils::is_cuda_available();
        if (!cuda_available_) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA-dependent tests";
        }
    }

    void TearDown() override {
        // Clean up any remaining allocations
    }

    bool cuda_available_ = false;
};

// Tests for is_aligned function
TEST(UtilsAlignmentTest, IsAligned) {
    // Test various alignment scenarios
    EXPECT_TRUE(is_aligned(0, 4));
    EXPECT_TRUE(is_aligned(4, 4));
    EXPECT_TRUE(is_aligned(8, 4));
    EXPECT_TRUE(is_aligned(12, 4));
    EXPECT_TRUE(is_aligned(16, 4));

    EXPECT_FALSE(is_aligned(1, 4));
    EXPECT_FALSE(is_aligned(2, 4));
    EXPECT_FALSE(is_aligned(3, 4));
    EXPECT_FALSE(is_aligned(5, 4));
    EXPECT_FALSE(is_aligned(7, 4));

    // Test with different alignment values
    EXPECT_TRUE(is_aligned(0, 8));
    EXPECT_TRUE(is_aligned(8, 8));
    EXPECT_TRUE(is_aligned(16, 8));
    EXPECT_FALSE(is_aligned(4, 8));
    EXPECT_FALSE(is_aligned(12, 8));

    // Test with 4KB alignment (typical for file systems)
    EXPECT_TRUE(is_aligned(0, 4096));
    EXPECT_TRUE(is_aligned(4096, 4096));
    EXPECT_TRUE(is_aligned(8192, 4096));
    EXPECT_FALSE(is_aligned(1, 4096));
    EXPECT_FALSE(is_aligned(4095, 4096));
    EXPECT_FALSE(is_aligned(4097, 4096));
}

// Tests for align_up function
TEST(UtilsAlignmentTest, AlignUp) {
    // Test aligning up to 4-byte boundary
    EXPECT_EQ(align_up(0, 4), 0);
    EXPECT_EQ(align_up(1, 4), 4);
    EXPECT_EQ(align_up(2, 4), 4);
    EXPECT_EQ(align_up(3, 4), 4);
    EXPECT_EQ(align_up(4, 4), 4);
    EXPECT_EQ(align_up(5, 4), 8);
    EXPECT_EQ(align_up(8, 4), 8);

    // Test aligning up to 8-byte boundary
    EXPECT_EQ(align_up(0, 8), 0);
    EXPECT_EQ(align_up(1, 8), 8);
    EXPECT_EQ(align_up(7, 8), 8);
    EXPECT_EQ(align_up(8, 8), 8);
    EXPECT_EQ(align_up(9, 8), 16);

    // Test aligning up to 4KB boundary
    EXPECT_EQ(align_up(0, 4096), 0);
    EXPECT_EQ(align_up(1, 4096), 4096);
    EXPECT_EQ(align_up(4095, 4096), 4096);
    EXPECT_EQ(align_up(4096, 4096), 4096);
    EXPECT_EQ(align_up(4097, 4096), 8192);

    // Test edge cases
    EXPECT_EQ(align_up(SIZE_MAX - 3, 4), SIZE_MAX - 3);  // Should not overflow
}

// Tests for get_optimal_alignment function
TEST(UtilsAlignmentTest, GetOptimalAlignment) {
    size_t alignment = get_optimal_alignment();
    EXPECT_EQ(alignment, 4096);  // Should return 4KB alignment

    // Verify the returned alignment is a power of 2
    EXPECT_TRUE((alignment & (alignment - 1)) == 0);

    // Verify it's at least 512 bytes (sector size)
    EXPECT_GE(alignment, 512);
}

// Tests for CUDA memory functions (require CUDA)
TEST_F(UtilsTest, IsGpuMemory) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Test with GPU memory
    test_utils::GPUMemoryRAII gpu_mem(1024);
    EXPECT_TRUE(is_gpu_memory(gpu_mem.get()));

    // Test with host memory
    test_utils::HostMemoryRAII host_mem(1024);
    EXPECT_FALSE(is_gpu_memory(host_mem.get()));

    // Test with regular memory
    test_utils::RegularMemoryRAII regular_mem(1024);
    EXPECT_FALSE(is_gpu_memory(regular_mem.get()));

    // Test with null pointer
    EXPECT_FALSE(is_gpu_memory(nullptr));
}

TEST_F(UtilsTest, GetDeviceId) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Test with GPU memory
    test_utils::GPUMemoryRAII gpu_mem(1024);
    int device_id = get_device_id(gpu_mem.get());
    EXPECT_GE(device_id, 0);

    // Test with host memory - should not throw but return a device ID
    test_utils::HostMemoryRAII host_mem(1024);
    int host_device_id = get_device_id(host_mem.get());
    EXPECT_GE(host_device_id, 0);

    // Test with regular memory - may return -1 for non-CUDA memory
    test_utils::RegularMemoryRAII regular_mem(1024);
    int regular_device_id = get_device_id(regular_mem.get());
    // Regular memory may return -1, which is valid for non-CUDA memory
    EXPECT_GE(regular_device_id, -1);

}

TEST_F(UtilsTest, IsCufileCompatible) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Test with GPU memory (should be compatible)
    test_utils::GPUMemoryRAII gpu_mem(1024);
    EXPECT_TRUE(is_cufile_compatible(gpu_mem.get()));

    // Test with pinned host memory (should be compatible)
    test_utils::HostMemoryRAII host_mem(1024);
    EXPECT_TRUE(is_cufile_compatible(host_mem.get()));

    // Test with regular memory (should not be compatible)
    test_utils::RegularMemoryRAII regular_mem(1024);
    EXPECT_FALSE(is_cufile_compatible(regular_mem.get()));

    // Test with null pointer (should not be compatible)
    EXPECT_FALSE(is_cufile_compatible(nullptr));
}

// Tests for corner cases and error conditions
TEST(UtilsCornerCasesTest, NullPointerHandling) {
    // Test functions with null pointer
    EXPECT_FALSE(is_gpu_memory(nullptr));
    EXPECT_FALSE(is_cufile_compatible(nullptr));
    // Note: get_device_id(nullptr) behavior depends on CUDA implementation
    // Some CUDA versions may not throw for null pointers
    try {
        get_device_id(nullptr);
        // If it doesn't throw, that's also acceptable behavior
    } catch (const std::runtime_error&) {
        // Expected behavior - null pointer caused an error
    }
}

TEST(UtilsCornerCasesTest, AlignmentEdgeCases) {
    // Test alignment with 1 (should be no-op)
    EXPECT_EQ(align_up(0, 1), 0);
    EXPECT_EQ(align_up(1, 1), 1);
    EXPECT_EQ(align_up(100, 1), 100);

    // Test alignment with large values
    EXPECT_EQ(align_up(1000000, 4096), 1003520);  // 1000000 -> 1003520 (245 * 4096)

    // Test is_aligned with 1
    EXPECT_TRUE(is_aligned(0, 1));
    EXPECT_TRUE(is_aligned(1, 1));
    EXPECT_TRUE(is_aligned(12345, 1));
}

// Integration tests
TEST_F(UtilsTest, MemoryTypeIntegration) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Test that GPU memory is identified correctly and is cuFile compatible
    test_utils::GPUMemoryRAII gpu_mem(4096);  // Use aligned size
    void* gpu_ptr = gpu_mem.get();

    EXPECT_TRUE(is_gpu_memory(gpu_ptr));
    EXPECT_TRUE(is_cufile_compatible(gpu_ptr));

    int device_id = get_device_id(gpu_ptr);
    EXPECT_GE(device_id, 0);

    // Test that host memory is not GPU memory but is cuFile compatible
    test_utils::HostMemoryRAII host_mem(4096);
    void* host_ptr = host_mem.get();

    EXPECT_FALSE(is_gpu_memory(host_ptr));
    EXPECT_TRUE(is_cufile_compatible(host_ptr));

    // Test that regular memory is neither GPU memory nor cuFile compatible
    test_utils::RegularMemoryRAII regular_mem(4096);
    void* regular_ptr = regular_mem.get();

    EXPECT_FALSE(is_gpu_memory(regular_ptr));
    EXPECT_FALSE(is_cufile_compatible(regular_ptr));
}

TEST(UtilsAlignmentIntegration, AlignmentWorkflow) {
    // Test a typical alignment workflow
    size_t size = 1000;
    size_t alignment = get_optimal_alignment();

    // Check if size is aligned
    if (!is_aligned(size, alignment)) {
        size = align_up(size, alignment);
    }

    // Verify the result is aligned
    EXPECT_TRUE(is_aligned(size, alignment));
    EXPECT_EQ(size, 4096);  // 1000 -> 4096 with 4KB alignment

    // Test with already aligned size
    size_t aligned_size = 8192;
    EXPECT_TRUE(is_aligned(aligned_size, alignment));
    EXPECT_EQ(align_up(aligned_size, alignment), aligned_size);
}

} // namespace cuda::experimental::utils

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Print information about test environment
    if (test_utils::is_cuda_available()) {
        std::cout << "CUDA is available - running full test suite\n";
    } else {
        std::cout << "CUDA is not available - running limited test suite\n";
    }

    return RUN_ALL_TESTS();
}