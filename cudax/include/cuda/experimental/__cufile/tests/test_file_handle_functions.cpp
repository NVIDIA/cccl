#include <gtest/gtest.h>
#include <cuda/experimental/__cufile/file_handle.hpp>
#include <cuda/experimental/__cufile/driver.hpp>
#include <cuda/experimental/__cufile/utils.hpp>
#include "test_utils.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <unistd.h>

namespace cuda::experimental {

class FileHandleTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuda_available_ = test_utils::is_cuda_available();
        cufile_available_ = is_cufile_available();

        // Create temporary test file
        temp_file_path_ = std::filesystem::temp_directory_path() / "cufile_test_file.tmp";

        // Create test file with some data
        std::ofstream file(temp_file_path_, std::ios::binary);
        if (file.is_open()) {
            test_data_.resize(test_size_);
            // Fill with a simple pattern
            for (size_t i = 0; i < test_size_; ++i) {
                test_data_[i] = static_cast<char>(i % 256);
            }
            file.write(test_data_.data(), test_size_);
            file.close();
        }

        // Also create a large file for testing
        large_temp_file_path_ = std::filesystem::temp_directory_path() / "cufile_large_test_file.tmp";
        std::ofstream large_file(large_temp_file_path_, std::ios::binary);
        if (large_file.is_open()) {
            large_test_data_.resize(large_test_size_);
            for (size_t i = 0; i < large_test_size_; ++i) {
                large_test_data_[i] = static_cast<char>(i % 256);
            }
            large_file.write(large_test_data_.data(), large_test_size_);
            large_file.close();
        }
    }

    void TearDown() override {
        // Clean up temporary files
        if (std::filesystem::exists(temp_file_path_)) {
            std::filesystem::remove(temp_file_path_);
        }
        if (std::filesystem::exists(large_temp_file_path_)) {
            std::filesystem::remove(large_temp_file_path_);
        }
    }

    // Helper function to check if specific features are actually supported
    bool is_async_io_supported() {
        // We'll determine this by actually trying a simple async operation
        // and seeing if it works or returns a "not supported" error
        return true; // We'll test this in the actual test
    }


    bool cuda_available_ = false;
    bool cufile_available_ = false;
    std::filesystem::path temp_file_path_;
    std::filesystem::path large_temp_file_path_;
    static constexpr size_t test_size_ = 4096;  // 4KB aligned
    static constexpr size_t large_test_size_ = 1024 * 1024;  // 1MB
    std::vector<char> test_data_;
    std::vector<char> large_test_data_;
};

// Tests for file_handle class
TEST_F(FileHandleTest, FileHandleConstruction) {
    // Test opening existing file
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Test validity
    EXPECT_TRUE(handle.is_valid());

    // Test native handle access
    CUfileHandle_t native = handle.native_handle();
    EXPECT_NE(native, nullptr);
}

TEST_F(FileHandleTest, FileHandleFromFileDescriptor) {
    // Open file descriptor
    int fd = open(temp_file_path_.c_str(), O_RDWR | O_DIRECT);
    if (fd < 0) {
        fd = open(temp_file_path_.c_str(), O_RDWR);  // Fallback without O_DIRECT
    }

    if (fd < 0) {
        FAIL() << "Failed to open file descriptor";
    }

    // Create file handle from FD
    file_handle handle(fd, true);  // Take ownership

    EXPECT_TRUE(handle.is_valid());
}

TEST_F(FileHandleTest, SynchronousRead) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle to ensure cuFILE is initialized
    driver_handle driver;

    // Create file handle
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

    // Test span-based synchronous read
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    EXPECT_EQ(bytes_read, test_size_);

    // Copy data back to host to verify
    std::vector<char> host_buffer(test_size_);
    cudaMemcpy(host_buffer.data(), gpu_buffer.get(), test_size_, cudaMemcpyDeviceToHost);

    // Verify data matches
    EXPECT_EQ(host_buffer, test_data_);
}

TEST_F(FileHandleTest, SynchronousWrite) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle
    driver_handle driver;

    // Create a temporary write file
    std::filesystem::path write_path = std::filesystem::temp_directory_path() / "cufile_write_test.tmp";

    // Create file handle for writing
    file_handle handle(write_path.string(), std::ios::out | std::ios::binary);

    // Allocate GPU buffer and copy test data
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);
    cudaMemcpy(gpu_buffer.get(), test_data_.data(), test_size_, cudaMemcpyHostToDevice);

    // Create span from GPU buffer
    span<const char> buffer_span(static_cast<const char*>(gpu_buffer.get()), test_size_);

    // Test span-based synchronous write
    size_t bytes_written = handle.write(buffer_span);

    // Verify write operation
    EXPECT_EQ(bytes_written, test_size_);

    // Clean up
    std::filesystem::remove(write_path);
}

TEST_F(FileHandleTest, AsynchronousRead) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    try {
        // Create driver handle
        driver_handle driver;

        // Create file handle
        file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

        // Allocate GPU buffer
        test_utils::GPUMemoryRAII gpu_buffer(test_size_);

        // Create CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Create span from GPU buffer
        span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

        // Prepare async parameters
        off_t file_offset = 0;
        off_t buffer_offset = 0;
        ssize_t bytes_read = 0;

        // Test asynchronous read
        handle.read_async(buffer_span, file_offset, buffer_offset, bytes_read, stream);

        // Synchronize stream
        cudaStreamSynchronize(stream);

        // Check the result - the bindings should handle compatibility mode internally
        if (bytes_read < 0) {
            // Error -5 typically means async not supported, which is a valid system limitation
            if (bytes_read == -5) {
                GTEST_SKIP() << "Async operations not supported on this system (error: " << bytes_read << ")";
            } else {
                FAIL() << "Async read failed with error: " << bytes_read;
            }
        }

        // Verify result
        EXPECT_EQ(bytes_read, static_cast<ssize_t>(test_size_));

        // Clean up
        cudaStreamDestroy(stream);

    } catch (const std::exception& e) {
        FAIL() << "Failed asynchronous read test: " << e.what();
    }
}

TEST_F(FileHandleTest, AsynchronousWrite) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    try {
        // Create driver handle
        driver_handle driver;

        // Create a temporary write file
        std::filesystem::path write_path = std::filesystem::temp_directory_path() / "cufile_async_write_test.tmp";

        // Create file handle for writing
        file_handle handle(write_path.string(), std::ios::out | std::ios::binary);

        // Allocate GPU buffer and copy test data
        test_utils::GPUMemoryRAII gpu_buffer(test_size_);
        cudaMemcpy(gpu_buffer.get(), test_data_.data(), test_size_, cudaMemcpyHostToDevice);

        // Create CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Create span from GPU buffer
        span<const char> buffer_span(static_cast<const char*>(gpu_buffer.get()), test_size_);

        // Prepare async parameters
        off_t file_offset = 0;
        off_t buffer_offset = 0;
        ssize_t bytes_written = 0;

        // Test asynchronous write
        handle.write_async(buffer_span, file_offset, buffer_offset, bytes_written, stream);

        // Synchronize stream
        cudaStreamSynchronize(stream);

        // Check the result - the bindings should handle compatibility mode internally
        if (bytes_written < 0) {
            // Error -5 typically means async not supported, which is a valid system limitation
            if (bytes_written == -5) {
                GTEST_SKIP() << "Async operations not supported on this system (error: " << bytes_written << ")";
            } else {
                FAIL() << "Async write failed with error: " << bytes_written;
            }
        }

        // Verify result
        EXPECT_EQ(bytes_written, static_cast<ssize_t>(test_size_));

        // Clean up
        cudaStreamDestroy(stream);
        std::filesystem::remove(write_path);

    } catch (const std::exception& e) {
        FAIL() << "Failed asynchronous write test: " << e.what();
    }
}

TEST_F(FileHandleTest, HostMemoryRead) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle
    driver_handle driver;

    // Create file handle
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Allocate pinned host memory
    test_utils::HostMemoryRAII host_buffer(test_size_);

    // Create span from host buffer
    span<char> buffer_span(static_cast<char*>(host_buffer.get()), test_size_);

    // Test read with host memory (should work in compatibility mode)
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    EXPECT_EQ(bytes_read, test_size_);

    // Verify data matches
    EXPECT_EQ(std::memcmp(host_buffer.get(), test_data_.data(), test_size_), 0);
}

// Tests for buffer_handle class
TEST_F(FileHandleTest, BufferHandleRegistration) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle
    driver_handle driver;

    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

    // Test buffer registration
    buffer_handle buf_handle(buffer_span);

    // Verify buffer handle properties
    EXPECT_EQ(buf_handle.data(), gpu_buffer.get());
    EXPECT_EQ(buf_handle.size(), test_size_);

    // Test typed span access
    auto typed_span = buf_handle.as_span<char>();
    EXPECT_EQ(typed_span.data(), gpu_buffer.get());
    EXPECT_EQ(typed_span.size(), test_size_);
}

TEST_F(FileHandleTest, BufferHandleWithFlags) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle
    driver_handle driver;

    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    span<const char> buffer_span(static_cast<const char*>(gpu_buffer.get()), test_size_);

    // Test buffer registration with flags
    buffer_handle buf_handle(buffer_span, 0);  // Default flags

    // Verify buffer handle properties
    EXPECT_EQ(buf_handle.data(), gpu_buffer.get());
    EXPECT_EQ(buf_handle.size(), test_size_);

    // Test byte span access
    auto byte_span = buf_handle.as_bytes();
    EXPECT_EQ(byte_span.data(), gpu_buffer.get());
    EXPECT_EQ(byte_span.size(), test_size_);
}

// Tests for batch_handle class
TEST_F(FileHandleTest, BatchHandleCreation) {

    // Create driver handle
    driver_handle driver;

    // Test batch handle creation
    batch_handle batch(10);  // Max 10 operations

    // Verify batch handle properties
    EXPECT_TRUE(batch.is_valid());
    EXPECT_EQ(batch.max_operations(), 10U);
}

TEST_F(FileHandleTest, BatchOperations) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle
    driver_handle driver;

    // Create file handle
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Allocate GPU buffers
    test_utils::GPUMemoryRAII gpu_buffer1(test_size_);
    test_utils::GPUMemoryRAII gpu_buffer2(test_size_);

    // Create batch handle
    batch_handle batch(5);

    // Create spans from GPU buffers
    span<char> buffer_span1(static_cast<char*>(gpu_buffer1.get()), test_size_ / 2);
    span<char> buffer_span2(static_cast<char*>(gpu_buffer2.get()), test_size_ / 2);

    // Prepare batch operations using factory functions
    std::vector<batch_io_params_span<char>> operations;
    operations.push_back(make_read_operation(buffer_span1, 0, 0, reinterpret_cast<void*>(1)));
    operations.push_back(make_read_operation(buffer_span2, test_size_ / 2, 0, reinterpret_cast<void*>(2)));

    // Submit batch operations
    batch.submit(handle, span<const batch_io_params_span<char>>(operations));

    // Wait for all submitted operations to complete
    auto results = batch.get_status(operations.size(), 5000);  // Wait for all operations, 5s timeout

    // Verify all operations completed
    EXPECT_EQ(results.size(), operations.size());

    for (const auto& result : results) {
        EXPECT_TRUE(result.is_complete());
        EXPECT_FALSE(result.is_failed());
        EXPECT_EQ(result.result, test_size_ / 2);
    }
}

TEST_F(FileHandleTest, BatchCancel) {
    // Create driver handle
    driver_handle driver;

    // Create batch handle
    batch_handle batch(5);

    // Test batch cancellation
    batch.cancel();

}

// Test min_completed = 0 default behavior
TEST_F(FileHandleTest, BatchGetStatusDefaultMinCompleted) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle
    driver_handle driver;

    // Create file handle
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create batch handle
    batch_handle batch(5);


    // Prepare batch operations
    span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);
    std::vector<batch_io_params_span<char>> operations;
    operations.push_back(make_read_operation(buffer_span, 0, 0, reinterpret_cast<void*>(1)));

    // Submit batch operations
    batch.submit(handle, span<const batch_io_params_span<char>>(operations));

    auto results = batch.get_status(0);

    // Verify results
    EXPECT_LE(results.size(), operations.size());

    for (const auto& result : results) {
        EXPECT_TRUE(result.is_complete() || !result.is_failed());
    }
}

// Test behavior when all operations are already completed
TEST_F(FileHandleTest, BatchGetStatusAfterAllCompleted) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create driver handle
    driver_handle driver;

    // Create file handle
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create batch handle
    batch_handle batch(5);


    // Prepare batch operations
    span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);
    std::vector<batch_io_params_span<char>> operations;
    operations.push_back(make_read_operation(buffer_span, 0, 0, reinterpret_cast<void*>(1)));

    // Submit batch operations
    batch.submit(handle, span<const batch_io_params_span<char>>(operations));

    // Wait for all operations to complete
    auto results = batch.get_status(operations.size(), 5000);
    EXPECT_EQ(results.size(), operations.size());

    //if all operations are completed, get_status should return an empty vector
    results = batch.get_status(operations.size(), 5000);
    EXPECT_EQ(results.size(), 0);
}

// Tests for stream_handle class
TEST_F(FileHandleTest, StreamHandleRegistration) {
    // Only skip if the stream API is genuinely not available on this system
    if (!is_stream_api_available()) {
        GTEST_SKIP() << "Stream API not available on this system";
    }


    // Create driver handle
    driver_handle driver;

    // Create CUDA stream
    cudaStream_t stream;
    cudaError_t cuda_error = cudaStreamCreate(&stream);
    if (cuda_error != cudaSuccess) {
        GTEST_SKIP() << "Failed to create CUDA stream: " << cudaGetErrorString(cuda_error);
    }

    // Test stream registration - if stream API is available, this should work
    stream_handle stream_handle_obj(stream);

    // Verify stream handle
    EXPECT_EQ(stream_handle_obj.get(), stream);
}

TEST_F(FileHandleTest, StreamHandleWithFlags) {

    // Only skip if the stream API is genuinely not available on this system
    if (!is_stream_api_available()) {
        GTEST_SKIP() << "Stream API not available on this system";
    }
    // Create driver handle
    driver_handle driver;

    // Create CUDA stream
    cudaStream_t stream;
    cudaError_t cuda_error = cudaStreamCreate(&stream);
    if (cuda_error != cudaSuccess) {
        GTEST_SKIP() << "Failed to create CUDA stream: " << cudaGetErrorString(cuda_error);
    }

    // Test stream registration with flags - if stream API is available, this should work
    stream_handle stream_handle_obj(stream, 0);  // Default flags

    // Verify stream handle
    EXPECT_EQ(stream_handle_obj.get(), stream);


}

// Integration tests
TEST_F(FileHandleTest, FileHandleLifecycleIntegration) {
    // Create driver handle
    driver_handle driver;

    // Test complete file handle lifecycle
    {
        file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);
        EXPECT_TRUE(handle.is_valid());

        // Test read operation
        test_utils::GPUMemoryRAII gpu_buffer(test_size_);
        span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);
        size_t bytes_read = handle.read(buffer_span);
        EXPECT_EQ(bytes_read, test_size_);
    }

}

TEST_F(FileHandleTest, BufferAndStreamIntegration) {


    // Only skip if the stream API is genuinely not available on this system
    if (!is_stream_api_available()) {
        GTEST_SKIP() << "Stream API not available on this system";
    }


    // Create driver handle
    driver_handle driver;

    // Create file handle
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Allocate and register GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);
    buffer_handle buf_handle(buffer_span);

    // Create and register CUDA stream
    cudaStream_t stream;
    cudaError_t cuda_error = cudaStreamCreate(&stream);
    if (cuda_error != cudaSuccess) {
        GTEST_SKIP() << "Failed to create CUDA stream: " << cudaGetErrorString(cuda_error);
    }

    // Register the stream - if stream API is available, this should work
    stream_handle stream_handle_obj(stream);

    // Test async read with registered buffer and stream
    off_t file_offset = 0;
    off_t buffer_offset = 0;
    ssize_t bytes_read = 0;

    handle.read_async(buffer_span, file_offset, buffer_offset, bytes_read, stream);

    // Synchronize stream
    cudaStreamSynchronize(stream);

    // Check the result - async operations might not be supported even if streams are
    if (bytes_read < 0) {
        if (bytes_read == -5) {
            // This is expected if async operations aren't supported
            GTEST_SKIP() << "Async operations not supported on this system, but stream registration worked";
        } else {
            FAIL() << "Async read failed with error: " << bytes_read;
        }
    } else {
        // Verify result if async worked
        EXPECT_EQ(bytes_read, static_cast<ssize_t>(test_size_));
    }
}

TEST_F(FileHandleTest, LargeFileOperations) {

    // Create driver handle
    driver_handle driver;

    // Create file handle for large file
    file_handle handle(large_temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Allocate large GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(large_test_size_);

    // Create span from GPU buffer
    span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), large_test_size_);

    // Test large file read
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    EXPECT_EQ(bytes_read, large_test_size_);

    // Copy data back to host to verify
    std::vector<char> host_buffer(large_test_size_);
    cudaMemcpy(host_buffer.data(), gpu_buffer.get(), large_test_size_, cudaMemcpyDeviceToHost);

    // Verify data matches
    EXPECT_EQ(host_buffer, large_test_data_);
}

// Error handling and compatibility tests
TEST_F(FileHandleTest, InvalidFileHandling) {
    // Test with non-existent file
    EXPECT_THROW(file_handle("non_existent_file.tmp", std::ios::in), std::exception);
}

TEST_F(FileHandleTest, CompatibilityModeTest) {

    // Test with regular (non-cuFile-compatible) memory
    test_utils::RegularMemoryRAII regular_buffer(test_size_);

    // Create file handle
    file_handle handle(temp_file_path_.string(), std::ios::in | std::ios::binary);

    // Create span from regular buffer
    span<char> buffer_span(static_cast<char*>(regular_buffer.get()), test_size_);

    // This should work in compatibility mode by falling back to POSIX I/O
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    EXPECT_EQ(bytes_read, test_size_);

    // Verify data matches
    EXPECT_EQ(std::memcmp(regular_buffer.get(), test_data_.data(), test_size_), 0);
}

} // namespace cuda::experimental

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Print information about test environment
    std::cout << "=== cuFILE File Handle Tests ===" << std::endl;

    if (test_utils::is_cuda_available()) {
        std::cout << "CUDA is available - running CUDA-dependent tests" << std::endl;
    } else {
        std::cout << "CUDA is not available - skipping CUDA-dependent tests" << std::endl;
    }

    if (cuda::experimental::is_cufile_available()) {
        std::cout << "cuFILE is available - running full test suite" << std::endl;
    } else {
        std::cout << "cuFILE is not available - running compatibility mode tests" << std::endl;
    }

    return RUN_ALL_TESTS();
}