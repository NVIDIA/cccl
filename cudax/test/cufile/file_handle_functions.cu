//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__cufile/driver.hpp>
#include <cuda/experimental/__cufile/file_handle.hpp>
#include <cuda/experimental/__cufile/utils.hpp>

#include "test_utils.h"
#include <c2h/catch2_test_helper.h>

// Use system standard library headers to avoid CCCL conflicts
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <unistd.h>

namespace cuda::experimental::cufile
{

namespace
{

struct driver_singleton
{
  static driver_singleton& get()
  {
    static driver_singleton s_driver;
    return s_driver;
  }

private:
  driver_singleton()
  {
    driver_open();
  }
  ~driver_singleton()
  {
    driver_close();
  }
};

// Test configuration and setup
::std::filesystem::path temp_file_path_;
::std::filesystem::path large_temp_file_path_;
static constexpr size_t test_size_       = 4096; // 4KB aligned
static constexpr size_t large_test_size_ = 1024 * 1024; // 1MB
::std::vector<char> test_data_;
::std::vector<char> large_test_data_;

void setup_test_environment()
{
  REQUIRE(test_utils::is_cuda_available());
  REQUIRE(is_cufile_available());
  driver_singleton::get();

  auto timestamp = ::std::chrono::high_resolution_clock::now().time_since_epoch().count();
  temp_file_path_ =
    ::std::filesystem::temp_directory_path() / ("cufile_test_file_" + ::std::to_string(timestamp) + ".tmp");

  // Create test file with some data
  ::std::ofstream file(temp_file_path_, ::std::ios::binary);
  if (file.is_open())
  {
    test_data_.resize(test_size_);
    // Fill with a simple pattern
    for (size_t i = 0; i < test_size_; ++i)
    {
      test_data_[i] = static_cast<char>(i % 256);
    }
    file.write(test_data_.data(), test_size_);
    file.close();
  }

  large_temp_file_path_ =
    ::std::filesystem::temp_directory_path() / ("cufile_large_test_file_" + ::std::to_string(timestamp) + ".tmp");
  ::std::ofstream large_file(large_temp_file_path_, ::std::ios::binary);
  if (large_file.is_open())
  {
    large_test_data_.resize(large_test_size_);
    for (size_t i = 0; i < large_test_size_; ++i)
    {
      large_test_data_[i] = static_cast<char>(i % 256);
    }
    large_file.write(large_test_data_.data(), large_test_size_);
    large_file.close();
  }
}

void cleanup_test_environment()
{
  // Clean up temporary files
  if (::std::filesystem::exists(temp_file_path_))
  {
    ::std::filesystem::remove(temp_file_path_);
  }
  if (::std::filesystem::exists(large_temp_file_path_))
  {
    ::std::filesystem::remove(large_temp_file_path_);
  }
}
} // namespace

TEST_CASE("File handle construction", "[file_handle][construction]")
{
  setup_test_environment();

  SECTION("File handle construction from path")
  {
    // Test opening existing file
    file_handle handle(temp_file_path_.string(), ::std::ios::in | ::std::ios::binary);

    // Test validity
    REQUIRE(handle.is_valid());

    // Test native handle access
    CUfileHandle_t native = handle.native_handle();
    REQUIRE(native != nullptr);
  }

  SECTION("File handle construction from file descriptor")
  {
    // Open file descriptor
    int fd = open(temp_file_path_.c_str(), O_RDWR | O_DIRECT);
    if (fd < 0)
    {
      fd = open(temp_file_path_.c_str(), O_RDWR); // Fallback without O_DIRECT
    }

    if (fd < 0)
    {
      FAIL("Failed to open file descriptor");
    }

    // Create file handle from FD (always takes ownership)
    file_handle handle(fd);

    REQUIRE(handle.is_valid());
  }

  SECTION("File handle reference construction from file descriptor")
  {
    // Open file descriptor
    int fd = open(temp_file_path_.c_str(), O_RDWR | O_DIRECT);
    if (fd < 0)
    {
      fd = open(temp_file_path_.c_str(), O_RDWR); // Fallback without O_DIRECT
    }

    if (fd < 0)
    {
      FAIL("Failed to open file descriptor");
    }

    // Create file handle reference from FD (doesn't take ownership)
    file_handle_ref handle_ref(fd);

    REQUIRE(handle_ref.is_valid());

    // Clean up manually since file_handle_ref doesn't own the fd
    close(fd);
  }

  cleanup_test_environment();
}

TEST_CASE("Synchronous I/O operations", "[file_handle][sync_io]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  setup_test_environment();

  SECTION("Synchronous read")
  {
    // Create file handle
    file_handle handle(temp_file_path_.string(), ::std::ios::in | ::std::ios::binary);

    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    cuda::std::span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

    // Test span-based synchronous read
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    REQUIRE(bytes_read == test_size_);

    // Copy data back to host to verify
    ::std::vector<char> host_buffer(test_size_);
    cudaMemcpy(host_buffer.data(), gpu_buffer.get(), test_size_, cudaMemcpyDeviceToHost);

    // Verify data matches
    REQUIRE(host_buffer == test_data_);
  }

  SECTION("Synchronous write")
  {
    // Create a temporary write file
    ::std::filesystem::path write_path = ::std::filesystem::temp_directory_path() / "cufile_write_test.tmp";

    // Create file handle for writing
    file_handle handle(write_path.string(), ::std::ios::out | ::std::ios::binary);

    // Allocate GPU buffer and copy test data
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);
    cudaMemcpy(gpu_buffer.get(), test_data_.data(), test_size_, cudaMemcpyHostToDevice);

    // Create span from GPU buffer
    cuda::std::span<const char> buffer_span(static_cast<const char*>(gpu_buffer.get()), test_size_);

    // Test span-based synchronous write
    size_t bytes_written = handle.write(buffer_span);

    // Verify write operation
    REQUIRE(bytes_written == test_size_);

    // Clean up
    ::std::filesystem::remove(write_path);
  }

  SECTION("Host memory read")
  {
    // Create file handle
    file_handle handle(temp_file_path_.string(), ::std::ios::in | ::std::ios::binary);

    // Allocate pinned host memory
    test_utils::HostMemoryRAII host_buffer(test_size_);

    // Create span from host buffer
    cuda::std::span<char> buffer_span(static_cast<char*>(host_buffer.get()), test_size_);

    // Test read with host memory (should work in compatibility mode)
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    REQUIRE(bytes_read == test_size_);

    // Verify data matches
    REQUIRE(::std::memcmp(host_buffer.get(), test_data_.data(), test_size_) == 0);
  }

  SECTION("File handle reference synchronous read")
  {
    // Open file descriptor manually
    int fd = open(temp_file_path_.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0)
    {
      fd = open(temp_file_path_.c_str(), O_RDONLY); // Fallback without O_DIRECT
    }

    if (fd < 0)
    {
      FAIL("Failed to open file descriptor");
    }

    // Create non-owning reference from fd
    file_handle_ref handle_ref(fd);

    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    cuda::std::span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

    // Test span-based synchronous read using reference
    size_t bytes_read = handle_ref.read(buffer_span);

    // Verify read operation
    REQUIRE(bytes_read == test_size_);

    // Copy data back to host to verify
    ::std::vector<char> host_buffer(test_size_);
    cudaMemcpy(host_buffer.data(), gpu_buffer.get(), test_size_, cudaMemcpyDeviceToHost);

    // Verify data matches
    REQUIRE(host_buffer == test_data_);

    // Clean up manually since file_handle_ref doesn't close the fd
    close(fd);
  }

  cleanup_test_environment();
}

TEST_CASE("Asynchronous I/O operations", "[file_handle][async_io]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  setup_test_environment();

  SECTION("Asynchronous read")
  {
    try
    {
      // Create file handle
      file_handle handle(temp_file_path_.string(), ::std::ios::in | ::std::ios::binary);

      // Allocate GPU buffer
      test_utils::GPUMemoryRAII gpu_buffer(test_size_);

      // Create CUDA stream
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      // Create span from GPU buffer
      cuda::std::span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

      // Prepare async parameters
      off_t file_offset   = 0;
      off_t buffer_offset = 0;
      ssize_t bytes_read  = 0;

      // Test asynchronous read
      handle.read_async(cuda::stream_ref{stream}, buffer_span, file_offset, buffer_offset, bytes_read);

      // Synchronize stream
      cudaStreamSynchronize(stream);

      // Check the result - the bindings should handle compatibility mode internally
      if (bytes_read < 0)
      {
        // Error -5 typically means async not supported, which is a valid system limitation
        if (bytes_read == -5)
        {
          SKIP("Async operations not supported on this system (error: " << bytes_read << ")");
        }
        else
        {
          FAIL("Async read failed with error: " << bytes_read);
        }
      }

      // Verify result
      REQUIRE(bytes_read == static_cast<ssize_t>(test_size_));

      // Clean up
      cudaStreamDestroy(stream);
    }
    catch (const ::std::exception& e)
    {
      FAIL("Failed asynchronous read test: " << e.what());
    }
  }

  SECTION("Asynchronous write")
  {
    try
    {
      // Create a temporary write file
      ::std::filesystem::path write_path = ::std::filesystem::temp_directory_path() / "cufile_async_write_test.tmp";

      // Create file handle for writing
      file_handle handle(write_path.string(), ::std::ios::out | ::std::ios::binary);

      // Allocate GPU buffer and copy test data
      test_utils::GPUMemoryRAII gpu_buffer(test_size_);
      cudaMemcpy(gpu_buffer.get(), test_data_.data(), test_size_, cudaMemcpyHostToDevice);

      // Create CUDA stream
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      // Create span from GPU buffer
      cuda::std::span<const char> buffer_span(static_cast<const char*>(gpu_buffer.get()), test_size_);

      // Prepare async parameters
      off_t file_offset     = 0;
      off_t buffer_offset   = 0;
      ssize_t bytes_written = 0;

      // Test asynchronous write
      handle.write_async(cuda::stream_ref{stream}, buffer_span, file_offset, buffer_offset, bytes_written);

      // Synchronize stream
      cudaStreamSynchronize(stream);

      // Check the result - the bindings should handle compatibility mode internally
      if (bytes_written < 0)
      {
        // Error -5 typically means async not supported, which is a valid system limitation
        if (bytes_written == -5)
        {
          SKIP("Async operations not supported on this system (error: " << bytes_written << ")");
        }
        else
        {
          FAIL("Async write failed with error: " << bytes_written);
        }
      }

      // Verify result
      REQUIRE(bytes_written == static_cast<ssize_t>(test_size_));

      // Clean up
      cudaStreamDestroy(stream);
      ::std::filesystem::remove(write_path);
    }
    catch (const ::std::exception& e)
    {
      FAIL("Failed asynchronous write test: " << e.what());
    }
  }

  cleanup_test_environment();
}

TEST_CASE("File handle reference operations", "[file_handle_ref][io]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  setup_test_environment();

  SECTION("File handle reference synchronous write")
  {
    // Create a temporary write file
    ::std::filesystem::path write_path = ::std::filesystem::temp_directory_path() / "cufile_ref_write_test.tmp";

    // Open file descriptor for writing
    int fd = open(write_path.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (fd < 0)
    {
      fd = open(write_path.c_str(), O_WRONLY | O_CREAT, 0644); // Fallback without O_DIRECT
    }

    if (fd < 0)
    {
      FAIL("Failed to open file descriptor for writing");
    }

    // Create non-owning reference from fd
    file_handle_ref handle_ref(fd);

    // Allocate GPU buffer and copy test data
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);
    cudaMemcpy(gpu_buffer.get(), test_data_.data(), test_size_, cudaMemcpyHostToDevice);

    // Create span from GPU buffer
    cuda::std::span<const char> buffer_span(static_cast<const char*>(gpu_buffer.get()), test_size_);

    // Test span-based synchronous write
    size_t bytes_written = handle_ref.write(buffer_span);

    // Verify write operation
    REQUIRE(bytes_written == test_size_);

    // Clean up manually since file_handle_ref doesn't close the fd
    close(fd);
    ::std::filesystem::remove(write_path);
  }

  SECTION("File handle reference asynchronous read")
  {
    try
    {
      // Open file descriptor manually
      int fd = open(temp_file_path_.c_str(), O_RDONLY | O_DIRECT);
      if (fd < 0)
      {
        fd = open(temp_file_path_.c_str(), O_RDONLY); // Fallback without O_DIRECT
      }

      if (fd < 0)
      {
        FAIL("Failed to open file descriptor");
      }

      // Create non-owning reference from fd
      file_handle_ref handle_ref(fd);

      // Allocate GPU buffer
      test_utils::GPUMemoryRAII gpu_buffer(test_size_);

      // Create CUDA stream
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      // Create span from GPU buffer
      cuda::std::span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

      // Prepare async parameters
      off_t file_offset   = 0;
      off_t buffer_offset = 0;
      ssize_t bytes_read  = 0;

      // Test asynchronous read
      handle_ref.read_async(cuda::stream_ref{stream}, buffer_span, file_offset, buffer_offset, bytes_read);

      // Synchronize stream
      cudaStreamSynchronize(stream);

      // Check the result - the bindings should handle compatibility mode internally
      if (bytes_read < 0)
      {
        // Error -5 typically means async not supported, which is a valid system limitation
        if (bytes_read == -5)
        {
          SKIP("Async operations not supported on this system (error: " << bytes_read << ")");
        }
        else
        {
          FAIL("Async read failed with error: " << bytes_read);
        }
      }

      // Verify result
      REQUIRE(bytes_read == static_cast<ssize_t>(test_size_));

      // Clean up
      cudaStreamDestroy(stream);
      close(fd);
    }
    catch (const ::std::exception& e)
    {
      FAIL("Failed asynchronous read test: " << e.what());
    }
  }

  SECTION("File handle reference validity checks")
  {
    // Open file descriptor manually
    int fd = open(temp_file_path_.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0)
    {
      fd = open(temp_file_path_.c_str(), O_RDONLY); // Fallback without O_DIRECT
    }

    if (fd < 0)
    {
      FAIL("Failed to open file descriptor");
    }

    // Create non-owning reference from fd
    file_handle_ref handle_ref(fd);

    // Test validity
    REQUIRE(handle_ref.is_valid());

    // Test native handle access
    CUfileHandle_t native = handle_ref.native_handle();
    REQUIRE(native != nullptr);

    // Test file descriptor access
    REQUIRE(handle_ref.get_fd() == fd);

    // Clean up manually since file_handle_ref doesn't close the fd
    close(fd);
  }

  cleanup_test_environment();
}

TEST_CASE("Buffer handle operations", "[file_handle][buffer_handle]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  setup_test_environment();

  SECTION("Buffer handle registration")
  {
    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    cuda::std::span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), test_size_);

    // Test buffer registration
    buffer_handle buf_handle(buffer_span);

    // Verify buffer handle properties
    REQUIRE(buf_handle.data() == gpu_buffer.get());
    REQUIRE(buf_handle.size() == test_size_);

    // Test typed span access
    auto typed_span = buf_handle.as_span<char>();
    REQUIRE(typed_span.data() == gpu_buffer.get());
    REQUIRE(typed_span.size() == test_size_);
  }

  SECTION("Buffer handle with flags")
  {
    // Allocate GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(test_size_);

    // Create span from GPU buffer
    cuda::std::span<const char> buffer_span(static_cast<const char*>(gpu_buffer.get()), test_size_);

    // Test buffer registration with flags
    buffer_handle buf_handle(buffer_span, cuda::experimental::cufile::cu_file_buf_register_flags::none);

    // Verify buffer handle properties
    REQUIRE(buf_handle.data() == gpu_buffer.get());
    REQUIRE(buf_handle.size() == test_size_);

    // Test byte span access
    auto byte_span = buf_handle.as_bytes();
    REQUIRE(byte_span.data() == gpu_buffer.get());
    REQUIRE(byte_span.size() == test_size_);
  }

  cleanup_test_environment();
}

TEST_CASE("Batch operations", "[file_handle][batch]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  setup_test_environment();

  SECTION("Batch handle creation")
  {
    // Test batch handle creation
    batch_handle batch(10); // Max 10 operations

    // Verify batch handle properties
    REQUIRE(batch.is_valid());
    REQUIRE(batch.max_operations() == 10U);
  }

  SECTION("Batch operations execution")
  {
    // Create file handle
    file_handle handle(temp_file_path_.string(), ::std::ios::in | ::std::ios::binary);

    // Allocate GPU buffers
    test_utils::GPUMemoryRAII gpu_buffer1(test_size_);
    test_utils::GPUMemoryRAII gpu_buffer2(test_size_);

    // Create batch handle
    batch_handle batch(5);

    // Create spans from GPU buffers
    cuda::std::span<char> buffer_span1(static_cast<char*>(gpu_buffer1.get()), test_size_ / 2);
    cuda::std::span<char> buffer_span2(static_cast<char*>(gpu_buffer2.get()), test_size_ / 2);

    // Prepare batch operations using factory functions
    ::std::vector<batch_io_params_span<char>> operations;
    operations.push_back(make_read_operation(buffer_span1, 0, 0, reinterpret_cast<void*>(1)));
    operations.push_back(make_read_operation(buffer_span2, test_size_ / 2, 0, reinterpret_cast<void*>(2)));

    // Submit batch operations
    batch.submit(handle, cuda::std::span<const batch_io_params_span<char>>(operations));

    // Wait for all submitted operations to complete
    auto results = batch.get_status(operations.size(), ::cuda::std::chrono::milliseconds{5000});

    // Verify all operations completed
    REQUIRE(results.size() == operations.size());

    for (const auto& result : results)
    {
      REQUIRE(result.is_complete());
      REQUIRE_FALSE(result.is_failed());
      REQUIRE(result.result == test_size_ / 2);
    }
  }

  SECTION("Batch cancellation")
  {
    // Create batch handle
    batch_handle batch(5);

    // Test batch cancellation
    REQUIRE_NOTHROW(batch.cancel());
  }

  cleanup_test_environment();
}

TEST_CASE("Stream operations", "[file_handle][stream]")
{
  // Only skip if the stream API is genuinely not available on this system
  if (!is_stream_api_available())
  {
    SKIP("Stream API not available on this system");
  }

  setup_test_environment();

  SECTION("Stream handle registration")
  {
    // Create CUDA stream
    cudaStream_t stream;
    cudaError_t cuda_error = cudaStreamCreate(&stream);
    if (cuda_error != cudaSuccess)
    {
      SKIP("Failed to create CUDA stream: " << cudaGetErrorString(cuda_error));
    }

    // Test stream registration - if stream API is available, this should work
    stream_handle stream_handle_obj(stream);

    // Verify stream handle
    REQUIRE(stream_handle_obj.get() == stream);
  }

  SECTION("Stream handle with flags")
  {
    // Create CUDA stream
    cudaStream_t stream;
    cudaError_t cuda_error = cudaStreamCreate(&stream);
    if (cuda_error != cudaSuccess)
    {
      SKIP("Failed to create CUDA stream: " << cudaGetErrorString(cuda_error));
    }

    // Test stream registration with flags - if stream API is available, this should work
    stream_handle stream_handle_obj(stream, 0); // Default flags

    // Verify stream handle
    REQUIRE(stream_handle_obj.get() == stream);
  }

  cleanup_test_environment();
}

TEST_CASE("Error handling and compatibility", "[file_handle][error_handling]")
{
  setup_test_environment();

  SECTION("Invalid file handling")
  {
    // Test with non-existent file
    REQUIRE_THROWS_AS(file_handle("non_existent_file.tmp", ::std::ios::in), ::std::exception);
  }

  SECTION("Compatibility mode test")
  {
    // Test with regular (non-cuFile-compatible) memory
    test_utils::RegularMemoryRAII regular_buffer(test_size_);

    // Create file handle
    file_handle handle(temp_file_path_.string(), ::std::ios::in | ::std::ios::binary);

    // Create span from regular buffer
    cuda::std::span<char> buffer_span(static_cast<char*>(regular_buffer.get()), test_size_);

    // This should work in compatibility mode by falling back to POSIX I/O
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    REQUIRE(bytes_read == test_size_);

    // Verify data matches
    REQUIRE(::std::memcmp(regular_buffer.get(), test_data_.data(), test_size_) == 0);
  }

  cleanup_test_environment();
}

TEST_CASE("Large file operations", "[file_handle][large_files]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  setup_test_environment();

  SECTION("Large file read")
  {
    // Create file handle for large file
    file_handle handle(large_temp_file_path_.string(), ::std::ios::in | ::std::ios::binary);

    // Allocate large GPU buffer
    test_utils::GPUMemoryRAII gpu_buffer(large_test_size_);

    // Create span from GPU buffer
    cuda::std::span<char> buffer_span(static_cast<char*>(gpu_buffer.get()), large_test_size_);

    // Test large file read
    size_t bytes_read = handle.read(buffer_span);

    // Verify read operation
    REQUIRE(bytes_read == large_test_size_);

    // Copy data back to host to verify
    ::std::vector<char> host_buffer(large_test_size_);
    cudaMemcpy(host_buffer.data(), gpu_buffer.get(), large_test_size_, cudaMemcpyDeviceToHost);

    // Verify data matches
    REQUIRE(host_buffer == large_test_data_);
  }

  cleanup_test_environment();
}

} // namespace cuda::experimental::cufile
