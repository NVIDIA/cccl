#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "detail/error_handling.hpp"
#include "detail/span_compat.hpp"
#include "buffer_handle.hpp"
#include "batch_handle.hpp"
#include "stream_handle.hpp"

#include <fcntl.h>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace cuda::experimental::cufile {

/**
 * @brief RAII file handle for cuFILE operations
 */
class file_handle : public detail::raii_handle<file_handle> {
private:
    int fd_;
    bool owns_fd_;
    std::string path_;
    CUfileHandle_t handle_;

    static int convert_ios_mode(std::ios_base::openmode mode);
    void register_file();

public:
    /**
     * @brief Open file for cuFILE operations
     * @param path File path
     * @param mode STL-compatible open mode flags
     */
    explicit file_handle(const std::string& path,
                        std::ios_base::openmode mode = std::ios_base::in);

    /**
     * @brief Create from existing file descriptor
     * @param fd File descriptor (should be opened with O_DIRECT)
     * @param take_ownership Whether to close fd in destructor
     */
    explicit file_handle(int fd, bool take_ownership = false);

    file_handle(file_handle&& other) noexcept;
    file_handle& operator=(file_handle&& other) noexcept;
    ~file_handle() noexcept;

    /**
     * @brief Read data from file using span
     * @tparam T Element type (must be trivially copyable)
     * @param buffer Span representing the destination buffer
     * @param file_offset Offset in file to read from
     * @param buffer_offset Offset in buffer to read into (in bytes)
     * @return Number of bytes read
     */
    template<typename T>
    size_t read(span<T> buffer, off_t file_offset = 0, off_t buffer_offset = 0);

    /**
     * @brief Write data to file using span
     * @tparam T Element type (must be trivially copyable)
     * @param buffer Span representing the source buffer
     * @param file_offset Offset in file to write to
     * @param buffer_offset Offset in buffer to write from (in bytes)
     * @return Number of bytes written
     */
    template<typename T>
    size_t write(span<const T> buffer, off_t file_offset = 0, off_t buffer_offset = 0);

    /**
     * @brief Asynchronous read using span
     * @tparam T Element type (must be trivially copyable)
     * @param buffer Span representing the destination buffer
     * @param file_offset Offset in file to read from
     * @param buffer_offset Offset in buffer to read into (in bytes)
     * @param bytes_read Output parameter for bytes read
     * @param stream CUDA stream for async operation
     */
    template<typename T>
    void read_async(span<T> buffer,
                   off_t file_offset,
                   off_t buffer_offset,
                   ssize_t& bytes_read,
                   cudaStream_t stream);

    /**
     * @brief Asynchronous write using span
     * @tparam T Element type (must be trivially copyable)
     * @param buffer Span representing the source buffer
     * @param file_offset Offset in file to write to
     * @param buffer_offset Offset in buffer to write from (in bytes)
     * @param bytes_written Output parameter for bytes written
     * @param stream CUDA stream for async operation
     */
    template<typename T>
    void write_async(span<const T> buffer,
                    off_t file_offset,
                    off_t buffer_offset,
                    ssize_t& bytes_written,
                    cudaStream_t stream);

    /**
     * @brief Get native cuFILE handle
     */
    CUfileHandle_t native_handle() const noexcept;

    /**
     * @brief Get file path
     */
    const std::string& path() const noexcept;

private:
    friend class detail::raii_handle<file_handle>;

    /**
     * @brief Cleanup method required by CRTP base class
     */
    void cleanup() noexcept;
};

} // namespace cuda::experimental::cufile

#include "detail/file_handle_impl.hpp"
#include "detail/batch_handle_impl.hpp"