#pragma once

// This file provides the implementation of file_handle methods
// It's included after the class definition to avoid circular dependency issues

#include <fcntl.h>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace cuda::experimental::cufile {

// Static method implementations
inline int file_handle::convert_ios_mode(std::ios_base::openmode mode) {
    int flags = 0;

    bool has_in = (mode & std::ios_base::in) != 0;
    bool has_out = (mode & std::ios_base::out) != 0;

    if (has_in && has_out) {
        flags |= O_RDWR;
    } else if (has_out) {
        flags |= O_WRONLY;
    } else {
        flags |= O_RDONLY;
    }

    if (mode & std::ios_base::trunc) {
        flags |= O_TRUNC;
    }

    if (mode & std::ios_base::app) {
        flags |= O_APPEND;
    }

    if (has_out) {
        flags |= O_CREAT;
    }

    flags |= O_DIRECT;

    return flags;
}

inline void file_handle::register_file() {
    CUfileDescr_t desc = {};
    desc.handle.fd = fd_;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    desc.fs_ops = nullptr;

    CUfileError_t error = cuFileHandleRegister(&handle_, &desc);
    detail::check_cufile_result(error, "cuFileHandleRegister");
    set_owns_resource(true);
}

// Constructor implementations
inline file_handle::file_handle(const std::string& path,
                               std::ios_base::openmode mode)
    : detail::raii_handle<file_handle>(false), owns_fd_(true), path_(path) {

    int flags = convert_ios_mode(mode);
    fd_ = open(path.c_str(), flags, 0644);

    if (fd_ < 0) {
        throw std::system_error(errno, std::system_category(),
                               "Failed to open file: " + path);
    }

    register_file();
}

inline file_handle::file_handle(int fd, bool take_ownership)
    : detail::raii_handle<file_handle>(false), fd_(fd), owns_fd_(take_ownership),
      path_("fd:" + std::to_string(fd)) {
    if (fd_ < 0) {
        throw std::invalid_argument("Invalid file descriptor");
    }

    register_file();
}

// Move constructor and assignment
inline file_handle::file_handle(file_handle&& other) noexcept
    : detail::raii_handle<file_handle>(std::move(other)),
      fd_(other.fd_), owns_fd_(other.owns_fd_), path_(std::move(other.path_)),
      handle_(other.handle_) {
    other.owns_fd_ = false;
}

inline file_handle& file_handle::operator=(file_handle&& other) noexcept {
    if (this != &other) {
        // Clean up file descriptor if we own it
        if (owns_fd_ && fd_ >= 0) {
            close(fd_);
        }

        detail::raii_handle<file_handle>::operator=(std::move(other));
        fd_ = other.fd_;
        owns_fd_ = other.owns_fd_;
        path_ = std::move(other.path_);
        handle_ = other.handle_;

        other.owns_fd_ = false;
    }
    return *this;
}

// Destructor
inline file_handle::~file_handle() noexcept {
    if (owns_fd_ && fd_ >= 0) {
        close(fd_);
    }
}

// Template method implementations
template<typename T>
size_t file_handle::read(span<T> buffer, off_t file_offset, off_t buffer_offset) {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    // Convert span to void* and size for cuFile API
    void* buffer_ptr = static_cast<void*>(buffer.data());
    size_t size_bytes = buffer.size_bytes();

    ssize_t result = cuFileRead(handle_, buffer_ptr, size_bytes, file_offset, buffer_offset);
    return static_cast<size_t>(detail::check_cufile_result(result, "cuFileRead"));
}

template<typename T>
size_t file_handle::write(span<const T> buffer, off_t file_offset, off_t buffer_offset) {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    // Convert span to void* and size for cuFile API
    const void* buffer_ptr = static_cast<const void*>(buffer.data());
    size_t size_bytes = buffer.size_bytes();

    ssize_t result = cuFileWrite(handle_, buffer_ptr, size_bytes, file_offset, buffer_offset);
    return static_cast<size_t>(detail::check_cufile_result(result, "cuFileWrite"));
}

template<typename T>
void file_handle::read_async(span<T> buffer,
                            off_t file_offset,
                            off_t buffer_offset,
                            ssize_t& bytes_read,
                            cudaStream_t stream) {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    // cuFile async API requires size parameters to be passed by pointer
    size_t size_bytes = buffer.size_bytes();
    void* buffer_ptr = static_cast<void*>(buffer.data());

    CUfileError_t error = cuFileReadAsync(handle_, buffer_ptr, &size_bytes,
                                         &file_offset, &buffer_offset,
                                         &bytes_read, stream);
    detail::check_cufile_result(error, "cuFileReadAsync");
}

template<typename T>
void file_handle::write_async(span<const T> buffer,
                             off_t file_offset,
                             off_t buffer_offset,
                             ssize_t& bytes_written,
                             cudaStream_t stream) {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    // cuFile async API requires size parameters to be passed by pointer
    size_t size_bytes = buffer.size_bytes();
    const void* buffer_ptr = static_cast<const void*>(buffer.data());

    CUfileError_t error = cuFileWriteAsync(handle_, const_cast<void*>(buffer_ptr), &size_bytes,
                                          &file_offset, &buffer_offset,
                                          &bytes_written, stream);
    detail::check_cufile_result(error, "cuFileWriteAsync");
}

// Simple getter implementations
inline CUfileHandle_t file_handle::native_handle() const noexcept {
    return handle_;
}

inline const std::string& file_handle::path() const noexcept {
    return path_;
}

// Cleanup method
inline void file_handle::cleanup() noexcept {
    cuFileHandleDeregister(handle_);
}

} // namespace cuda::experimental::cufile