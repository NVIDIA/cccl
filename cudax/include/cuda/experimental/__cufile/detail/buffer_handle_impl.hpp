#pragma once

// This file provides the implementation of buffer_handle methods
// It's included after the class definition

namespace cuda::experimental::cufile {

// Constructor implementations
template<typename T>
buffer_handle::buffer_handle(span<T> buffer, int flags)
    : detail::raii_handle<buffer_handle>(false), buffer_(buffer.data()), size_(buffer.size_bytes()) {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    CUfileError_t error = cuFileBufRegister(buffer_, size_, flags);
    detail::check_cufile_result(error, "cuFileBufRegister");
    set_owns_resource(true);
}

template<typename T>
buffer_handle::buffer_handle(span<const T> buffer, int flags)
    : detail::raii_handle<buffer_handle>(false), buffer_(buffer.data()), size_(buffer.size_bytes()) {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    CUfileError_t error = cuFileBufRegister(buffer_, size_, flags);
    detail::check_cufile_result(error, "cuFileBufRegister");
    set_owns_resource(true);
}

// Move constructor and assignment
inline buffer_handle::buffer_handle(buffer_handle&& other) noexcept
    : detail::raii_handle<buffer_handle>(std::move(other)), buffer_(other.buffer_), size_(other.size_) {
    // Base class handles owns_resource_ transfer
}

inline buffer_handle& buffer_handle::operator=(buffer_handle&& other) noexcept {
    if (this != &other) {
        detail::raii_handle<buffer_handle>::operator=(std::move(other));
        buffer_ = other.buffer_;
        size_ = other.size_;
    }
    return *this;
}

// Simple getter implementations
inline const void* buffer_handle::data() const noexcept {
    return buffer_;
}

inline size_t buffer_handle::size() const noexcept {
    return size_;
}

inline span<const std::byte> buffer_handle::as_bytes() const noexcept {
    return span<const std::byte>(static_cast<const std::byte*>(buffer_), size_);
}

inline span<std::byte> buffer_handle::as_writable_bytes() const noexcept {
    return span<std::byte>(static_cast<std::byte*>(const_cast<void*>(buffer_)), size_);
}

// Template method implementations
template<typename T>
span<T> buffer_handle::as_span() const noexcept {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
    return span<T>(static_cast<T*>(const_cast<void*>(buffer_)), size_ / sizeof(T));
}

template<typename T>
span<const T> buffer_handle::as_const_span() const noexcept {
    static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
    return span<const T>(static_cast<const T*>(buffer_), size_ / sizeof(T));
}

// Cleanup method
inline void buffer_handle::cleanup() noexcept {
    cuFileBufDeregister(buffer_);
}

} // namespace cuda::experimental::cufile