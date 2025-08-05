#pragma once

// This file provides the implementation of buffer_handle methods
// It's included after the class definition

namespace cuda::experimental::cufile {

// Constructor implementations
template<typename T>
buffer_handle::buffer_handle(cuda::std::span<T> buffer, int flags)
    : buffer_(buffer.data()), size_(buffer.size_bytes()) {
    static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    CUfileError_t error = cuFileBufRegister(buffer_, size_, flags);
    detail::check_cufile_result(error, "cuFileBufRegister");

    registered_buffer_.emplace(buffer_, [](const void* buf) { cuFileBufDeregister(buf); });
}

template<typename T>
buffer_handle::buffer_handle(cuda::std::span<const T> buffer, int flags)
    : buffer_(buffer.data()), size_(buffer.size_bytes()) {
    static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");

    CUfileError_t error = cuFileBufRegister(buffer_, size_, flags);
    detail::check_cufile_result(error, "cuFileBufRegister");

    registered_buffer_.emplace(buffer_, [](const void* buf) { cuFileBufDeregister(buf); });
}

// Move constructor and assignment
inline buffer_handle::buffer_handle(buffer_handle&& other) noexcept
    : buffer_(other.buffer_), size_(other.size_), registered_buffer_(::std::move(other.registered_buffer_)) {
}

inline buffer_handle& buffer_handle::operator=(buffer_handle&& other) noexcept {
    if (this != &other) {
        buffer_ = other.buffer_;
        size_ = other.size_;
        registered_buffer_ = ::std::move(other.registered_buffer_);
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

inline cuda::std::span<const ::std::byte> buffer_handle::as_bytes() const noexcept {
    return cuda::std::span<const ::std::byte>(static_cast<const ::std::byte*>(buffer_), size_);
}

inline cuda::std::span<::std::byte> buffer_handle::as_writable_bytes() const noexcept {
    return cuda::std::span<::std::byte>(static_cast<::std::byte*>(const_cast<void*>(buffer_)), size_);
}

// Template method implementations
template<typename T>
cuda::std::span<T> buffer_handle::as_span() const noexcept {
    static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
    return cuda::std::span<T>(static_cast<T*>(const_cast<void*>(buffer_)), size_ / sizeof(T));
}

template<typename T>
cuda::std::span<const T> buffer_handle::as_const_span() const noexcept {
    static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
    return cuda::std::span<const T>(static_cast<const T*>(buffer_), size_ / sizeof(T));
}

// is_valid method implementation
inline bool buffer_handle::is_valid() const noexcept {
    return registered_buffer_.has_value();
}



} // namespace cuda::experimental::cufile