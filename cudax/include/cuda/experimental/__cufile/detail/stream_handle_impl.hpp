#pragma once

// This file provides the implementation of stream_handle methods
// It's included after the class definition

namespace cuda::experimental {

// Constructor implementation
inline stream_handle::stream_handle(cudaStream_t stream, unsigned int flags)
    : detail::raii_handle<stream_handle>(false), stream_(stream) {
    CUfileError_t error = cuFileStreamRegister(stream, flags);
    detail::check_cufile_result(error, "cuFileStreamRegister");
    set_owns_resource(true);
}

// Move constructor and assignment
inline stream_handle::stream_handle(stream_handle&& other) noexcept
    : detail::raii_handle<stream_handle>(std::move(other)), stream_(other.stream_) {
    // Base class handles owns_resource_ transfer
}

inline stream_handle& stream_handle::operator=(stream_handle&& other) noexcept {
    if (this != &other) {
        detail::raii_handle<stream_handle>::operator=(std::move(other));
        stream_ = other.stream_;
    }
    return *this;
}

// Simple getter implementation
inline cudaStream_t stream_handle::get() const noexcept {
    return stream_;
}

// Cleanup method
inline void stream_handle::cleanup() noexcept {
    cuFileStreamDeregister(stream_);
}

} // namespace cuda::experimental