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
#include <functional>
#include <optional>

namespace cuda::experimental::cufile {

/**
 * @brief RAII wrapper for CUDA stream registration with cuFILE
 */
class stream_handle {
private:
    cudaStream_t stream_;
    std::optional<detail::raii_resource<cudaStream_t, std::function<void(cudaStream_t)>>> registered_stream_;

public:
    /**
     * @brief Register CUDA stream
     * @param stream CUDA stream to register
     * @param flags Stream flags (see CU_FILE_STREAM_* constants)
     */
    stream_handle(cudaStream_t stream, unsigned int flags = 0);

    stream_handle(stream_handle&& other) noexcept;
    stream_handle& operator=(stream_handle&& other) noexcept;

    /**
     * @brief Get the registered CUDA stream
     */
    cudaStream_t get() const noexcept;

    /**
     * @brief Check if the handle owns a valid resource
     */
    bool is_valid() const noexcept;


};

} // namespace cuda::experimental::cufile

#include "detail/stream_handle_impl.hpp"