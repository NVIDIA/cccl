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

namespace cuda::experimental::cufile {

/**
 * @brief RAII wrapper for CUDA stream registration with cuFILE
 */
class stream_handle : public detail::raii_handle<stream_handle> {
private:
    cudaStream_t stream_;

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

private:
    friend class detail::raii_handle<stream_handle>;

    /**
     * @brief Cleanup method required by CRTP base class
     */
    void cleanup() noexcept;
};

} // namespace cuda::experimental::cufile

#include "detail/stream_handle_impl.hpp"