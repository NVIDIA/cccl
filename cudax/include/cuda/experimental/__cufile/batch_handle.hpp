#pragma once

#include "detail/error_handling.hpp"
#include "detail/span_compat.hpp"
#include <vector>

namespace cuda::io {

// Forward declarations
class file_handle;

/**
 * @brief Batch I/O operation descriptor using span
 * @tparam T Element type (must be trivially copyable)
 */
template<typename T>
struct batch_io_params_span {
    span<T> buffer;             ///< Buffer span
    off_t file_offset;          ///< File offset
    off_t buffer_offset;        ///< Buffer offset (in bytes)
    CUfileOpcode_t opcode;      ///< CUFILE_READ or CUFILE_WRITE
    void* cookie;               ///< User data for tracking
    
    // Constructor
    batch_io_params_span(span<T> buf, off_t f_off, off_t b_off, CUfileOpcode_t op, void* ck = nullptr);
};

/**
 * @brief Batch I/O operation result
 */
struct batch_io_result {
    void* cookie;               ///< User data from operation
    CUfileStatus_t status;      ///< Operation status
    size_t result;              ///< Bytes transferred or error code
    
    bool is_complete() const noexcept;
    bool is_failed() const noexcept;
    bool has_error() const noexcept;
};

/**
 * @brief RAII wrapper for batch operations
 */
class batch_handle : public detail::raii_handle<batch_handle> {
private:
    CUfileBatchHandle_t handle_;
    unsigned int max_operations_;

public:
    /**
     * @brief Create batch handle
     * @param max_operations Maximum number of operations
     */
    explicit batch_handle(unsigned int max_operations);
    
    batch_handle(batch_handle&& other) noexcept;
    batch_handle& operator=(batch_handle&& other) noexcept;
    
    /**
     * @brief Submit batch operations using span
     * @tparam T Element type (must be trivially copyable)
     * @param file_handle_ref File handle to operate on
     * @param operations Span of span-based batch operations
     * @param flags Additional flags (default: 0)
     */
    template<typename T>
    void submit(const file_handle& file_handle_ref,
               span<const batch_io_params_span<T>> operations,
               unsigned int flags = 0);
    

    
    /**
     * @brief Get batch status
     */
    std::vector<batch_io_result> get_status(unsigned int min_completed,
                                           int timeout_ms = 0);
    
    /**
     * @brief Cancel batch operations
     */
    void cancel();
    
    /**
     * @brief Get maximum operations capacity
     */
    unsigned int max_operations() const noexcept;

private:
    friend class detail::raii_handle<batch_handle>;
    
    /**
     * @brief Cleanup method required by CRTP base class
     */
    void cleanup() noexcept;
};

/**
 * @brief Create a read operation for batch processing
 * @tparam T Element type
 * @param buffer Buffer span to read into
 * @param file_offset File offset to read from
 * @param buffer_offset Buffer offset (in bytes)
 * @param cookie User data for tracking
 */
template<typename T>
batch_io_params_span<T> make_read_operation(span<T> buffer, off_t file_offset, 
                                            off_t buffer_offset = 0, void* cookie = nullptr);

/**
 * @brief Create a write operation for batch processing
 * @tparam T Element type
 * @param buffer Buffer span to write from
 * @param file_offset File offset to write to
 * @param buffer_offset Buffer offset (in bytes)
 * @param cookie User data for tracking
 */
template<typename T>
batch_io_params_span<const T> make_write_operation(span<const T> buffer, off_t file_offset, 
                                                   off_t buffer_offset = 0, void* cookie = nullptr);

} // namespace cuda::io 