#pragma once

#include "detail/error_handling.hpp"
#include "detail/span_compat.hpp"

namespace cuda::io {

/**
 * @brief RAII wrapper for GPU buffer registration
 */
class buffer_handle : public detail::raii_handle<buffer_handle> {
private:
    const void* buffer_;
    size_t size_;

public:
    /**
     * @brief Register GPU buffer using span
     * @tparam T Element type (must be trivially copyable)
     * @param buffer Span representing the GPU buffer
     * @param flags Registration flags (default: 0)
     */
    template<typename T>
    explicit buffer_handle(span<T> buffer, int flags = 0);
    
    /**
     * @brief Register GPU buffer using span - const version
     * @tparam T Element type (must be trivially copyable)
     * @param buffer Span representing the GPU buffer
     * @param flags Registration flags (default: 0)
     */
    template<typename T>
    explicit buffer_handle(span<const T> buffer, int flags = 0);
    
    buffer_handle(buffer_handle&& other) noexcept;
    buffer_handle& operator=(buffer_handle&& other) noexcept;
    
    /**
     * @brief Get the registered buffer pointer
     */
    const void* data() const noexcept;
    
    /**
     * @brief Get the buffer size in bytes
     */
    size_t size() const noexcept;
    
    /**
     * @brief Get the buffer as a span of bytes
     */
    span<const std::byte> as_bytes() const noexcept;
    
    /**
     * @brief Get the buffer as a span of mutable bytes
     */
    span<std::byte> as_writable_bytes() const noexcept;
    
    /**
     * @brief Get the buffer as a typed span
     * @tparam T Element type (must be trivially copyable)
     * @return Span of type T over the buffer
     */
    template<typename T>
    span<T> as_span() const noexcept;
    
    /**
     * @brief Get the buffer as a typed const span
     * @tparam T Element type (must be trivially copyable)
     * @return Const span of type T over the buffer
     */
    template<typename T>
    span<const T> as_const_span() const noexcept;

private:
    friend class detail::raii_handle<buffer_handle>;
    
    /**
     * @brief Cleanup method required by CRTP base class
     */
    void cleanup() noexcept;
};

} // namespace cuda::io

#include "detail/buffer_handle_impl.hpp" 