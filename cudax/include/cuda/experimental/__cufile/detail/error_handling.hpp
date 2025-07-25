#pragma once

#include <cufile.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda::experimental::detail {

/**
 * @brief CRTP base class for RAII handles with move semantics
 *
 * Provides common resource ownership and cleanup patterns.
 * Derived classes must implement cleanup() method.
 */
template<typename Derived>
class raii_handle {
protected:
    bool owns_resource_;

    explicit raii_handle(bool owns = false) : owns_resource_(owns) {}

    // Move constructor
    raii_handle(raii_handle&& other) noexcept
        : owns_resource_(other.owns_resource_) {
        other.owns_resource_ = false;
    }

    // Move assignment
    raii_handle& operator=(raii_handle&& other) noexcept {
        if (this != &other) {
            if (owns_resource_) {
                static_cast<Derived*>(this)->cleanup();
            }
            owns_resource_ = other.owns_resource_;
            other.owns_resource_ = false;
        }
        return *this;
    }

    // Non-copyable
    raii_handle(const raii_handle&) = delete;
    raii_handle& operator=(const raii_handle&) = delete;

    ~raii_handle() {
        if (owns_resource_) {
            static_cast<Derived*>(this)->cleanup();
        }
    }

public:
    /**
     * @brief Check if the handle owns a valid resource
     */
    bool is_valid() const noexcept { return owns_resource_; }

protected:
    /**
     * @brief Set resource ownership state
     */
    void set_owns_resource(bool owns) noexcept { owns_resource_ = owns; }
};

/**
 * @brief Unified cuFile exception class
 */
class cufile_exception : public std::runtime_error {
private:
    CUfileError_t error_;

public:
    explicit cufile_exception(CUfileError_t error)
        : std::runtime_error(format_error_message(error)), error_(error) {}

    explicit cufile_exception(const std::string& message)
        : std::runtime_error(message), error_{CU_FILE_SUCCESS, CUDA_SUCCESS} {}

    CUfileError_t error() const noexcept { return error_; }

private:
    static std::string format_error_message(CUfileError_t error) {
        return std::string("cuFile error: ") + std::to_string(error.err)
               + " (CUDA: " + std::to_string(error.cu_err) + ")";
    }
};

/**
 * @brief Check cuFile operation result and throw on error
 */
inline void check_cufile_result(CUfileError_t error, const std::string& operation = "") {
    if (error.err != CU_FILE_SUCCESS) {
        std::string message = operation.empty() ? "" : operation + ": ";
        throw cufile_exception(error);
    }
}

/**
 * @brief Check cuFile operation result and throw on error (for ssize_t returns)
 */
inline ssize_t check_cufile_result(ssize_t result, const std::string& operation = "") {
    if (result < 0) {
        CUfileError_t error = {static_cast<CUfileOpError>(result), CUDA_SUCCESS};
        std::string message = operation.empty() ? "" : operation + ": ";
        throw cufile_exception(error);
    }
    return result;
}

/**
 * @brief RAII wrapper for automatic resource cleanup
 */
template<typename T, typename Deleter>
class raii_wrapper {
private:
    T resource_;
    Deleter deleter_;
    bool owns_resource_;

public:
    explicit raii_wrapper(T resource, Deleter deleter)
        : resource_(resource), deleter_(deleter), owns_resource_(true) {}

    ~raii_wrapper() {
        if (owns_resource_) {
            deleter_(resource_);
        }
    }

    raii_wrapper(const raii_wrapper&) = delete;
    raii_wrapper& operator=(const raii_wrapper&) = delete;

    raii_wrapper(raii_wrapper&& other) noexcept
        : resource_(other.resource_), deleter_(std::move(other.deleter_)), owns_resource_(other.owns_resource_) {
        other.owns_resource_ = false;
    }

    raii_wrapper& operator=(raii_wrapper&& other) noexcept {
        if (this != &other) {
            if (owns_resource_) {
                deleter_(resource_);
            }
            resource_ = other.resource_;
            deleter_ = std::move(other.deleter_);
            owns_resource_ = other.owns_resource_;
            other.owns_resource_ = false;
        }
        return *this;
    }

    T get() const noexcept { return resource_; }
    T release() noexcept {
        owns_resource_ = false;
        return resource_;
    }
};

} // namespace cuda::experimental::detail