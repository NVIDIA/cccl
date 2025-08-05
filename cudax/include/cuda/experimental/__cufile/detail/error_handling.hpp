#pragma once

#include <cufile.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda::experimental::cufile::detail {



/**
 * @brief Unified cuFile exception class
 */
class cufile_exception : public ::std::runtime_error {
private:
    CUfileError_t error_;

public:
    explicit cufile_exception(CUfileError_t error)
        : ::std::runtime_error(format_error_message(error)), error_(error) {}

    explicit cufile_exception(const ::std::string& message)
        : ::std::runtime_error(message), error_{CU_FILE_SUCCESS, CUDA_SUCCESS} {}

    CUfileError_t error() const noexcept { return error_; }

private:
    static ::std::string format_error_message(CUfileError_t error) {
        return ::std::string("cuFile error: ") + ::std::to_string(error.err)
               + " (CUDA: " + ::std::to_string(error.cu_err) + ")";
    }
};

/**
 * @brief Check cuFile operation result and throw on error
 */
inline void check_cufile_result(CUfileError_t error, const ::std::string& operation = "") {
    if (error.err != CU_FILE_SUCCESS) {
        ::std::string message = operation.empty() ? "" : operation + ": ";
        throw cufile_exception(error);
    }
}

/**
 * @brief Check cuFile operation result and throw on error (for ssize_t returns)
 */
inline ssize_t check_cufile_result(ssize_t result, const ::std::string& operation = "") {
    if (result < 0) {
        CUfileError_t error = {static_cast<CUfileOpError>(result), CUDA_SUCCESS};
        ::std::string message = operation.empty() ? "" : operation + ": ";
        throw cufile_exception(error);
    }
    return result;
}
} // namespace cuda::experimental::cufile::detail