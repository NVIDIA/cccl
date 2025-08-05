#pragma once

// This file provides the implementation of batch_handle methods
// It's included after the class definition to avoid circular dependency issues

#include <span>

namespace cuda::experimental::cufile {

// Forward declaration to avoid including file_handle.hpp in batch_handle.hpp
class file_handle;

// batch_io_params_span constructor implementation
template<typename T>
batch_io_params_span<T>::batch_io_params_span(cuda::std::span<T> buf, off_t f_off, off_t b_off, CUfileOpcode_t op, void* ck)
    : buffer(buf), file_offset(f_off), buffer_offset(b_off), opcode(op), cookie(ck) {
    static_assert(::std::is_trivially_copyable_v<T>, "Type must be trivially copyable for cuFile operations");
}

// batch_io_result method implementations
inline bool batch_io_result::is_complete() const noexcept {
    return status == CUFILE_COMPLETE;
}

inline bool batch_io_result::is_failed() const noexcept {
    return status == CUFILE_FAILED;
}

inline bool batch_io_result::has_error() const noexcept {
    return static_cast<ssize_t>(result) < 0;
}

// batch_handle constructor implementation
inline batch_handle::batch_handle(unsigned int max_operations)
    : max_operations_(max_operations) {
    CUfileError_t error = cuFileBatchIOSetUp(&handle_, max_operations);
    detail::check_cufile_result(error, "cuFileBatchIOSetUp");

    batch_resource_.emplace(handle_, [](CUfileBatchHandle_t h) { cuFileBatchIODestroy(h); });
}

// batch_handle move constructor and assignment
inline batch_handle::batch_handle(batch_handle&& other) noexcept
    : handle_(other.handle_), max_operations_(other.max_operations_), batch_resource_(::std::move(other.batch_resource_)) {
}

inline batch_handle& batch_handle::operator=(batch_handle&& other) noexcept {
    if (this != &other) {
        handle_ = other.handle_;
        max_operations_ = other.max_operations_;
        batch_resource_ = ::std::move(other.batch_resource_);
    }
    return *this;
}

// batch_handle method implementations
inline ::std::vector<batch_io_result> batch_handle::get_status(unsigned int min_completed,
                                                           int timeout_ms) {
    ::std::vector<CUfileIOEvents_t> events(max_operations_);
    unsigned int num_events = max_operations_;

    struct timespec timeout_spec = {};
    struct timespec* timeout_ptr = nullptr;

    if (timeout_ms > 0) {
        timeout_spec.tv_sec = timeout_ms / 1000;
        timeout_spec.tv_nsec = (timeout_ms % 1000) * 1000000;
        timeout_ptr = &timeout_spec;
    }

    CUfileError_t error = cuFileBatchIOGetStatus(handle_, min_completed,
                                                &num_events, events.data(),
                                                timeout_ptr);
    detail::check_cufile_result(error, "cuFileBatchIOGetStatus");

    ::std::vector<batch_io_result> results;
    results.reserve(num_events);

    for (unsigned int i = 0; i < num_events; ++i) {
        batch_io_result result = {};
        result.cookie = events[i].cookie;
        result.status = events[i].status;
        result.result = events[i].ret;
        results.push_back(result);
    }

    return results;
}

inline void batch_handle::cancel() {
    CUfileError_t error = cuFileBatchIOCancel(handle_);
    detail::check_cufile_result(error, "cuFileBatchIOCancel");
    batch_resource_.release();
}

inline unsigned int batch_handle::max_operations() const noexcept {
    return max_operations_;
}

inline bool batch_handle::is_valid() const noexcept {
    return batch_resource_.has_value();
}



// Template method implementations that require complete file_handle definition

template<typename T>
void batch_handle::submit(const file_handle& file_handle_ref,
                         cuda::std::span<const batch_io_params_span<T>> operations,
                         unsigned int flags) {
    ::std::vector<CUfileIOParams_t> cufile_ops;
    cufile_ops.reserve(operations.size());

    for (const auto& op : operations) {
        CUfileIOParams_t cufile_op = {};
        cufile_op.mode = CUFILE_BATCH;
        cufile_op.u.batch.devPtr_base = op.buffer.data();
        cufile_op.u.batch.file_offset = op.file_offset;
        cufile_op.u.batch.devPtr_offset = op.buffer_offset;
        cufile_op.u.batch.size = op.buffer.size_bytes();
        cufile_op.fh = file_handle_ref.native_handle();
        cufile_op.opcode = op.opcode;
        cufile_op.cookie = op.cookie;
        cufile_ops.push_back(cufile_op);
    }

    CUfileError_t error = cuFileBatchIOSubmit(handle_, cufile_ops.size(),
                                             cufile_ops.data(), flags);
    detail::check_cufile_result(error, "cuFileBatchIOSubmit");
}



// Free function template implementations
template<typename T>
batch_io_params_span<T> make_read_operation(cuda::std::span<T> buffer, off_t file_offset,
                                            off_t buffer_offset, void* cookie) {
    return batch_io_params_span<T>(buffer, file_offset, buffer_offset, CUFILE_READ, cookie);
}

template<typename T>
batch_io_params_span<const T> make_write_operation(cuda::std::span<const T> buffer, off_t file_offset,
                                                   off_t buffer_offset, void* cookie) {
    return batch_io_params_span<const T>(buffer, file_offset, buffer_offset, CUFILE_WRITE, cookie);
}

} // namespace cuda::experimental::cufile