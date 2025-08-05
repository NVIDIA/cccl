#pragma once

#include "../driver.hpp"
#include "error_handling.hpp"

namespace cuda::experimental::cufile {

// ================================================================================================
// driver_properties Implementation (using C struct with C++ accessors)
// ================================================================================================

inline driver_properties::driver_properties() {
    CUfileError_t error = cuFileDriverGetProperties(&props_);
    detail::check_cufile_result(error, "cuFileDriverGetProperties");
}

inline unsigned int driver_properties::get_nvfs_major_version() const noexcept {
    return props_.nvfs.major_version;
}

inline unsigned int driver_properties::get_nvfs_minor_version() const noexcept {
    return props_.nvfs.minor_version;
}

inline size_t driver_properties::get_poll_threshold_size() const noexcept {
    return props_.nvfs.poll_thresh_size;
}

inline size_t driver_properties::get_max_direct_io_size() const noexcept {
    return props_.nvfs.max_direct_io_size;
}

inline unsigned int driver_properties::get_status_flags() const noexcept {
    return props_.nvfs.dstatusflags;
}

inline unsigned int driver_properties::get_control_flags() const noexcept {
    return props_.nvfs.dcontrolflags;
}

inline unsigned int driver_properties::get_feature_flags() const noexcept {
    return props_.fflags;
}

inline unsigned int driver_properties::get_max_device_cache_size() const noexcept {
    return props_.max_device_cache_size;
}

inline unsigned int driver_properties::get_per_buffer_cache_size() const noexcept {
    return props_.per_buffer_cache_size;
}

inline unsigned int driver_properties::get_max_pinned_memory_size() const noexcept {
    return props_.max_device_pinned_mem_size;
}

inline unsigned int driver_properties::get_max_batch_io_size() const noexcept {
    return props_.max_batch_io_size;
}

inline unsigned int driver_properties::get_max_batch_io_timeout_msecs() const noexcept {
    return props_.max_batch_io_timeout_msecs;
}

inline const CUfileDrvProps_t& driver_properties::get_raw_properties() const noexcept {
    return props_;
}

// ================================================================================================
// Driver Management Functions Implementation
// ================================================================================================

inline void driver_open() {
    CUfileError_t error = cuFileDriverOpen();
    detail::check_cufile_result(error, "cuFileDriverOpen");
}

inline void driver_close() {
    CUfileError_t error = cuFileDriverClose();
    detail::check_cufile_result(error, "cuFileDriverClose");
}

inline long driver_use_count() {
    return cuFileUseCount();
}

inline driver_properties get_driver_properties() {
    return driver_properties{}; // Will initialize in constructor
}

inline int get_version() {
    int version = 0;
    CUfileError_t error = cuFileGetVersion(&version);
    detail::check_cufile_result(error, "cuFileGetVersion");
    return version;
}

// ================================================================================================
// Driver Configuration Functions Implementation
// ================================================================================================

inline void set_poll_mode(bool enabled, size_t threshold_kb) {
    CUfileError_t error = cuFileDriverSetPollMode(enabled, threshold_kb);
    detail::check_cufile_result(error, "cuFileDriverSetPollMode");
}

inline void set_max_direct_io_size(size_t size_kb) {
    CUfileError_t error = cuFileDriverSetMaxDirectIOSize(size_kb);
    detail::check_cufile_result(error, "cuFileDriverSetMaxDirectIOSize");
}

inline void set_max_cache_size(size_t size_kb) {
    CUfileError_t error = cuFileDriverSetMaxCacheSize(size_kb);
    detail::check_cufile_result(error, "cuFileDriverSetMaxCacheSize");
}

inline void set_max_pinned_memory_size(size_t size_kb) {
    CUfileError_t error = cuFileDriverSetMaxPinnedMemSize(size_kb);
    detail::check_cufile_result(error, "cuFileDriverSetMaxPinnedMemSize");
}

// ================================================================================================
// Parameter Management Implementation
// ================================================================================================

inline size_t get_parameter_size_t(CUFileSizeTConfigParameter_t param) {
    size_t value;
    CUfileError_t error = cuFileGetParameterSizeT(param, &value);
    detail::check_cufile_result(error, "cuFileGetParameterSizeT");
    return value;
}

inline bool get_parameter_bool(CUFileBoolConfigParameter_t param) {
    bool value;
    CUfileError_t error = cuFileGetParameterBool(param, &value);
    detail::check_cufile_result(error, "cuFileGetParameterBool");
    return value;
}

inline ::std::string get_parameter_string(CUFileStringConfigParameter_t param) {
    char buffer[1024]; // Reasonable buffer size
    CUfileError_t error = cuFileGetParameterString(param, buffer, sizeof(buffer));
    detail::check_cufile_result(error, "cuFileGetParameterString");
    return ::std::string(buffer);
}

inline void set_parameter_size_t(CUFileSizeTConfigParameter_t param, size_t value) {
    CUfileError_t error = cuFileSetParameterSizeT(param, value);
    detail::check_cufile_result(error, "cuFileSetParameterSizeT");
}

inline void set_parameter_bool(CUFileBoolConfigParameter_t param, bool value) {
    CUfileError_t error = cuFileSetParameterBool(param, value);
    detail::check_cufile_result(error, "cuFileSetParameterBool");
}

inline void set_parameter_string(CUFileStringConfigParameter_t param, const ::std::string& value) {
    CUfileError_t error = cuFileSetParameterString(param, value.c_str());
    detail::check_cufile_result(error, "cuFileSetParameterString");
}

// ================================================================================================
// Statistics Management Implementation
// ================================================================================================

#ifdef CUfileStatsLevel1_t
inline void set_stats_level(int level) {
    CUfileError_t error = cuFileSetStatsLevel(level);
    detail::check_cufile_result(error, "cuFileSetStatsLevel");
}

inline int get_stats_level() {
    int level;
    CUfileError_t error = cuFileGetStatsLevel(&level);
    detail::check_cufile_result(error, "cuFileGetStatsLevel");
    return level;
}

inline void stats_start() {
    CUfileError_t error = cuFileStatsStart();
    detail::check_cufile_result(error, "cuFileStatsStart");
}

inline void stats_stop() {
    CUfileError_t error = cuFileStatsStop();
    detail::check_cufile_result(error, "cuFileStatsStop");
}

inline void stats_reset() {
    CUfileError_t error = cuFileStatsReset();
    detail::check_cufile_result(error, "cuFileStatsReset");
}

inline CUfileStatsLevel1_t get_stats_l1() {
    CUfileStatsLevel1_t stats;
    CUfileError_t error = cuFileGetStatsL1(&stats);
    detail::check_cufile_result(error, "cuFileGetStatsL1");
    return stats;
}

inline CUfileStatsLevel2_t get_stats_l2() {
    CUfileStatsLevel2_t stats;
    CUfileError_t error = cuFileGetStatsL2(&stats);
    detail::check_cufile_result(error, "cuFileGetStatsL2");
    return stats;
}

inline CUfileStatsLevel3_t get_stats_l3() {
    CUfileStatsLevel3_t stats;
    CUfileError_t error = cuFileGetStatsL3(&stats);
    detail::check_cufile_result(error, "cuFileGetStatsL3");
    return stats;
}
#endif // CUfileStatsLevel1_t

#ifdef cuFileGetBARSizeInKB
inline size_t get_bar_size_kb(int gpu_index) {
    size_t bar_size;
    CUfileError_t error = cuFileGetBARSizeInKB(gpu_index, &bar_size);
    detail::check_cufile_result(error, "cuFileGetBARSizeInKB");
    return bar_size;
}
#endif // cuFileGetBARSizeInKB

// ================================================================================================
// Capability Check Functions Implementation
// ================================================================================================

inline bool is_cufile_library_available() noexcept {
    try {
        // Try to get version - if it succeeds, library is available
        int version = 0;
        CUfileError_t error = cuFileGetVersion(&version);
        return (error.err == CU_FILE_SUCCESS);
    } catch (...) {
        return false;
    }
}

inline bool is_cufile_available() noexcept {
    try {
        get_driver_properties();
        return true;
    } catch (...) {
        return false;
    }
}

inline bool is_batch_api_available() noexcept {
    try {
        auto props = get_driver_properties();
        return props.batch_io_supported();
    } catch (...) {
        return false;
    }
}

inline bool is_stream_api_available() noexcept {
    try {
        auto props = get_driver_properties();
        return props.streams_supported();
    } catch (...) {
        return false;
    }
}

} // namespace cuda::experimental::cufile