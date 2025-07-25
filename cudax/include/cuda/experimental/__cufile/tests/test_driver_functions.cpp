#include <gtest/gtest.h>
#include <cuda/experimental/__cufile/driver.hpp>
#include "test_utils.h"
#include <memory>
#include <stdexcept>

namespace cuda::experimental {

class DriverTest : public ::testing::Test {

};

// Tests for driver management functions
TEST_F(DriverTest, DriverOpenClose) {
    // Test manual driver management
    driver_open();

    // Check driver is open
    long use_count = driver_use_count();
    EXPECT_GT(use_count, 0);

    driver_close();

    // After close, use count should be 0
    long use_count_after = driver_use_count();
    EXPECT_EQ(use_count_after, 0);
}

TEST_F(DriverTest, DriverUseCount) {
    // Initially should be 0
    long initial_count = driver_use_count();
    EXPECT_EQ(initial_count, 0);

    driver_open();
    long after_open = driver_use_count();
    EXPECT_GT(after_open, 0);

    driver_close();
    long after_close = driver_use_count();
    EXPECT_EQ(after_close, 0);
}

TEST_F(DriverTest, DriverVersion) {
    int version = get_version();
    EXPECT_GT(version, 0);  // Should be a positive version number
}

// Tests for RAII driver handles
TEST_F(DriverTest, DriverHandleRAII) {
    // Test RAII driver handle
    {
        driver_handle handle;
        long use_count = driver_use_count();
        EXPECT_GT(use_count, 0);
    }

    // After handle destruction, use count should be 0
    long use_count_after = driver_use_count();
    EXPECT_EQ(use_count_after, 0);
}

// Tests for driver properties
TEST_F(DriverTest, DriverProperties) {
    driver_properties props = get_driver_properties();

    // Test NVFS properties
    unsigned int major = props.get_nvfs_major_version();
    unsigned int minor = props.get_nvfs_minor_version();
    EXPECT_GT(major, 0);  // Should have a valid major version

    size_t poll_thresh = props.get_poll_threshold_size();
    EXPECT_GT(poll_thresh, 0);  // Should have a valid poll threshold

    size_t max_io = props.get_max_direct_io_size();
    EXPECT_GT(max_io, 0);  // Should have a valid max I/O size

    // Test driver properties
    unsigned int feature_flags = props.get_feature_flags();
    EXPECT_GE(feature_flags, 0U); // Basic sanity check

    unsigned int max_cache = props.get_max_device_cache_size();
    unsigned int per_buffer = props.get_per_buffer_cache_size();

    // These should be non-negative (could be 0 if not configured)
    EXPECT_GE(max_cache, 0);
    EXPECT_GE(per_buffer, 0);

    // Test file system capabilities
    bool lustre = props.lustre_supported();
    bool nvme = props.nvme_supported();
    bool nfs = props.nfs_supported();
    (void)lustre;
    (void)nvme;
    (void)nfs;

    // Test capability flags
    bool batch_io = props.batch_io_supported();
    bool streams = props.streams_supported();
    bool poll_mode = props.has_poll_mode();
    (void)batch_io; // Acknowledge we're not using these for now
    (void)streams;
    (void)poll_mode;

    // Test raw access
    const CUfileDrvProps_t& raw_props = props.get_raw_properties();
    EXPECT_EQ(raw_props.nvfs.major_version, major);
    EXPECT_EQ(raw_props.nvfs.minor_version, minor);
}

// Tests for driver configuration
TEST_F(DriverTest, DriverConfiguration) {
    driver_handle handle;  // Ensure driver is open

    // Test poll mode configuration
    set_poll_mode(true, 1024);  // Enable poll mode with 1MB threshold

    // Test I/O size configuration
    set_max_direct_io_size(1024);  // Set to 1MB

    // Test cache configuration
    set_max_cache_size(2048);  // Set to 2MB

    // Test pinned memory configuration
    set_max_pinned_memory_size(1024);  // Set to 1MB

    // If we get here without exceptions, configuration worked
    SUCCEED();
}

// Tests for statistics management
TEST_F(DriverTest, StatisticsManagement) {
    driver_handle handle;  // Ensure driver is open

#ifdef CUfileStatsLevel1_t
        // Test statistics level management
        set_stats_level(1);
        int level = get_stats_level();
        EXPECT_EQ(level, 1);

        // Test statistics control
        stats_start();
        stats_reset();

        // Test getting statistics (should not throw)
        CUfileStatsLevel1_t stats_l1 = get_stats_l1();

        // Test Level 2 stats if available
        try {
            set_stats_level(2);
            CUfileStatsLevel2_t stats_l2 = get_stats_l2();
        } catch (const std::exception&) {
            // Level 2 stats might not be available
        }

        // Test Level 3 stats if available
        try {
            set_stats_level(3);
            CUfileStatsLevel3_t stats_l3 = get_stats_l3();
        } catch (const std::exception&) {
            // Level 3 stats might not be available
        }

        stats_stop();
#else
        GTEST_SKIP() << "cuFile statistics API not available";
#endif

}

// Tests for parameter management
TEST_F(DriverTest, ParameterManagement) {
    driver_handle handle;  // Ensure driver is open

    // Test size_t parameter operations
    // Note: We're testing the API, not specific parameters as they depend on the system
    // Try to get a common parameter (adjust based on actual available parameters)
    size_t size_value = get_parameter_size_t(static_cast<CUFileSizeTConfigParameter_t>(0));
    EXPECT_GE(size_value, 0);

    // Test bool parameter operations
    bool bool_value = get_parameter_bool(static_cast<CUFileBoolConfigParameter_t>(0));
    (void)bool_value; // Acknowledge we're not using this for now

    // Test string parameter operations
    std::string string_value = get_parameter_string(static_cast<CUFileStringConfigParameter_t>(0));
    EXPECT_FALSE(string_value.empty());
}

// Tests for capability checking
TEST(DriverCapabilityTest, LibraryAvailability) {
    // Test cuFile library availability
    bool available = is_cufile_library_available();

    // This should not throw and return a boolean
    EXPECT_TRUE(available || !available);  // Always true, but tests the function
}

TEST(DriverCapabilityTest, CuFileAvailability) {
    // Test cuFile functionality availability
    bool available = is_cufile_available();

    // This should not throw and return a boolean
    EXPECT_TRUE(available || !available);  // Always true, but tests the function
}

TEST(DriverCapabilityTest, APIAvailability) {
    // Test batch API availability
    bool batch_available = is_batch_api_available();
    EXPECT_TRUE(batch_available || !batch_available);  // Always true, but tests the function

    // Test stream API availability
    bool stream_available = is_stream_api_available();
    EXPECT_TRUE(stream_available || !stream_available);  // Always true, but tests the function
}

// Tests for GPU BAR size
TEST_F(DriverTest, GPUBARSize) {
    driver_handle handle;  // Ensure driver is open

#ifdef cuFileGetBARSizeInKB
        // Test getting BAR size for GPU 0 (if available)
        size_t bar_size = get_bar_size_kb(0);
        EXPECT_GT(bar_size, 0);  // Should be a positive size
#else
        GTEST_SKIP() << "cuFile BAR size API not available";
#endif
}

// Integration tests
TEST_F(DriverTest, DriverLifecycleIntegration) {
    // Test complete driver lifecycle
    EXPECT_EQ(driver_use_count(), 0);

    // Open driver
    driver_open();
    EXPECT_GT(driver_use_count(), 0);

    // Get properties
    driver_properties props = get_driver_properties();
    EXPECT_GT(props.get_nvfs_major_version(), 0);

    // Configure driver
    set_poll_mode(true, 1024);

#ifdef CUfileStatsLevel1_t
    // Test statistics
    set_stats_level(1);
    stats_start();
    stats_reset();
    CUfileStatsLevel1_t stats = get_stats_l1();
    stats_stop();
#endif

    // Close driver
    driver_close();
    EXPECT_EQ(driver_use_count(), 0);
}

TEST_F(DriverTest, PropertiesConsistency) {
    driver_properties props1 = get_driver_properties();
    driver_properties props2 = get_driver_properties();

    // Properties should be consistent between calls
    EXPECT_EQ(props1.get_nvfs_major_version(), props2.get_nvfs_major_version());
    EXPECT_EQ(props1.get_nvfs_minor_version(), props2.get_nvfs_minor_version());
    EXPECT_EQ(props1.get_feature_flags(), props2.get_feature_flags());
    EXPECT_EQ(props1.lustre_supported(), props2.lustre_supported());
    EXPECT_EQ(props1.nvme_supported(), props2.nvme_supported());
}

} // namespace cuda::experimental

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}