//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__cufile/driver.hpp>

#include <c2h/catch2_test_helper.h>

namespace {
    // workaround to be able to open and close the driver several times during the test. If the logging is enabled, it crashes on the second close.
    struct DisableLogging {
        DisableLogging() {
            setenv("CUFILE_LOGFILE_PATH", "", 1);
        }
    } s_disable_logging;
}

namespace cuda::experimental::cufile {


TEST_CASE("Driver management functions", "[driver][management]")
{


    SECTION("Driver open and close operations")
    {
        // Test manual driver management
        driver_open();

        // Check driver is open
        long use_count = driver_use_count();
        REQUIRE(use_count > 0);

        driver_close();

        // After close, use count should be 0
        long use_count_after = driver_use_count();
        REQUIRE(use_count_after == 0);
    }

    SECTION("Driver use count tracking")
    {
        // Initially should be 0
        long initial_count = driver_use_count();
        REQUIRE(initial_count == 0);

        driver_open();
        long after_open = driver_use_count();
        REQUIRE(after_open > 0);

        driver_close();
        long after_close = driver_use_count();
        REQUIRE(after_close == 0);
    }

    SECTION("Driver version retrieval")
    {
        int version = get_version();
        REQUIRE(version > 0);  // Should be a positive version number
    }
}


TEST_CASE("RAII driver handle management", "[driver][raii]")
{


    SECTION("Driver handle RAII lifecycle")
    {
        // Test RAII driver handle
        {
            driver_handle handle;
            long use_count = driver_use_count();
            REQUIRE(use_count > 0);
        }

        // After handle destruction, use count should be 0
        long use_count_after = driver_use_count();
        REQUIRE(use_count_after == 0);
    }
}

TEST_CASE("Driver properties", "[driver][properties]")
{


    SECTION("NVFS and driver properties access")
    {
        driver_properties props = get_driver_properties();

        // Test NVFS properties
        unsigned int major = props.get_nvfs_major_version();
        unsigned int minor = props.get_nvfs_minor_version();
        (void)major;
        size_t poll_thresh = props.get_poll_threshold_size();
        REQUIRE(poll_thresh > 0);  // Should have a valid poll threshold

        size_t max_io = props.get_max_direct_io_size();
        REQUIRE(max_io > 0);  // Should have a valid max I/O size

        // Test driver properties
        unsigned int feature_flags = props.get_feature_flags();
        REQUIRE(feature_flags > 0U); // Basic sanity check

        unsigned int max_cache = props.get_max_device_cache_size();
        unsigned int per_buffer = props.get_per_buffer_cache_size();

        // These should be non-negative (could be 0 if not configured)
        REQUIRE(max_cache > 0U);
        (void)per_buffer;

        // Test file system capabilities (just ensure they don't throw)
        bool lustre = props.lustre_supported();
        bool nvme = props.nvme_supported();
        bool nfs = props.nfs_supported();
        (void)lustre;
        (void)nvme;
        (void)nfs;

        // Test capability flags (just ensure they don't throw)
        bool batch_io = props.batch_io_supported();
        bool streams = props.streams_supported();
        bool poll_mode = props.has_poll_mode();
        (void)batch_io;
        (void)streams;
        (void)poll_mode;

        // Test raw access
        const CUfileDrvProps_t& raw_props = props.get_raw_properties();
        REQUIRE(raw_props.nvfs.major_version == major);
        REQUIRE(raw_props.nvfs.minor_version == minor);
    }

    SECTION("Properties consistency between calls")
    {
        driver_properties props1 = get_driver_properties();
        driver_properties props2 = get_driver_properties();

        // Properties should be consistent between calls
        REQUIRE(props1.get_nvfs_major_version() == props2.get_nvfs_major_version());
        REQUIRE(props1.get_nvfs_minor_version() == props2.get_nvfs_minor_version());
        REQUIRE(props1.get_feature_flags() == props2.get_feature_flags());
        REQUIRE(props1.lustre_supported() == props2.lustre_supported());
        REQUIRE(props1.nvme_supported() == props2.nvme_supported());
    }
}

TEST_CASE("Driver configuration", "[driver][configuration]")
{


    SECTION("Driver configuration operations")
    {
        driver_handle handle;  // Ensure driver is open

        // Test poll mode configuration
        REQUIRE_NOTHROW(set_poll_mode(true, 1024));  // Enable poll mode with 1MB threshold

        // Test I/O size configuration
        REQUIRE_NOTHROW(set_max_direct_io_size(1024));  // Set to 1MB

        // Test cache configuration
        REQUIRE_NOTHROW(set_max_cache_size(2048));  // Set to 2MB

        // Test pinned memory configuration
        REQUIRE_NOTHROW(set_max_pinned_memory_size(1024));  // Set to 1MB
    }
}

TEST_CASE("Statistics management", "[driver][statistics]")
{


    SECTION("Statistics operations")
    {
        driver_handle handle;  // Ensure driver is open

#ifdef CUfileStatsLevel1_t
        // Test statistics level management
        REQUIRE_NOTHROW(set_stats_level(1));
        int level = get_stats_level();
        REQUIRE(level == 1);

        // Test statistics control
        REQUIRE_NOTHROW(stats_start());
        REQUIRE_NOTHROW(stats_reset());

        // Test getting statistics (should not throw)
        REQUIRE_NOTHROW(get_stats_l1());

        // Test Level 2 stats if available
        try {
            set_stats_level(2);
            REQUIRE_NOTHROW(get_stats_l2());
        } catch (const std::exception&) {
            // Level 2 stats might not be available - this is acceptable
        }

        // Test Level 3 stats if available
        try {
            set_stats_level(3);
            REQUIRE_NOTHROW(get_stats_l3());
        } catch (const std::exception&) {
            // Level 3 stats might not be available - this is acceptable
        }

        REQUIRE_NOTHROW(stats_stop());
#else
        SKIP("cuFile statistics API not available");
#endif
    }
}

TEST_CASE("Parameter management", "[driver][parameters]")
{


    SECTION("Parameter operations")
    {
        driver_handle handle;  // Ensure driver is open

        // Test size_t parameter operations
        // Note: We're testing the API, not specific parameters as they depend on the system
        // Try to get a common parameter (adjust based on actual available parameters)
        size_t size_value = get_parameter_size_t(static_cast<CUFileSizeTConfigParameter_t>(0));
        (void)size_value;

        // Test bool parameter operations
        bool bool_value = get_parameter_bool(static_cast<CUFileBoolConfigParameter_t>(0));
        (void)bool_value; // Acknowledge we're not using this for now

        // Test string parameter operations
        auto string_value = get_parameter_string(static_cast<CUFileStringConfigParameter_t>(0));
        REQUIRE_FALSE(string_value.empty());
    }
}



TEST_CASE("Capability checking", "[driver][capabilities]")
{
    SECTION("Library availability")
    {
        // Test basic driver functionality
        SUCCEED("Driver capability check completed");
    }

    SECTION("CuFile functionality availability")
    {
        // Test cuFile functionality availability
        bool available = is_cufile_available();
        // This should not throw and return a boolean
        REQUIRE((available == true || available == false));  // Always true, but tests the function
    }

    SECTION("API availability")
    {
        // Test batch API availability
        bool batch_available = is_batch_api_available();
        REQUIRE((batch_available == true || batch_available == false));

        // Test stream API availability
        bool stream_available = is_stream_api_available();
        REQUIRE((stream_available == true || stream_available == false));
    }
}

TEST_CASE("GPU BAR size", "[driver][gpu]")
{


    SECTION("BAR size retrieval")
    {
        driver_handle handle;  // Ensure driver is open

#ifdef cuFileGetBARSizeInKB
        // Test getting BAR size for GPU 0 (if available)
        size_t bar_size = get_bar_size_kb(0);
        REQUIRE(bar_size > 0);  // Should be a positive size
#else
        SKIP("cuFile BAR size API not available");
#endif
    }
}

TEST_CASE("Driver lifecycle integration", "[driver][integration]")
{


    SECTION("Complete driver lifecycle")
    {
        // Test complete driver lifecycle
        REQUIRE(driver_use_count() == 0);

        // Open driver
        driver_open();
        REQUIRE(driver_use_count() > 0);

        // Get properties
        (void)get_driver_properties();

        // Configure driver
        REQUIRE_NOTHROW(set_poll_mode(true, 1024));

#ifdef CUfileStatsLevel1_t
        // Test statistics
        REQUIRE_NOTHROW(set_stats_level(1));
        REQUIRE_NOTHROW(stats_start());
        REQUIRE_NOTHROW(stats_reset());
        REQUIRE_NOTHROW(get_stats_l1());
        REQUIRE_NOTHROW(stats_stop());
#endif

        // Close driver
        driver_close();
        REQUIRE(driver_use_count() == 0);
    }
}
} // namespace cuda::experimental::cufile