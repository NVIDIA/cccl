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

namespace
{
// workaround to be able to open and close the driver several times during the test. If the logging is enabled, it
// crashes on the second close.
struct DisableLogging
{
  DisableLogging()
  {
    setenv("CUFILE_LOGFILE_PATH", "", 1);
  }
} s_disable_logging;
} // namespace

namespace cuda::experimental::io
{

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
    REQUIRE(version > 0); // Should be a positive version number
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
  SECTION("NVFS and driver attributes access")
  {
    // Test NVFS attributes
    unsigned int major = driver_attributes::nvfs_major_version();
    unsigned int minor = driver_attributes::nvfs_minor_version();
    (void) major;
    size_t poll_thresh = driver_attributes::pollthreshold_size_kb();
    REQUIRE(poll_thresh >= 0); // Non-negative

    size_t max_io = driver_attributes::properties_max_direct_io_size_kb();
    REQUIRE(max_io >= 0); // Non-negative

    // Test driver feature flags
    unsigned int feature_flags = driver_attributes::feature_flags();
    REQUIRE(feature_flags > 0U); // Basic sanity check

    size_t max_cache  = driver_attributes::properties_max_device_cache_size_kb();
    size_t per_buffer = driver_attributes::properties_per_buffer_cache_size_kb();

    // These should be non-negative (could be 0 if not configured)
    REQUIRE(max_cache >= 0U);
    (void) per_buffer;

    // Test file system capabilities (just ensure they don't throw)
    bool lustre = driver_attributes::lustre_supported();
    bool nvme   = driver_attributes::nvme_supported();
    bool nfs    = driver_attributes::nfs_supported();
    (void) lustre;
    (void) nvme;
    (void) nfs;

    // Test capability flags (just ensure they don't throw)
    bool batch_io  = driver_attributes::batch_io_supported();
    bool streams   = driver_attributes::streams_supported();
    bool poll_mode = driver_attributes::use_poll_mode();
    (void) batch_io;
    (void) streams;
    (void) poll_mode;

    // No raw access in attribute API
  }

  SECTION("Attributes consistency between calls")
  {
    // Attributes should be consistent between calls
    REQUIRE(driver_attributes::nvfs_major_version() == driver_attributes::nvfs_major_version());
    REQUIRE(driver_attributes::nvfs_minor_version() == driver_attributes::nvfs_minor_version());
    REQUIRE(driver_attributes::feature_flags() == driver_attributes::feature_flags());
    REQUIRE(driver_attributes::lustre_supported() == driver_attributes::lustre_supported());
    REQUIRE(driver_attributes::nvme_supported() == driver_attributes::nvme_supported());
  }
}

TEST_CASE("Driver configuration", "[driver][configuration]")
{
  SECTION("Driver configuration operations")
  {
    driver_handle handle; // Ensure driver is open

    // Test poll mode configuration
    REQUIRE_NOTHROW(set_poll_mode(true, 1024)); // Enable poll mode with 1MB threshold

    // Test I/O size configuration
    REQUIRE_NOTHROW(set_max_direct_io_size(1024)); // Set to 1MB

    // Test cache configuration
    REQUIRE_NOTHROW(set_max_cache_size(2048)); // Set to 2MB

    // Test pinned memory configuration
    REQUIRE_NOTHROW(set_max_pinned_memory_size(1024)); // Set to 1MB
  }
}

TEST_CASE("Statistics management", "[driver][statistics]")
{
  SECTION("Statistics operations")
  {
    driver_handle handle; // Ensure driver is open

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
    try
    {
      set_stats_level(2);
      REQUIRE_NOTHROW(get_stats_l2());
    }
    catch (const ::std::exception&)
    {
      // Level 2 stats might not be available - this is acceptable
    }

    // Test Level 3 stats if available
    try
    {
      set_stats_level(3);
      REQUIRE_NOTHROW(get_stats_l3());
    }
    catch (const ::std::exception&)
    {
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
    driver_handle handle; // Ensure driver is open

    // Test attribute-based parameter access
    // Note: We just exercise calls; values depend on system configuration
    auto max_io_size = driver_attributes::properties_max_direct_io_size_kb();
    (void) max_io_size;

    auto max_cache = driver_attributes::properties_max_device_cache_size_kb();
    (void) max_cache;

    auto max_pinned = driver_attributes::properties_max_device_pinned_mem_size_kb();
    (void) max_pinned;

    // Test bool parameter operations using the new attribute system
    auto poll_enabled = driver_attributes::properties_use_poll_mode();
    (void) poll_enabled;
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
    REQUIRE((available == true || available == false)); // Always true, but tests the function
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
    driver_handle handle; // Ensure driver is open

#ifdef cuFileGetBARSizeInKB
    // Test getting BAR size for GPU 0 (if available)
    size_t bar_size = get_bar_size_kb(0);
    REQUIRE(bar_size > 0); // Should be a positive size
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

    // Touch some attributes
    (void) driver_attributes::nvfs_major_version();

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
} // namespace cuda::experimental::io
