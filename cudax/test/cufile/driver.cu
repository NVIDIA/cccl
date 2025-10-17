//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/cufile.cuh>

#include <stdexcept>

#include <testing.cuh>

void test_is_open(bool is_open)
{
  STATIC_REQUIRE(cuda::std::is_same_v<bool, decltype(cudax::cufile_driver.is_open())>);
  STATIC_REQUIRE(noexcept(cudax::cufile_driver.is_open()));

  CUDAX_REQUIRE(cudax::cufile_driver.is_open() == is_open);
}

template <class Attr>
void test_attribute_range([[maybe_unused]] const Attr& attr, [[maybe_unused]] const bool is_open)
{
#if _CCCL_CTK_AT_LEAST(13, 0)
  STATIC_REQUIRE(cuda::std::is_same_v<cudax::cufile_driver_attribute_range<Attr>,
                                      decltype(cudax::cufile_driver.attribute_range(attr))>);
  STATIC_REQUIRE(!noexcept(cudax::cufile_driver.attribute_range(attr)));

  const bool expect_success =
    is_open
    || !(cuda::std::is_same_v<Attr, cudax::cufile_driver_attributes::max_device_cache_size_kb_t>
         || cuda::std::is_same_v<Attr, cudax::cufile_driver_attributes::max_device_pinned_mem_size_kb_t>);

  if (expect_success)
  {
    CHECK_NOTHROW((void) cudax::cufile_driver.attribute_range(attr));
  }
  else
  {
    CHECK_THROWS_AS((void) cudax::cufile_driver.attribute_range(attr), std::runtime_error);
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)
}

void test_attribute_ranges(const bool is_open)
{
  using namespace cudax::cufile_driver_attributes;

  // CUFileSizeTConfigParameter_t
  test_attribute_range(max_io_queue_depth, is_open);
  test_attribute_range(max_io_threads, is_open);
  test_attribute_range(min_io_threshold_size_kb, is_open);
  test_attribute_range(max_request_parallelism, is_open);
  test_attribute_range(max_direct_io_size_kb, is_open);
  test_attribute_range(max_device_cache_size_kb, is_open);
  test_attribute_range(per_buffer_cache_size_kb, is_open);
  test_attribute_range(max_device_pinned_mem_size_kb, is_open);
  test_attribute_range(io_batchsize, is_open);
  test_attribute_range(pollthreshold_size_kb, is_open);
  test_attribute_range(batch_io_timeout_ms, is_open);

  // CUFileBoolConfigParameter_t are unsupported
  // CUfileDriverStatusFlags are unsupported
  // CUfileFeatureFlags are unsupported
}

template <class Attr>
void test_get_attribute(const Attr& attr, const bool is_open)
{
  using ValueType = typename Attr::type;

  STATIC_REQUIRE(!noexcept(cudax::cufile_driver.attribute(attr)));
  STATIC_REQUIRE(cuda::std::is_same_v<ValueType, decltype(cudax::cufile_driver.attribute(attr))>);

  if (is_open || Attr::__can_be_get_when_closed)
  {
    CHECK_NOTHROW((void) cudax::cufile_driver.attribute(attr));
  }
  else
  {
    CHECK_THROWS_AS((void) cudax::cufile_driver.attribute(attr), std::runtime_error);
  }
}

void test_get_attributes(const bool is_open)
{
  using namespace cudax::cufile_driver_attributes;

  // CUFileSizeTConfigParameter_t
  test_get_attribute(max_io_queue_depth, is_open);
  test_get_attribute(max_io_threads, is_open);
  test_get_attribute(min_io_threshold_size_kb, is_open);
  test_get_attribute(max_request_parallelism, is_open);
  test_get_attribute(max_direct_io_size_kb, is_open);
  test_get_attribute(max_device_cache_size_kb, is_open);
  test_get_attribute(per_buffer_cache_size_kb, is_open);
  test_get_attribute(max_device_pinned_mem_size_kb, is_open);
  test_get_attribute(io_batchsize, is_open);
  test_get_attribute(pollthreshold_size_kb, is_open);
  test_get_attribute(batch_io_timeout_ms, is_open);

  // CUFileBoolConfigParameter_t
  test_get_attribute(use_poll_mode, is_open);
  test_get_attribute(allow_compat_mode, is_open);
  test_get_attribute(force_compat_mode, is_open);
  test_get_attribute(fs_misc_api_check_aggressive, is_open);
  test_get_attribute(parallel_io, is_open);
  // test_get_attribute(profile_nvtx is_open);
  test_get_attribute(allow_system_memory, is_open);
  test_get_attribute(use_pcip2pdma, is_open);
  test_get_attribute(prefer_io_uring, is_open);
  test_get_attribute(force_odirect_mode, is_open);
  test_get_attribute(skip_topology_detection, is_open);
  test_get_attribute(stream_memops_bypass, is_open);

  // CUfileDriverStatusFlags
  test_get_attribute(has_luster_support, is_open);
  test_get_attribute(has_wekafs_support, is_open);
  test_get_attribute(has_nfs_support, is_open);
  test_get_attribute(has_gpfs_support, is_open);
  test_get_attribute(has_nvme_support, is_open);
  test_get_attribute(has_nvmeof_support, is_open);
  test_get_attribute(has_scsi_support, is_open);
  test_get_attribute(has_scaleflux_csd_support, is_open);
  test_get_attribute(has_nvmesh_support, is_open);
  test_get_attribute(has_beegfs_support, is_open);
  test_get_attribute(has_nvme_p2p_support, is_open);
  test_get_attribute(has_scatefs_support, is_open);

  // CUfileFeatureFlags
  test_get_attribute(has_dynamic_routing_support, is_open);
  test_get_attribute(has_batch_io_support, is_open);
  test_get_attribute(has_streams_support, is_open);
  test_get_attribute(has_parallel_io_support, is_open);
}

template <class Attr>
void test_set_attribute(const Attr& attr, const bool is_open)
{
  using ValueType = typename Attr::type;

  STATIC_REQUIRE(!noexcept(cudax::cufile_driver.set_attribute(attr, cuda::std::declval<ValueType>())));
  STATIC_REQUIRE(
    cuda::std::is_same_v<void, decltype(cudax::cufile_driver.set_attribute(attr, cuda::std::declval<ValueType>()))>);

  const bool expect_success = is_open == Attr::__can_be_set_when_opened || !is_open == Attr::__can_be_set_when_closed;

  ValueType value{};
  if (expect_success)
  {
    CHECK_NOTHROW(value = cudax::cufile_driver.attribute(attr));
  }

  if (expect_success)
  {
    CHECK_NOTHROW(cudax::cufile_driver.set_attribute(attr, value));
  }
  else
  {
    CHECK_THROWS_AS(cudax::cufile_driver.set_attribute(attr, value), std::runtime_error);
  }
}

void test_set_attributes(const bool is_open)
{
  using namespace cudax::cufile_driver_attributes;

  // CUFileSizeTConfigParameter_t
  test_get_attribute(max_io_queue_depth, is_open);
  test_get_attribute(max_io_threads, is_open);
  test_get_attribute(min_io_threshold_size_kb, is_open);
  test_get_attribute(max_request_parallelism, is_open);
  test_get_attribute(max_direct_io_size_kb, is_open);
  test_get_attribute(max_device_cache_size_kb, is_open);
  test_get_attribute(per_buffer_cache_size_kb, is_open);
  test_get_attribute(max_device_pinned_mem_size_kb, is_open);
  test_get_attribute(io_batchsize, is_open);
  test_get_attribute(pollthreshold_size_kb, is_open);
  test_get_attribute(batch_io_timeout_ms, is_open);

  // CUFileBoolConfigParameter_t
  test_get_attribute(use_poll_mode, is_open);
  test_get_attribute(allow_compat_mode, is_open);
  test_get_attribute(force_compat_mode, is_open);
  test_get_attribute(fs_misc_api_check_aggressive, is_open);
  test_get_attribute(parallel_io, is_open);
  // test_get_attribute(profile_nvtx is_open);
  test_get_attribute(allow_system_memory, is_open);
  test_get_attribute(use_pcip2pdma, is_open);
  test_get_attribute(prefer_io_uring, is_open);
  test_get_attribute(force_odirect_mode, is_open);
  test_get_attribute(skip_topology_detection, is_open);
  test_get_attribute(stream_memops_bypass, is_open);

  // CUfileDriverStatusFlags are unsupported
  // CUfileFeatureFlags are unsupported
}

void test_open()
{
  STATIC_REQUIRE(!noexcept(cudax::cufile_driver.open()));
  STATIC_REQUIRE(cuda::std::is_same_v<void, decltype(cudax::cufile_driver.open())>);

  CHECK_NOTHROW(cudax::cufile_driver.open());
}

void test_close()
{
  STATIC_REQUIRE(!noexcept(cudax::cufile_driver.close()));
  STATIC_REQUIRE(cuda::std::is_same_v<void, decltype(cudax::cufile_driver.close())>);

  CHECK_NOTHROW(cudax::cufile_driver.close());
}

C2H_CCCLRT_TEST("cuFile driver", "[cufile][driver]")
{
  // 1. Test that the driver initial state is closed.
  test_is_open(false);

  // 2. Test that the attribute ranges can be queried when the driver is closed.
  test_attribute_ranges(false);

  // 3. Test getting attribute values when the driver is closed.
  test_get_attributes(false);

  // 4. Test setting attribute values when the driver is closed.
  test_set_attributes(false);

  // 5. Open the driver.
  test_open();

  // 6. Test that the driver is opened.
  test_is_open(true);

  // 7. Test that the attribute ranges can be queried when the driver is opened.
  test_attribute_ranges(true);

  // 8. Test getting attribute values when the driver is opened.
  test_get_attributes(true);

  // 9. Test setting attribute values when the driver is opened.
  test_set_attributes(true);

  // 10. Close the driver.
  test_close();

  // 11. Test that the driver is closed.
  test_is_open(false);

  // 12. Test that the attribute ranges can be queried when the driver after is closed.
  test_attribute_ranges(false);

  // 13. Test getting attribute values when the driver after is closed.
  test_get_attributes(false);

  // 14. Test setting attribute values when the driver after is closed.
  test_set_attributes(false);

  // 15. Reopen the driver.
  test_open();

  // 16. Test that the driver is opened.
  test_is_open(true);

  // 17. Reopen the driver for second time.
  test_open();

  // 18. Test that the driver is opened.
  test_is_open(true);

  // 19. Close the driver.
  test_close();

  // 20. Test that the driver is closed.
  test_is_open(false);

  // 21. Close the driver again.
  test_close();

  // 22. Test that the driver is closed.
  test_is_open(false);
}
