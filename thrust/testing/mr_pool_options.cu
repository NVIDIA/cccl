#include <thrust/mr/pool_options.h>

#include <unittest/unittest.h>

void TestPoolOptionsBasicValidity()
{
  thrust::mr::pool_options options = thrust::mr::pool_options();
  REQUIRE_FALSE(options.validate());

  options.max_blocks_per_chunk = 1024;
  options.max_bytes_per_chunk  = 1024 * 1024;
  options.smallest_block_size  = 8;
  options.largest_block_size   = 1024;
  REQUIRE(options.validate());

  // the minimum number of blocks per chunk is bigger than the max
  options.min_blocks_per_chunk = 1025;
  REQUIRE_FALSE(options.validate());
  options.min_blocks_per_chunk = 128;
  REQUIRE(options.validate());

  // the minimum number of bytes per chunk is bigger than the max
  options.min_bytes_per_chunk = 1025 * 1024;
  REQUIRE_FALSE(options.validate());
  options.min_bytes_per_chunk = 1024;
  REQUIRE(options.validate());

  // smallest block size is bigger than the largest block size
  options.smallest_block_size = 2048;
  REQUIRE_FALSE(options.validate());
  options.smallest_block_size = 8;
  REQUIRE(options.validate());
}
TEST_CASE("TestPoolOptionsBasicValidity", "[mr_pool_options]")
{
  TestPoolOptionsBasicValidity();
}

void TestPoolOptionsComplexValidity()
{
  thrust::mr::pool_options options = thrust::mr::pool_options();
  REQUIRE_FALSE(options.validate());

  options.max_blocks_per_chunk = 1024;
  options.max_bytes_per_chunk  = 1024 * 1024;
  options.smallest_block_size  = 8;
  options.largest_block_size   = 1024;
  REQUIRE(options.validate());

  options.min_bytes_per_chunk = 2 * 1024;
  options.max_bytes_per_chunk = 256 * 1024;

  // the biggest allowed allocation (deduced from blocks in chunks)
  // is smaller than the minimal allowed one (defined in bytes)
  options.max_blocks_per_chunk = 1;
  REQUIRE_FALSE(options.validate());
  options.max_blocks_per_chunk = 1024;
  REQUIRE(options.validate());

  // the smallest allowed allocation (deduced from blocks in chunks)
  // is bigger than the maximum allowed one (defined in bytes)
  options.min_blocks_per_chunk = 1024 * 1024;
  REQUIRE_FALSE(options.validate());
  options.min_blocks_per_chunk = 128;
  REQUIRE(options.validate());
}
TEST_CASE("TestPoolOptionsComplexValidity", "[mr_pool_options]")
{
  TestPoolOptionsComplexValidity();
}
