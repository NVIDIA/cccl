#include <thrust/reduce.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestReduce(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector v{1, -2, 3};

  // no initializer
  REQUIRE(thrust::reduce(exec, v.begin(), v.end()) == 2);

  // with initializer
  REQUIRE(thrust::reduce(exec, v.begin(), v.end(), (T) 10) == 12);
}

TEST_CASE("TestReducePar", "[hpx]")
{
  TestReduce(thrust::hpx::par);
}

TEST_CASE("TestReduceParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::par.on(exec));
}

TEST_CASE("TestReduceParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestReduce(thrust::hpx::par.on(exec));
}

TEST_CASE("TestReduceParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::par.with(acs));
}

TEST_CASE("TestReduceParUnseq", "[hpx]")
{
  TestReduce(thrust::hpx::par_unseq);
}

TEST_CASE("TestReduceParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestReduceParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestReduce(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestReduceParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestReduceSeq", "[hpx]")
{
  TestReduce(thrust::hpx::seq);
}

TEST_CASE("TestReduceSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestReduceSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestReduce(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestReduceSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestReduceUnseq", "[hpx]")
{
  TestReduce(thrust::hpx::unseq);
}

TEST_CASE("TestReduceUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestReduceUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestReduce(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestReduceUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::unseq.with(acs));
}
