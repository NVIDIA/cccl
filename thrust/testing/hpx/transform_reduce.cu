#include <thrust/transform_reduce.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestTransformReduce(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector v{1, -2, 3};

  auto square = [](T x) {
    return x * x;
  };

  REQUIRE(thrust::transform_reduce(exec, v.begin(), v.end(), square, (T) 0, thrust::plus<T>()) == 14);
}

TEST_CASE("TestTransformReducePar", "[hpx]")
{
  TestTransformReduce(thrust::hpx::par);
}

TEST_CASE("TestTransformReduceParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransformReduce(thrust::hpx::par.on(exec));
}

TEST_CASE("TestTransformReduceParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransformReduce(thrust::hpx::par.on(exec));
}

TEST_CASE("TestTransformReduceParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformReduce(thrust::hpx::par.with(acs));
}

TEST_CASE("TestTransformReduceParUnseq", "[hpx]")
{
  TestTransformReduce(thrust::hpx::par_unseq);
}

TEST_CASE("TestTransformReduceParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransformReduce(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestTransformReduceParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransformReduce(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestTransformReduceParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformReduce(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestTransformReduceSeq", "[hpx]")
{
  TestTransformReduce(thrust::hpx::seq);
}

TEST_CASE("TestTransformReduceSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransformReduce(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestTransformReduceSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransformReduce(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestTransformReduceSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformReduce(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestTransformReduceUnseq", "[hpx]")
{
  TestTransformReduce(thrust::hpx::unseq);
}

TEST_CASE("TestTransformReduceUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransformReduce(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestTransformReduceUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransformReduce(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestTransformReduceUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformReduce(thrust::hpx::unseq.with(acs));
}
