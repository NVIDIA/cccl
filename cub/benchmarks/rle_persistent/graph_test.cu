#include <cstdio>
#include <random>
#include <vector>

#include "rle_dispatch.cuh"

// key/output types: one translation unit per instantiation (mirrors CUB's per-type bench TUs).
// KeyT needs only operator== with CUB's equality semantics (floats: NaN breaks runs; complex:
// componentwise). LenT is the run-length output type; run-length arithmetic is tile-local int,
// widened at the store (valid while the longest run < 2^31, i.e. num_items < 2^31).
#ifndef RLE_KEY_T
#  define RLE_KEY_T int
#endif
#ifndef RLE_LEN_T
#  define RLE_LEN_T int
#endif
#ifndef RLE_NUM_RUNS_T
#  define RLE_NUM_RUNS_T int
#endif
#ifndef RLE_OFFSET_T
#  define RLE_OFFSET_T int
#endif
#ifndef K_IPT
#  define K_IPT 0
#endif
using KeyT     = RLE_KEY_T;
using LenT     = RLE_LEN_T;
using NumRunsT = RLE_NUM_RUNS_T;
using OffT     = RLE_OFFSET_T;
static_assert(sizeof(OffT) == 4 || sizeof(OffT) == 8, "OffT: int32 or int64");
using RleConfigT        = rle_impl::winner_config<KeyT, K_IPT>;
using TilePartialStateT = rle_impl::TilePartialStateT;

constexpr int kTileSize = RleConfigT::kTileSize;

static inline cudaError_t persistent_rle_encode(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys,
  KeyT* d_unique,
  LenT* d_counts,
  NumRunsT* d_num_runs,
  OffT num_items,
  cudaStream_t stream = 0)
{
  return rle_impl::persistent_rle_encode<RleConfigT>(
    d_temp_storage, temp_storage_bytes, d_keys, d_unique, d_counts, d_num_runs, num_items, stream);
}

int main()
{
  const long long n = (1 << 22) + 12345;
  std::vector<KeyT> h((size_t) n);
  std::mt19937 rng(9);
  std::uniform_int_distribution<int> seg(1, 8), kd(0, 1000000);
  long long i           = 0;
  int prev              = -1;
  long long expect_runs = 0;
  while (i < n)
  {
    int v = kd(rng);
    if (v == prev)
    {
      v = (v + 1) % 1000001;
    }
    prev = v;
    ++expect_runs;
    long long e = std::min<long long>(i + seg(rng), n);
    for (; i < e; ++i)
    {
      h[(size_t) i] = KeyT(v);
    }
  }
  KeyT *dk, *du;
  LenT* dc;
  NumRunsT* dn;
  void* dtemp;
  size_t tempb = 0;
  cudaMalloc(&dk, sizeof(KeyT) * n);
  cudaMemcpy(dk, h.data(), sizeof(KeyT) * n, cudaMemcpyHostToDevice);
  cudaMalloc(&du, sizeof(KeyT) * n);
  cudaMalloc(&dc, sizeof(LenT) * n);
  cudaMalloc(&dn, sizeof(NumRunsT));
  persistent_rle_encode(nullptr, tempb, dk, du, dc, dn, (OffT) n);
  cudaMalloc(&dtemp, tempb);
  cudaMemset(dtemp, 0xAB, tempb);

  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaGraph_t graph;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  persistent_rle_encode(dtemp, tempb, dk, du, dc, dn, (OffT) n, s);
  cudaStreamEndCapture(s, &graph);
  cudaGraphExec_t exec;
  cudaError_t ge = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (ge != cudaSuccess)
  {
    std::printf("GRAPH INSTANTIATE FAILED: %s\n", cudaGetErrorString(ge));
    return 1;
  }
  int fails = 0;
  for (int rep = 0; rep < 4; ++rep)
  {
    cudaMemset(dn, 0xEE, sizeof(NumRunsT));
    cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);
    NumRunsT r = -1;
    cudaMemcpy(&r, dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost);
    const bool ok = ((long long) r == expect_runs);
    std::printf("replay %d: runs=%lld expect=%lld %s\n", rep, (long long) r, expect_runs, ok ? "OK" : "FAIL");
    fails += !ok;
  }
  std::printf(fails ? "GRAPH TEST FAILED\n" : "GRAPH TEST PASS\n");
  return fails ? 1 : 0;
}
