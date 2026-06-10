#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

// --- 16-byte tier: int64_t, N=2 ---
// 8-byte tier skipped: N=1

using Vec_i64_2 = simd::basic_vec<cuda::std::int64_t, simd::fixed_size<2>>;

__global__ void test_load_i64_2(const cuda::std::int64_t* in, cuda::std::int64_t* out)
{
  Vec_i64_2 v = simd::unchecked_load<Vec_i64_2>(in, Vec_i64_2::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i64_2::size(), simd::flag_aligned);
}

// --- 32-byte tier: int64_t, N=4 ---

using Vec_i64_4 = simd::basic_vec<cuda::std::int64_t, simd::fixed_size<4>>;

__global__ void test_load_i64_4(const cuda::std::int64_t* in, cuda::std::int64_t* out)
{
  Vec_i64_4 v = simd::unchecked_load<Vec_i64_4>(in, Vec_i64_4::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i64_4::size(), simd::flag_aligned);
}

/*

; SMXXX-LABEL: .visible .entry {{.*}}test_load_i64_2{{.*}}(
; SMXXX: {{.*}}ld.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}
; SMXXX: {{.*}}st.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}

; SM100-PLUS-LABEL: .visible .entry {{.*}}test_load_i64_4{{.*}}(
; SM100-PLUS: {{.*}}ld.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}
; SM100-PLUS: {{.*}}st.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}

*/
