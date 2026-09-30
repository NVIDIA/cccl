#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>

namespace simd = cuda::std::simd;

// --- 16-byte tier: double, N=2 ---
// 8-byte tier skipped: N=1

using Vec_f64_2 = simd::basic_vec<double, simd::fixed_size<2>>;

__global__ void test_load_f64_2(const double* in, double* out)
{
  Vec_f64_2 v = simd::unchecked_load<Vec_f64_2>(in, Vec_f64_2::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f64_2::size(), simd::flag_aligned);
}

// --- 32-byte tier: double, N=4 ---

using Vec_f64_4 = simd::basic_vec<double, simd::fixed_size<4>>;

__global__ void test_load_f64_4(const double* in, double* out)
{
  Vec_f64_4 v = simd::unchecked_load<Vec_f64_4>(in, Vec_f64_4::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f64_4::size(), simd::flag_aligned);
}

/*

; SMXXX-LABEL: .visible .entry {{.*}}test_load_f64_2{{.*}}(
; SMXXX: {{.*}}ld.global.{{(v4\.[bus]32|v2\.[bfus]64)}}{{.*}}
; SMXXX: {{.*}}st.global.{{(v4\.[bus]32|v2\.[bfus]64)}}{{.*}}

; SM100-PLUS-LABEL: .visible .entry {{.*}}test_load_f64_4{{.*}}(
; SM100-PLUS: {{.*}}ld.global.{{(v4\.[bfus]64|v8\.[bus]32)}}{{.*}}
; SM100-PLUS: {{.*}}st.global.{{(v4\.[bfus]64|v8\.[bus]32)}}{{.*}}

*/
