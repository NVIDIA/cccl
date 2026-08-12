#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>

namespace simd = cuda::std::simd;

// --- 8-byte tier: float, N=2 ---

using Vec_f32_2 = simd::basic_vec<float, simd::fixed_size<2>>;

__global__ void test_load_f32_2(const float* in, float* out)
{
  Vec_f32_2 v = simd::unchecked_load<Vec_f32_2>(in, Vec_f32_2::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f32_2::size(), simd::flag_aligned);
}

// --- 16-byte tier: float, N=4 ---

using Vec_f32_4 = simd::basic_vec<float, simd::fixed_size<4>>;

__global__ void test_load_f32_4(const float* in, float* out)
{
  Vec_f32_4 v = simd::unchecked_load<Vec_f32_4>(in, Vec_f32_4::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f32_4::size(), simd::flag_aligned);
}

// --- 32-byte tier: float, N=8 ---

using Vec_f32_8 = simd::basic_vec<float, simd::fixed_size<8>>;

__global__ void test_load_f32_8(const float* in, float* out)
{
  Vec_f32_8 v = simd::unchecked_load<Vec_f32_8>(in, Vec_f32_8::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f32_8::size(), simd::flag_aligned);
}

/*

; SMXXX-LABEL: .visible .entry {{.*}}test_load_f32_2{{.*}}(
; SMXXX: {{.*}}ld.global.{{([bus]64|v4\.[bus]16|v2\.[bfus]32)}}{{.*}}
; SMXXX: {{.*}}st.global.{{([bus]64|v4\.[bus]16|v2\.[bfus]32)}}{{.*}}

; SMXXX-LABEL: .visible .entry {{.*}}test_load_f32_4{{.*}}(
; SMXXX: {{.*}}ld.global.{{(v4\.[bfus]32|v2\.[bus]64)}}{{.*}}
; SMXXX: {{.*}}st.global.{{(v4\.[bfus]32|v2\.[bus]64)}}{{.*}}

; SM100-PLUS-LABEL: .visible .entry {{.*}}test_load_f32_8{{.*}}(
; SM100-PLUS: {{.*}}ld.global.{{(v4\.[bus]64|v8\.[bfus]32)}}{{.*}}
; SM100-PLUS: {{.*}}st.global.{{(v4\.[bus]64|v8\.[bfus]32)}}{{.*}}

*/
