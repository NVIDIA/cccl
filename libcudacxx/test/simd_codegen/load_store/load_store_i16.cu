#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

// --- 8-byte tier: int16_t, N=4 ---

using Vec_i16_4 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<4>>;

__global__ void test_load_i16_4(const cuda::std::int16_t* in, cuda::std::int16_t* out)
{
  Vec_i16_4 v = simd::unchecked_load<Vec_i16_4>(in, Vec_i16_4::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i16_4::size(), simd::flag_aligned);
}

// --- 16-byte tier: int16_t, N=8 ---

using Vec_i16_8 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<8>>;

__global__ void test_load_i16_8(const cuda::std::int16_t* in, cuda::std::int16_t* out)
{
  Vec_i16_8 v = simd::unchecked_load<Vec_i16_8>(in, Vec_i16_8::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i16_8::size(), simd::flag_aligned);
}

// --- 32-byte tier: int16_t, N=16 ---

using Vec_i16_16 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<16>>;

__global__ void test_load_i16_16(const cuda::std::int16_t* in, cuda::std::int16_t* out)
{
  Vec_i16_16 v = simd::unchecked_load<Vec_i16_16>(in, Vec_i16_16::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i16_16::size(), simd::flag_aligned);
}

/*

; SMXXX-LABEL: .visible .entry {{.*}}test_load_i16_4{{.*}}(
; SMXXX: {{.*}}ld.global.{{([bus]64|v4\.[bus]16|v2\.[bus]32)}}{{.*}}
; SMXXX: {{.*}}st.global.{{([bus]64|v4\.[bus]16|v2\.[bus]32)}}{{.*}}

; SMXXX-LABEL: .visible .entry {{.*}}test_load_i16_8{{.*}}(
; SMXXX: {{.*}}ld.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}
; SMXXX: {{.*}}st.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}

; SM100-PLUS-LABEL: .visible .entry {{.*}}test_load_i16_16{{.*}}(
; SM100-PLUS: {{.*}}ld.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}
; SM100-PLUS: {{.*}}st.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}

*/
