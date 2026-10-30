#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

// --- 8-byte tier: int8_t, N=8 ---

using Vec_i8_8 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<8>>;

__global__ void test_load_i8_8(const cuda::std::int8_t* in, cuda::std::int8_t* out)
{
  Vec_i8_8 v = simd::unchecked_load<Vec_i8_8>(in, Vec_i8_8::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i8_8::size(), simd::flag_aligned);
}

// --- 16-byte tier: int8_t, N=16 ---

using Vec_i8_16 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<16>>;

__global__ void test_load_i8_16(const cuda::std::int8_t* in, cuda::std::int8_t* out)
{
  Vec_i8_16 v = simd::unchecked_load<Vec_i8_16>(in, Vec_i8_16::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i8_16::size(), simd::flag_aligned);
}

// --- 32-byte tier: int8_t, N=32 ---

using Vec_i8_32 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<32>>;

__global__ void test_load_i8_32(const cuda::std::int8_t* in, cuda::std::int8_t* out)
{
  Vec_i8_32 v = simd::unchecked_load<Vec_i8_32>(in, Vec_i8_32::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i8_32::size(), simd::flag_aligned);
}

/*

; SMXXX-LABEL: .visible .entry {{.*}}test_load_i8_8{{.*}}(
; SMXXX: {{.*}}ld.global.{{([bus]64|v4\.[bus]16|v2\.[bus]32)}}{{.*}}
; SMXXX: {{.*}}st.global.{{([bus]64|v4\.[bus]16|v2\.[bus]32)}}{{.*}}

; SMXXX-LABEL: .visible .entry {{.*}}test_load_i8_16{{.*}}(
; SMXXX: {{.*}}ld.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}
; SMXXX: {{.*}}st.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}

; SM100-PLUS-LABEL: .visible .entry {{.*}}test_load_i8_32{{.*}}(
; SM100-PLUS: {{.*}}ld.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}
; SM100-PLUS: {{.*}}st.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}

*/
