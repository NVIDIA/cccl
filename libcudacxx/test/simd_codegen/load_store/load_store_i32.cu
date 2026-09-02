#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

// --- 8-byte tier: int32_t, N=2 ---

using Vec_i32_2 = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<2>>;

__global__ void test_load_i32_2(const cuda::std::int32_t* in, cuda::std::int32_t* out)
{
  Vec_i32_2 v = simd::unchecked_load<Vec_i32_2>(in, Vec_i32_2::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i32_2::size(), simd::flag_aligned);
}

// --- 16-byte tier: int32_t, N=4 ---

using Vec_i32_4 = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<4>>;

__global__ void test_load_i32_4(const cuda::std::int32_t* in, cuda::std::int32_t* out)
{
  Vec_i32_4 v = simd::unchecked_load<Vec_i32_4>(in, Vec_i32_4::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i32_4::size(), simd::flag_aligned);
}

// --- 32-byte tier: int32_t, N=8 ---

using Vec_i32_8 = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<8>>;

__global__ void test_load_i32_8(const cuda::std::int32_t* in, cuda::std::int32_t* out)
{
  Vec_i32_8 v = simd::unchecked_load<Vec_i32_8>(in, Vec_i32_8::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i32_8::size(), simd::flag_aligned);
}

/*

; SMXXX-LABEL: .visible .entry {{.*}}test_load_i32_2{{.*}}(
; SMXXX: {{.*}}ld.global.{{([bus]64|v4\.[bus]16|v2\.[bus]32)}}{{.*}}
; SMXXX: {{.*}}st.global.{{([bus]64|v4\.[bus]16|v2\.[bus]32)}}{{.*}}

; SMXXX-LABEL: .visible .entry {{.*}}test_load_i32_4{{.*}}(
; SMXXX: {{.*}}ld.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}
; SMXXX: {{.*}}st.global.{{(v4\.[bus]32|v2\.[bus]64)}}{{.*}}

; SM100-PLUS-LABEL: .visible .entry {{.*}}test_load_i32_8{{.*}}(
; SM100-PLUS: {{.*}}ld.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}
; SM100-PLUS: {{.*}}st.global.{{(v4\.[bus]64|v8\.[bus]32)}}{{.*}}

*/
