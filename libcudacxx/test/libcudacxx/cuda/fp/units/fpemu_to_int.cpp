/**
 * @brief Unit test for fp64 -> integer emulated conversions
 *
 * Validates that the fpemu fp64->integer conversions reproduce, bit-for-bit,
 * the saturating semantics of the CUDA hardware rounding intrinsics:
 *
 *   signed   : NaN -> 0 ; +overflow -> INT_MAX  ; -overflow -> INT_MIN
 *   unsigned : NaN -> 0 ; +overflow -> UINT_MAX ; any negative -> 0
 *
 *
 * Four target types x four rounding modes (rn, rz, ru, rd) are covered for:
 *   1. C builtins        : __nv_fp64emu_to_{int,uint,ll,ull}_{rn,rz,ru,rd}
 *   2. C++ packed ops     : __double2{int,uint,ll,ull}_{rn,rz,ru,rd}(fp64emu)
 *   3. C++ packed casts   : (int32_t|uint32_t|int64_t|uint64_t)fp64emu      (rz)
 *   4. C++ unpacked casts : (int32_t|uint32_t|int64_t|uint64_t)unpacked       (rz)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cinttypes>
#include <random>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if __CUDACC__
    #define MALLOC(x,s) cudaMallocManaged(&x,s)
    #define FREE(x) cudaFree(x)
    #define LAUNCH(fn,...) fn<<<1,1>>>(__VA_ARGS__)
    #define SYNC() cudaDeviceSynchronize()
    #define TARGET __global__
    #define HD __host__ __device__
#else
    #define MALLOC(x,s) x = (decltype(x))malloc(s)
    #define FREE(x) free(x)
    #define LAUNCH(fn,...) fn(__VA_ARGS__)
    #define SYNC()
    #define TARGET
    #define HD
#endif

// Target type / rounding-mode indices. conv index = type*4 + mode.
enum { T_I32 = 0, T_U32, T_I64, T_U64, T_COUNT };
enum { M_RN = 0, M_RZ, M_RU, M_RD, M_COUNT };
static const int  NCONV = T_COUNT * M_COUNT;          // 16
static const char* type_name[T_COUNT] = { "i32", "u32", "i64", "u64" };
static const char* mode_name[M_COUNT] = { "rn", "rz", "ru", "rd" };

// Subset of conversions a surface can exercise (cast operators are rz only).
static const int RZ4[4] = { T_I32*4 + M_RZ, T_U32*4 + M_RZ, T_I64*4 + M_RZ, T_U64*4 + M_RZ };
static int ALL16[NCONV];

static double   from_d_bits(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static uint64_t d_bits(double d)        { uint64_t b; memcpy(&b, &d, 8); return b; }

// Width-preserving encode of an integer result into a uint64_t slot.
static HD uint64_t enc_i32(int32_t v)  { return (uint64_t)(uint32_t)v; }
static HD uint64_t enc_u32(uint32_t v) { return (uint64_t)v; }
static HD uint64_t enc_i64(int64_t v)  { return (uint64_t)v; }
static HD uint64_t enc_u64(uint64_t v) { return v; }

static void print_target()
{
#if __CUDACC__
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        printf("  ** Running on device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    } else {
        printf("  ** Running on device: <unknown CUDA device>\n");
    }
#else
    printf("  ** Running on host (CPU)\n");
#endif
}

// ============================================================================
// Reference: CUDA intrinsics on device, portable saturating math on host.
// ============================================================================

#if !defined(__CUDA_ARCH__)
// Round-half-to-even of an already-finite double.
static double ref_round_even(double d)
{
    double f    = floor(d);
    double diff = d - f;
    if (diff < 0.5) return f;
    if (diff > 0.5) return f + 1.0;
    double half = f * 0.5;            // tie: pick the even neighbour
    return (floor(half) == half) ? f : f + 1.0;
}
#endif

[[maybe_unused]] static HD uint64_t ref_one(double d, int type, int mode)
{
#if defined(__CUDA_ARCH__)
    switch (type * 4 + mode) {
        case T_I32*4 + M_RN: return enc_i32(__double2int_rn(d));
        case T_I32*4 + M_RZ: return enc_i32(__double2int_rz(d));
        case T_I32*4 + M_RU: return enc_i32(__double2int_ru(d));
        case T_I32*4 + M_RD: return enc_i32(__double2int_rd(d));
        case T_U32*4 + M_RN: return enc_u32(__double2uint_rn(d));
        case T_U32*4 + M_RZ: return enc_u32(__double2uint_rz(d));
        case T_U32*4 + M_RU: return enc_u32(__double2uint_ru(d));
        case T_U32*4 + M_RD: return enc_u32(__double2uint_rd(d));
        case T_I64*4 + M_RN: return enc_i64(__double2ll_rn(d));
        case T_I64*4 + M_RZ: return enc_i64(__double2ll_rz(d));
        case T_I64*4 + M_RU: return enc_i64(__double2ll_ru(d));
        case T_I64*4 + M_RD: return enc_i64(__double2ll_rd(d));
        case T_U64*4 + M_RN: return enc_u64(__double2ull_rn(d));
        case T_U64*4 + M_RZ: return enc_u64(__double2ull_rz(d));
        case T_U64*4 + M_RU: return enc_u64(__double2ull_ru(d));
        case T_U64*4 + M_RD: return enc_u64(__double2ull_rd(d));
    }
    return 0;
#else
    // NaN -> integer indefinite (sign bit only), per CUDA hardware.
    if (std::isnan(d))
        return (type <= T_U32) ? UINT64_C(0x0000000080000000)
                               : UINT64_C(0x8000000000000000);

    double r;
    switch (mode) {
        case M_RN: r = ref_round_even(d); break;
        case M_RZ: r = trunc(d);          break;
        case M_RU: r = ceil(d);           break;
        default:   r = floor(d);          break;   // M_RD
    }

    switch (type) {
        case T_I32:
            if (r >=  2147483648.0) return enc_i32(INT32_MAX);
            if (r <= -2147483648.0) return enc_i32(INT32_MIN);
            return enc_i32((int32_t)r);
        case T_U32:
            if (r <  0.0)           return enc_u32(0);
            if (r >= 4294967296.0)  return enc_u32(UINT32_MAX);
            return enc_u32((uint32_t)r);
        case T_I64:
            if (r >=  9223372036854775808.0) return enc_i64(INT64_MAX);
            if (r <= -9223372036854775808.0) return enc_i64(INT64_MIN);
            return enc_i64((int64_t)r);
        default: // T_U64
            if (r <  0.0)                     return enc_u64(0);
            if (r >= 18446744073709551616.0)  return enc_u64(UINT64_MAX);
            return enc_u64((uint64_t)r);
    }
#endif
}

// ============================================================================
// Kernels: each fills out[i*NCONV + conv] for the conversions it supports.
// ============================================================================

TARGET void kern_ref(const double* x, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++)
        for (int c = 0; c < NCONV; c++)
            out[i * NCONV + c] = ref_one(x[i], c >> 2, c & 3);
}

TARGET void kern_builtin(const double* x, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fpbits64_t e = __nv_fp64emu_from_double(x[i]);
        uint64_t* o  = out + i * NCONV;
        o[T_I32*4 + M_RN] = enc_i32(__nv_fp64emu_to_int_rn(e));
        o[T_I32*4 + M_RZ] = enc_i32(__nv_fp64emu_to_int_rz(e));
        o[T_I32*4 + M_RU] = enc_i32(__nv_fp64emu_to_int_ru(e));
        o[T_I32*4 + M_RD] = enc_i32(__nv_fp64emu_to_int_rd(e));
        o[T_U32*4 + M_RN] = enc_u32(__nv_fp64emu_to_uint_rn(e));
        o[T_U32*4 + M_RZ] = enc_u32(__nv_fp64emu_to_uint_rz(e));
        o[T_U32*4 + M_RU] = enc_u32(__nv_fp64emu_to_uint_ru(e));
        o[T_U32*4 + M_RD] = enc_u32(__nv_fp64emu_to_uint_rd(e));
        o[T_I64*4 + M_RN] = enc_i64(__nv_fp64emu_to_ll_rn(e));
        o[T_I64*4 + M_RZ] = enc_i64(__nv_fp64emu_to_ll_rz(e));
        o[T_I64*4 + M_RU] = enc_i64(__nv_fp64emu_to_ll_ru(e));
        o[T_I64*4 + M_RD] = enc_i64(__nv_fp64emu_to_ll_rd(e));
        o[T_U64*4 + M_RN] = enc_u64(__nv_fp64emu_to_ull_rn(e));
        o[T_U64*4 + M_RZ] = enc_u64(__nv_fp64emu_to_ull_rz(e));
        o[T_U64*4 + M_RU] = enc_u64(__nv_fp64emu_to_ull_ru(e));
        o[T_U64*4 + M_RD] = enc_u64(__nv_fp64emu_to_ull_rd(e));
    }
}

TARGET void kern_packed_named(const double* x, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fp64emu e = x[i];
        uint64_t* o = out + i * NCONV;
        o[T_I32*4 + M_RN] = enc_i32(__double2int_rn(e));
        o[T_I32*4 + M_RZ] = enc_i32(__double2int_rz(e));
        o[T_I32*4 + M_RU] = enc_i32(__double2int_ru(e));
        o[T_I32*4 + M_RD] = enc_i32(__double2int_rd(e));
        o[T_U32*4 + M_RN] = enc_u32(__double2uint_rn(e));
        o[T_U32*4 + M_RZ] = enc_u32(__double2uint_rz(e));
        o[T_U32*4 + M_RU] = enc_u32(__double2uint_ru(e));
        o[T_U32*4 + M_RD] = enc_u32(__double2uint_rd(e));
        o[T_I64*4 + M_RN] = enc_i64(__double2ll_rn(e));
        o[T_I64*4 + M_RZ] = enc_i64(__double2ll_rz(e));
        o[T_I64*4 + M_RU] = enc_i64(__double2ll_ru(e));
        o[T_I64*4 + M_RD] = enc_i64(__double2ll_rd(e));
        o[T_U64*4 + M_RN] = enc_u64(__double2ull_rn(e));
        o[T_U64*4 + M_RZ] = enc_u64(__double2ull_rz(e));
        o[T_U64*4 + M_RU] = enc_u64(__double2ull_ru(e));
        o[T_U64*4 + M_RD] = enc_u64(__double2ull_rd(e));
    }
}

TARGET void kern_packed_cast(const double* x, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fp64emu e = x[i];
        uint64_t* o = out + i * NCONV;
        o[T_I32*4 + M_RZ] = enc_i32((int32_t)e);
        o[T_U32*4 + M_RZ] = enc_u32((uint32_t)e);
        o[T_I64*4 + M_RZ] = enc_i64((int64_t)e);
        o[T_U64*4 + M_RZ] = enc_u64((uint64_t)e);
    }
}

#if __FPEMU_UNPACKED__ == 1
TARGET void kern_unpacked_cast(const double* x, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fp64emu_unpacked e = (fp64emu_unpacked)x[i];
        uint64_t* o = out + i * NCONV;
        o[T_I32*4 + M_RZ] = enc_i32((int32_t)e);
        o[T_U32*4 + M_RZ] = enc_u32((uint32_t)e);
        o[T_I64*4 + M_RZ] = enc_i64((int64_t)e);
        o[T_U64*4 + M_RZ] = enc_u64((uint64_t)e);
    }
}
#endif

// ============================================================================
// Generic runner: launch a surface kernel and the reference kernel, then diff
// the requested set of conversions.
// ============================================================================

template<typename KernelFn>
static int run_surface(const char* label, KernelFn kernel,
                       const int* convs, int nconv,
                       const double* xs, int n)
{
    double*   d_x;
    uint64_t* d_out;
    uint64_t* d_ref;
    MALLOC(d_x,   n * sizeof(double));
    MALLOC(d_out, n * NCONV * sizeof(uint64_t));
    MALLOC(d_ref, n * NCONV * sizeof(uint64_t));
    memcpy(d_x, xs, n * sizeof(double));

    LAUNCH(kern_ref, d_x, d_ref, n); SYNC();
    LAUNCH(kernel,   d_x, d_out, n); SYNC();

    int per_conv[NCONV] = {0};
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < nconv; k++) {
            int c = convs[k];
            uint64_t got = d_out[i * NCONV + c];
            uint64_t ref = d_ref[i * NCONV + c];
            if (got == ref) continue;
            per_conv[c]++;
            if (mismatches < 10) {
                printf("    MISMATCH %s_%s : x=0x%016" PRIx64
                       "  emu=0x%016" PRIx64 "  ref=0x%016" PRIx64 "\n",
                       type_name[c >> 2], mode_name[c & 3],
                       d_bits(xs[i]), got, ref);
            }
            mismatches++;
        }
    }

    printf("  %-22s: %6d vals x %2d conv: %s", label, n, nconv, mismatches ? "FAIL" : "PASS");
    if (mismatches) {
        printf("  [");
        for (int c = 0; c < NCONV; c++)
            if (per_conv[c]) printf(" %s_%s=%d", type_name[c >> 2], mode_name[c & 3], per_conv[c]);
        printf(" ]");
    }
    printf("\n");

    FREE(d_x); FREE(d_out); FREE(d_ref);
    return mismatches;
}

// ============================================================================
// Test data
// ============================================================================

static const double g_special[] = {
    0.0, -0.0,
    0.5, -0.5, 1.5, -1.5, 2.5, -2.5,
    0.49999999999999994, -0.49999999999999994,
    1.0, -1.0, 2.0, -2.0, 100.0, -100.0,
    3.14159265358979, -3.14159265358979,
    // exact powers / type boundaries
    2147483647.0, 2147483648.0, 2147483649.0,             // ~INT32_MAX
    -2147483648.0, -2147483649.0,                          // ~INT32_MIN
    4294967295.0, 4294967296.0, 4294967297.0,             // ~UINT32_MAX
    9223372036854775807.0, 9223372036854775808.0,         // ~INT64_MAX (2^63)
    -9223372036854775808.0, -9223372036854777856.0,        // ~INT64_MIN
    18446744073709551615.0, 18446744073709551616.0,       // ~UINT64_MAX (2^64)
    1e18, -1e18, 1e30, -1e30,
    from_d_bits(0x0000000000000001ULL),                    // min subnormal
    from_d_bits(0x8000000000000001ULL),                    // -min subnormal
    HUGE_VAL, -HUGE_VAL,                                    // +inf, -inf
    from_d_bits(0x7FF8000000000000ULL),                    // +qNaN
    from_d_bits(0xFFF8000000000000ULL),                    // -qNaN
    from_d_bits(0x7FF0000000000001ULL),                    // +sNaN
};
static const int g_special_n = (int)(sizeof(g_special)/sizeof(g_special[0]));

// Random doubles biased toward integer-conversion-interesting magnitudes.
static int fill_random(double* xs, int N, unsigned seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> small(-4.0, 4.0);
    std::uniform_real_distribution<double> med(-1.0e10, 1.0e10);
    std::uniform_real_distribution<double> big(-2.0e19, 2.0e19);
    for (int i = 0; i < N; i++) {
        switch (gen() % 6) {
            case 0:  xs[i] = g_special[gen() % g_special_n]; break;
            case 1:  xs[i] = small(gen); break;            // fractional / ties
            case 2:  xs[i] = med(gen);   break;            // 32-bit range
            case 3:  xs[i] = big(gen);   break;            // 64-bit range / overflow
            default: xs[i] = from_d_bits(gen()); break;    // full bit-pattern soup
        }
    }
    return N;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    for (int i = 0; i < NCONV; i++) ALL16[i] = i;

    printf("  ** fpemu fp64 -> integer conversion unit test\n");
    printf("  ** Verifying saturating conversions against CUDA rounding intrinsics\n");
    print_target();

    constexpr int NR = 200000;
    double* rnd = new double[NR];
    fill_random(rnd, NR, 0xC0FFEE);

    int errors = 0;

    printf("\n  --- C builtins (__nv_fp64emu_to_*_*) ---\n");
    errors += run_surface("special values", kern_builtin, ALL16, NCONV, g_special, g_special_n);
    errors += run_surface("random values",  kern_builtin, ALL16, NCONV, rnd, NR);

    printf("\n  --- C++ packed named ops (__double2*_*) ---\n");
    errors += run_surface("special values", kern_packed_named, ALL16, NCONV, g_special, g_special_n);
    errors += run_surface("random values",  kern_packed_named, ALL16, NCONV, rnd, NR);

    printf("\n  --- C++ packed cast operators (rz) ---\n");
    errors += run_surface("special values", kern_packed_cast, RZ4, 4, g_special, g_special_n);
    errors += run_surface("random values",  kern_packed_cast, RZ4, 4, rnd, NR);

#if __FPEMU_UNPACKED__ == 1
    printf("\n  --- C++ unpacked cast operators (rz) ---\n");
    errors += run_surface("special values", kern_unpacked_cast, RZ4, 4, g_special, g_special_n);
    errors += run_surface("random values",  kern_unpacked_cast, RZ4, 4, rnd, NR);
#endif

    delete[] rnd;

    printf("\nfpemu_to_int: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
