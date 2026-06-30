/**
 * @brief Unit test for fp64 emulated division
 *
 * Validates that the fpemu division reproduces, bit-for-bit, correctly-rounded
 * IEEE-754 binary64 division for all four rounding modes (rn, rz, ru, rd).
 *
 *
 * Surfaces covered:
 *   1. C builtins     : __nv_fp64emu_ddiv_{rn,rz,ru,rd}
 *   2. C++ packed op   : (fp64emu)a / (fp64emu)b           (rn)
 *   3. C++ unpacked op : (fp64emu_unpacked)a / (...)b        (rn)
 *
 * NaN results are compared by class (both NaN == match), since exact NaN
 * payloads are platform-defined; all finite/infinite results are compared
 * bit-exactly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cinttypes>
#include <random>
#if !defined(__CUDA_ARCH__)
    #include <cfenv>
#endif
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

enum { M_RN = 0, M_RZ, M_RU, M_RD, M_COUNT };
static const int   NCONV = M_COUNT;                  // 4 (one quotient per mode)
static const char* mode_name[M_COUNT] = { "rn", "rz", "ru", "rd" };

static const int RN1[1]  = { M_RN };
static int       ALL4[NCONV];

static HD double   from_d_bits(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static HD uint64_t d_bits(double d)        { uint64_t b; memcpy(&b, &d, 8); return b; }

static bool is_nan_bits(uint64_t b)
{
    return ((b & UINT64_C(0x7FF0000000000000)) == UINT64_C(0x7FF0000000000000))
        &&  (b & UINT64_C(0x000FFFFFFFFFFFFF));
}
// NaN payloads are platform-defined: treat any two NaNs as a match.
static bool match(uint64_t got, uint64_t ref)
{
    return (got == ref) || (is_nan_bits(got) && is_nan_bits(ref));
}

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
// Reference: CUDA __ddiv_* intrinsics on device, fenv-directed division on host
// ============================================================================

[[maybe_unused]] static HD uint64_t ref_one(double a, double b, int mode)
{
#if defined(__CUDA_ARCH__)
    double q;
    switch (mode) {
        case M_RZ: q = __ddiv_rz(a, b); break;
        case M_RU: q = __ddiv_ru(a, b); break;
        case M_RD: q = __ddiv_rd(a, b); break;
        default:   q = __ddiv_rn(a, b); break;   // M_RN
    }
    uint64_t r; memcpy(&r, &q, 8); return r;
#else
    int old = fegetround();
    int fe;
    switch (mode) {
        case M_RZ: fe = FE_TOWARDZERO; break;
        case M_RU: fe = FE_UPWARD;     break;
        case M_RD: fe = FE_DOWNWARD;   break;
        default:   fe = FE_TONEAREST;  break;   // M_RN
    }
    fesetround(fe);
    // volatile forces the divsd to execute (in memory) between the fesetround
    // calls and prevents compile-time constant folding under the wrong mode.
    volatile double va = a, vb = b;
    volatile double q  = va / vb;
    double r = q;
    fesetround(old);
    return d_bits(r);
#endif
}

// ============================================================================
// Kernels: each fills out[i*NCONV + mode] for the modes it supports.
// ============================================================================

TARGET void kern_ref(const double* x, const double* y, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++)
        for (int m = 0; m < NCONV; m++)
            out[i * NCONV + m] = ref_one(x[i], y[i], m);
}

TARGET void kern_builtin(const double* x, const double* y, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fpbits64_t a = __nv_fp64emu_from_double(x[i]);
        fpbits64_t b = __nv_fp64emu_from_double(y[i]);
        uint64_t*  o = out + i * NCONV;
        o[M_RN] = (uint64_t)__nv_fp64emu_ddiv_rn(a, b);
        o[M_RZ] = (uint64_t)__nv_fp64emu_ddiv_rz(a, b);
        o[M_RU] = (uint64_t)__nv_fp64emu_ddiv_ru(a, b);
        o[M_RD] = (uint64_t)__nv_fp64emu_ddiv_rd(a, b);
    }
}

TARGET void kern_packed_op(const double* x, const double* y, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fp64emu a = x[i];
        fp64emu b = y[i];
        out[i * NCONV + M_RN] = d_bits((double)(a / b));
    }
}

#if __FPEMU_UNPACKED__ == 1
TARGET void kern_unpacked_op(const double* x, const double* y, uint64_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fp64emu_unpacked a = (fp64emu_unpacked)x[i];
        fp64emu_unpacked b = (fp64emu_unpacked)y[i];
        out[i * NCONV + M_RN] = d_bits((double)(a / b));
    }
}
#endif

// ============================================================================
// Generic runner
// ============================================================================

template<typename KernelFn>
static int run_surface(const char* label, KernelFn kernel,
                       const int* modes, int nmode,
                       const double* xs, const double* ys, int n)
{
    double*   d_x;
    double*   d_y;
    uint64_t* d_out;
    uint64_t* d_ref;
    MALLOC(d_x,   n * sizeof(double));
    MALLOC(d_y,   n * sizeof(double));
    MALLOC(d_out, n * NCONV * sizeof(uint64_t));
    MALLOC(d_ref, n * NCONV * sizeof(uint64_t));
    memcpy(d_x, xs, n * sizeof(double));
    memcpy(d_y, ys, n * sizeof(double));

    LAUNCH(kern_ref, d_x, d_y, d_ref, n); SYNC();
    LAUNCH(kernel,   d_x, d_y, d_out, n); SYNC();

    int per_mode[NCONV] = {0};
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < nmode; k++) {
            int m = modes[k];
            uint64_t got = d_out[i * NCONV + m];
            uint64_t ref = d_ref[i * NCONV + m];
            if (match(got, ref)) continue;
            per_mode[m]++;
            if (mismatches < 10) {
                printf("    MISMATCH %s : x=0x%016" PRIx64 " y=0x%016" PRIx64
                       "  emu=0x%016" PRIx64 "  ref=0x%016" PRIx64 "\n",
                       mode_name[m], d_bits(xs[i]), d_bits(ys[i]), got, ref);
            }
            mismatches++;
        }
    }

    printf("  %-22s: %7d vals x %d mode: %s", label, n, nmode, mismatches ? "FAIL" : "PASS");
    if (mismatches) {
        printf("  [");
        for (int m = 0; m < NCONV; m++)
            if (per_mode[m]) printf(" %s=%d", mode_name[m], per_mode[m]);
        printf(" ]");
    }
    printf("\n");

    FREE(d_x); FREE(d_y); FREE(d_out); FREE(d_ref);
    return mismatches;
}

// ============================================================================
// Test data
// ============================================================================

static const double g_special[] = {
    0.0, -0.0,
    1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5,
    100.0, -100.0, 3.14159265358979, -3.14159265358979,
    1e-300, -1e-300, 1e300, -1e300,
    from_d_bits(0x0000000000000001ULL),   // min subnormal
    from_d_bits(0x800FFFFFFFFFFFFFULL),   // -max subnormal
    from_d_bits(0x0010000000000000ULL),   // min normal
    HUGE_VAL, -HUGE_VAL,                   // +inf, -inf
    from_d_bits(0x7FF8000000000000ULL),    // +qNaN
    from_d_bits(0xFFF8000000000000ULL),    // -qNaN
    from_d_bits(0x7FF0000000000001ULL),    // +sNaN
};
static const int g_special_n = (int)(sizeof(g_special)/sizeof(g_special[0]));

static int fill_special_pairs(double* xs, double* ys)
{
    int k = 0;
    for (int i = 0; i < g_special_n; i++)
        for (int j = 0; j < g_special_n; j++) {
            xs[k] = g_special[i];
            ys[k] = g_special[j];
            k++;
        }
    return k;
}

// Random doubles spanning the full exponent range (incl. subnormals).
static int fill_random(double* xs, double* ys, int N, unsigned seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> small(-4.0, 4.0);
    std::uniform_real_distribution<double> med(-1.0e150, 1.0e150);
    auto one = [&](int kind) -> double {
        switch (kind) {
            case 0:  return g_special[gen() % g_special_n];
            case 1:  return small(gen);
            case 2:  return med(gen);
            default: return from_d_bits(gen());   // full bit-pattern soup
        }
    };
    for (int i = 0; i < N; i++) {
        xs[i] = one((int)(gen() % 4));
        ys[i] = one((int)(gen() % 4));
    }
    return N;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    for (int i = 0; i < NCONV; i++) ALL4[i] = i;

    printf("  ** fpemu fp64 division unit test\n");
    printf("  ** Verifying correctly-rounded division against the hardware divider\n");
    print_target();

    const int NS = g_special_n * g_special_n;
    double* sx = new double[NS];
    double* sy = new double[NS];
    fill_special_pairs(sx, sy);

    constexpr int NR = 300000;
    double* rx = new double[NR];
    double* ry = new double[NR];
    fill_random(rx, ry, NR, 0xD1D1DEu);

    int errors = 0;

    printf("\n  --- C builtins (__nv_fp64emu_ddiv_*) ---\n");
    errors += run_surface("special pairs", kern_builtin, ALL4, NCONV, sx, sy, NS);
    errors += run_surface("random pairs",  kern_builtin, ALL4, NCONV, rx, ry, NR);

    printf("\n  --- C++ packed operator/ (rn) ---\n");
    errors += run_surface("special pairs", kern_packed_op, RN1, 1, sx, sy, NS);
    errors += run_surface("random pairs",  kern_packed_op, RN1, 1, rx, ry, NR);

#if __FPEMU_UNPACKED__ == 1
    printf("\n  --- C++ unpacked operator/ (rn) ---\n");
    errors += run_surface("special pairs", kern_unpacked_op, RN1, 1, sx, sy, NS);
    errors += run_surface("random pairs",  kern_unpacked_op, RN1, 1, rx, ry, NR);
#endif

    delete[] sx; delete[] sy; delete[] rx; delete[] ry;

    printf("\nfpemu_div: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
