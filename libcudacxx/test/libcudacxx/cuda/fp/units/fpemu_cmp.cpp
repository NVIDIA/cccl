/**
 * @brief Unit test for fp64emu comparison operations
 *
 * Validates that the fpemu comparison primitives produce results identical to
 * native IEEE-754 binary64 comparisons for every value class:
 *   - Normal values (positive/negative, large/small)
 *   - Subnormals, ±0, ±inf
 *   - Quiet and signaling NaN (unordered: ==,<,<=,>,>= are false; != is true)
 *
 * Three surfaces are exercised and cross-checked against native `double`:
 *   1. C builtins        : __nv_fp64emu_cmp_{eq,ne,lt,le,gt,ge}
 *   2. C++ packed ops    : fp64emu {==,!=,<,<=,>,>=}
 *   3. C++ unpacked ops  : fp64emu_unpacked {==,!=,<,<=,>,>=}  (when available)
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
#else
    #define MALLOC(x,s) x = (decltype(x))malloc(s)
    #define FREE(x) free(x)
    #define LAUNCH(fn,...) fn(__VA_ARGS__)
    #define SYNC()
    #define TARGET
#endif

// Comparison operation indices (also used as bit positions in result codes).
enum cmp_op { OP_EQ = 0, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE, OP_COUNT };
static const char* op_name[OP_COUNT] = { "==", "!=", "<", "<=", ">", ">=" };

static double   from_d_bits(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static uint64_t d_bits(double d)        { uint64_t b; memcpy(&b, &d, 8); return b; }

// Print the device (GPU) or host (CPU) on which the test executes.
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
// Native reference: pack the six comparison results into one bit code.
// ============================================================================

static uint32_t native_codes(double x, double y)
{
    uint32_t c = 0;
    c |= (uint32_t)(x == y) << OP_EQ;
    c |= (uint32_t)(x != y) << OP_NE;
    c |= (uint32_t)(x <  y) << OP_LT;
    c |= (uint32_t)(x <= y) << OP_LE;
    c |= (uint32_t)(x >  y) << OP_GT;
    c |= (uint32_t)(x >= y) << OP_GE;
    return c;
}

// ============================================================================
// Device kernels: compute comparison codes for each (x[i], y[i]) pair.
// ============================================================================

TARGET void kern_builtin(const double* x, const double* y, uint32_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fpbits64_t ex = __nv_fp64emu_from_double(x[i]);
        fpbits64_t ey = __nv_fp64emu_from_double(y[i]);
        uint32_t c = 0;
        c |= (uint32_t)__nv_fp64emu_cmp_eq(ex, ey) << OP_EQ;
        c |= (uint32_t)__nv_fp64emu_cmp_ne(ex, ey) << OP_NE;
        c |= (uint32_t)__nv_fp64emu_cmp_lt(ex, ey) << OP_LT;
        c |= (uint32_t)__nv_fp64emu_cmp_le(ex, ey) << OP_LE;
        c |= (uint32_t)__nv_fp64emu_cmp_gt(ex, ey) << OP_GT;
        c |= (uint32_t)__nv_fp64emu_cmp_ge(ex, ey) << OP_GE;
        out[i] = c;
    }
}

TARGET void kern_packed_op(const double* x, const double* y, uint32_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fp64emu ex = x[i];
        fp64emu ey = y[i];
        uint32_t c = 0;
        c |= (uint32_t)(ex == ey) << OP_EQ;
        c |= (uint32_t)(ex != ey) << OP_NE;
        c |= (uint32_t)(ex <  ey) << OP_LT;
        c |= (uint32_t)(ex <= ey) << OP_LE;
        c |= (uint32_t)(ex >  ey) << OP_GT;
        c |= (uint32_t)(ex >= ey) << OP_GE;
        out[i] = c;
    }
}

#if __FPEMU_UNPACKED__ == 1
TARGET void kern_unpacked_op(const double* x, const double* y, uint32_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        fp64emu_unpacked ex = (fp64emu_unpacked)x[i];
        fp64emu_unpacked ey = (fp64emu_unpacked)y[i];
        uint32_t c = 0;
        c |= (uint32_t)(ex == ey) << OP_EQ;
        c |= (uint32_t)(ex != ey) << OP_NE;
        c |= (uint32_t)(ex <  ey) << OP_LT;
        c |= (uint32_t)(ex <= ey) << OP_LE;
        c |= (uint32_t)(ex >  ey) << OP_GT;
        c |= (uint32_t)(ex >= ey) << OP_GE;
        out[i] = c;
    }
}
#endif

// ============================================================================
// Host launchers: wrap each kernel so it can be passed as a plain function
// pointer (a __global__ kernel cannot be launched through a function pointer).
// ============================================================================

typedef void (*launch_fn)(const double*, const double*, uint32_t*, int);

static void launch_builtin(const double* x, const double* y, uint32_t* o, int n)
{ LAUNCH(kern_builtin, x, y, o, n); SYNC(); }

static void launch_packed_op(const double* x, const double* y, uint32_t* o, int n)
{ LAUNCH(kern_packed_op, x, y, o, n); SYNC(); }

#if __FPEMU_UNPACKED__ == 1
static void launch_unpacked_op(const double* x, const double* y, uint32_t* o, int n)
{ LAUNCH(kern_unpacked_op, x, y, o, n); SYNC(); }
#endif

// ============================================================================
// Generic runner: launch a kernel and diff every op against native.
// ============================================================================

static int run_surface(const char* label, launch_fn launch_helper,
                       const double* xs, const double* ys, int n)
{
    double*   d_x;
    double*   d_y;
    uint32_t* d_out;

    MALLOC(d_x,   n * sizeof(double));
    MALLOC(d_y,   n * sizeof(double));
    MALLOC(d_out, n * sizeof(uint32_t));

    memcpy(d_x, xs, n * sizeof(double));
    memcpy(d_y, ys, n * sizeof(double));

    launch_helper(d_x, d_y, d_out, n);

    int per_op[OP_COUNT] = {0};
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        uint32_t ref = native_codes(xs[i], ys[i]);
        uint32_t got = d_out[i];
        if (ref == got) continue;
        for (int op = 0; op < OP_COUNT; op++) {
            bool rb = (ref >> op) & 1;
            bool gb = (got >> op) & 1;
            if (rb != gb) {
                per_op[op]++;
                if (mismatches < 8) {
                    printf("    MISMATCH %-2s : x=0x%016" PRIx64 " y=0x%016" PRIx64
                           "  emu=%d native=%d\n",
                           op_name[op], d_bits(xs[i]), d_bits(ys[i]), gb, rb);
                }
                mismatches++;
            }
        }
    }

    printf("  %-22s: %6d pairs: %s", label, n, mismatches ? "FAIL" : "PASS");
    if (mismatches) {
        printf("  [");
        for (int op = 0; op < OP_COUNT; op++)
            if (per_op[op]) printf(" %s=%d", op_name[op], per_op[op]);
        printf(" ]");
    }
    printf("\n");

    FREE(d_x); FREE(d_y); FREE(d_out);
    return mismatches;
}

// ============================================================================
// Test data
// ============================================================================

static const double g_special[] = {
    0.0, -0.0,
    1.0, -1.0,
    2.0, -2.0,
    0.5, -0.5,
    3.14159265358979, -3.14159265358979,
    1.0e308, -1.0e308,
    1.0e-308, -1.0e-308,
    HUGE_VAL, -HUGE_VAL,                       // +inf, -inf
    from_d_bits(0x0010000000000000ULL),        // min normal
    from_d_bits(0x8010000000000000ULL),        // -min normal
    from_d_bits(0x000FFFFFFFFFFFFFULL),         // max subnormal
    from_d_bits(0x0000000000000001ULL),         // min subnormal
    from_d_bits(0x8000000000000001ULL),         // -min subnormal
    from_d_bits(0x7FEFFFFFFFFFFFFFULL),         // max finite
    from_d_bits(0xFFEFFFFFFFFFFFFFULL),         // -max finite
    from_d_bits(0x7FF8000000000000ULL),         // +qNaN
    from_d_bits(0xFFF8000000000000ULL),         // -qNaN
    from_d_bits(0x7FF0000000000001ULL),         // +sNaN
    from_d_bits(0xFFF0000000000001ULL),         // -sNaN
    from_d_bits(0x7FF80000DEADBEEFULL),         // qNaN with payload
};
static const int g_special_n = (int)(sizeof(g_special)/sizeof(g_special[0]));

// All ordered pairs of the special values (covers NaN/inf/zero corners).
static int run_special_pairs(int (*runner)(const char*, launch_fn, const double*, const double*, int),
                             const char* label, launch_fn k)
{
    const int n = g_special_n * g_special_n;
    double* xs = new double[n];
    double* ys = new double[n];
    int idx = 0;
    for (int i = 0; i < g_special_n; i++)
        for (int j = 0; j < g_special_n; j++) {
            xs[idx] = g_special[i];
            ys[idx] = g_special[j];
            idx++;
        }
    int e = runner(label, k, xs, ys, n);
    delete[] xs; delete[] ys;
    return e;
}

// Random finite + occasional NaN/inf pairs.
static int run_random_pairs(int (*runner)(const char*, launch_fn, const double*, const double*, int),
                            const char* label, launch_fn k, unsigned seed)
{
    constexpr int N = 65536;
    double* xs = new double[N];
    double* ys = new double[N];
    std::mt19937_64 gen(seed);

    auto gen_one = [&]() -> double {
        uint64_t r = gen();
        // ~6% of the time emit a special bit pattern to stress unordered paths.
        switch ((r >> 60) & 0xF) {
            case 0:  return g_special[gen() % g_special_n];
            default: break;
        }
        return from_d_bits(r);
    };

    for (int i = 0; i < N; i++) {
        xs[i] = gen_one();
        // Half the pairs share a magnitude so equality / ordering ties appear.
        if ((gen() & 3) == 0) ys[i] = xs[i];
        else                  ys[i] = gen_one();
    }
    int e = runner(label, k, xs, ys, N);
    delete[] xs; delete[] ys;
    return e;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    printf("  ** fpemu comparison unit test\n");
    printf("  ** Verifying ==, !=, <, <=, >, >= against native IEEE-754 double\n");
    print_target();

    int errors = 0;

    printf("\n  --- C builtins (__nv_fp64emu_cmp_*) ---\n");
    errors += run_special_pairs(run_surface, "special pairs", launch_builtin);
    errors += run_random_pairs (run_surface, "random pairs",  launch_builtin, 0xC0FFEE);

    printf("\n  --- C++ packed operators (fp64emu) ---\n");
    errors += run_special_pairs(run_surface, "special pairs", launch_packed_op);
    errors += run_random_pairs (run_surface, "random pairs",  launch_packed_op, 0xBADF00D);

#if __FPEMU_UNPACKED__ == 1
    printf("\n  --- C++ unpacked operators (fp64emu_unpacked) ---\n");
    errors += run_special_pairs(run_surface, "special pairs", launch_unpacked_op);
    errors += run_random_pairs (run_surface, "random pairs",  launch_unpacked_op, 0x1234567);
#endif

    printf("\nfpemu_cmp: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
