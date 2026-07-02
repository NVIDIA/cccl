/**
 * @brief Unit test for float ↔ double emulated conversions (fp64emu ↔ float)
 *
 * Validates that fpemu library float-to-double and double-to-float conversions
 * produce bit-identical results to native casts for all value classes:
 *   - Normal values (positive/negative, small/large)
 *   - Subnormal doubles → zero as float
 *   - Subnormal floats → normal double
 *   - Inf, NaN, ±0
 *   - Rounding boundary cases (double → float is narrowing, requires round-to-nearest-even)
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

static uint64_t d_bits(double d)  { uint64_t b; memcpy(&b, &d, 8); return b; }
static uint32_t f_bits(float f)   { uint32_t b; memcpy(&b, &f, 4); return b; }
static double   from_d_bits(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static float    from_f_bits(uint32_t b) { float  f; memcpy(&f, &b, 4); return f; }

static bool is_nan_d(uint64_t b) { return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFull) != 0; }
static bool is_nan_f(uint32_t b) { return ((b >> 23) & 0xFF) == 0xFF && (b & 0x007FFFFFu) != 0; }

// When the packed API is routed through the unpacked cores (PACKED_VIA_UNPACKED=y), the
// unpack/pack round-trip canonicalizes NaNs: unpack collapses every NaN to a
// magic exponent, so pack rebuilds a canonical NaN and the original payload is
// not preserved. That payload loss is inherent to the unpacked representation,
// so a NaN-vs-NaN bit difference is reported as a WARNING (not a failure) in
// that mode only; the legacy bit-reinterpret path must stay strictly bit-exact.
#if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
static constexpr bool g_relax_nan_payload = true;
#else
static constexpr bool g_relax_nan_payload = false;
#endif

// ============================================================================
// Device kernels: float → fp64emu → double (widening, exact)
// ============================================================================

TARGET void kern_float_to_double(const float* in, double* emu, double* ref, int n) {
    for (int i = 0; i < n; i++) {
        fp64emu e(in[i]);
        emu[i] = (double)e;
        ref[i] = (double)in[i];
    }
}

// ============================================================================
// Device kernels: double → fp64emu → float (narrowing, requires rounding)
// ============================================================================

TARGET void kern_double_to_float(const double* in, float* emu, float* ref, int n) {
    for (int i = 0; i < n; i++) {
        fp64emu e(in[i]);
        emu[i] = (float)e;
        ref[i] = (float)in[i];
    }
}

// ============================================================================
// Generic test runner for float → double
// ============================================================================

static int test_float_to_double(const char* label, const float* vals, int n) {
    float*  d_in;
    double* d_emu;
    double* d_ref;

    MALLOC(d_in,  n * sizeof(float));
    MALLOC(d_emu, n * sizeof(double));
    MALLOC(d_ref, n * sizeof(double));

    memcpy(d_in, vals, n * sizeof(float));

    LAUNCH(kern_float_to_double, d_in, d_emu, d_ref, n);
    SYNC();

    int mismatches = 0;
    int warnings   = 0;
    for (int i = 0; i < n; i++) {
        const uint64_t be = d_bits(d_emu[i]);
        const uint64_t br = d_bits(d_ref[i]);
        if (be != br) {
            const bool nan_payload = g_relax_nan_payload && is_nan_d(be) && is_nan_d(br);
            if (nan_payload) {
                if (warnings < 5) {
                    printf("    WARN [%d]: float=0x%08x  emu_d=0x%016" PRIx64 "  ref_d=0x%016" PRIx64 "  (NaN payload, unpacked round-trip)\n",
                           i, f_bits(vals[i]), be, br);
                }
                warnings++;
            } else {
                if (mismatches < 5) {
                    printf("    MISMATCH [%d]: float=0x%08x  emu_d=0x%016" PRIx64 "  ref_d=0x%016" PRIx64 "\n",
                           i, f_bits(vals[i]), be, br);
                }
                mismatches++;
            }
        }
    }

    printf("  %-20s: %5d tested: %s", label, n,
           mismatches ? "FAIL" : (warnings ? "WARN" : "PASS"));
    if (mismatches) printf(" (%d mismatches)", mismatches);
    if (warnings)   printf(" (%d NaN-payload warnings)", warnings);
    printf("\n");

    FREE(d_in); FREE(d_emu); FREE(d_ref);
    return mismatches;
}

// ============================================================================
// Generic test runner for double → float
// ============================================================================

static int test_double_to_float(const char* label, const double* vals, int n) {
    double* d_in;
    float*  d_emu;
    float*  d_ref;

    MALLOC(d_in,  n * sizeof(double));
    MALLOC(d_emu, n * sizeof(float));
    MALLOC(d_ref, n * sizeof(float));

    memcpy(d_in, vals, n * sizeof(double));

    LAUNCH(kern_double_to_float, d_in, d_emu, d_ref, n);
    SYNC();

    int mismatches = 0;
    int warnings   = 0;
    for (int i = 0; i < n; i++) {
        const uint32_t be = f_bits(d_emu[i]);
        const uint32_t br = f_bits(d_ref[i]);
        if (be != br) {
            const bool nan_payload = g_relax_nan_payload && is_nan_f(be) && is_nan_f(br);
            if (nan_payload) {
                if (warnings < 5) {
                    printf("    WARN [%d]: double=0x%016" PRIx64 "  emu_f=0x%08x  ref_f=0x%08x  (NaN payload, unpacked round-trip)\n",
                           i, d_bits(vals[i]), be, br);
                }
                warnings++;
            } else {
                if (mismatches < 5) {
                    printf("    MISMATCH [%d]: double=0x%016" PRIx64 "  emu_f=0x%08x  ref_f=0x%08x\n",
                           i, d_bits(vals[i]), be, br);
                }
                mismatches++;
            }
        }
    }

    printf("  %-20s: %5d tested: %s", label, n,
           mismatches ? "FAIL" : (warnings ? "WARN" : "PASS"));
    if (mismatches) printf(" (%d mismatches)", mismatches);
    if (warnings)   printf(" (%d NaN-payload warnings)", warnings);
    printf("\n");

    FREE(d_in); FREE(d_emu); FREE(d_ref);
    return mismatches;
}

// ============================================================================
// Test data: float → double (widening, exact)
// ============================================================================

static int run_f2d_tests() {
    printf("\n  --- float → double (widening, exact) ---\n");
    int errors = 0;

    // Boundary values
    {
        float vals[] = {
            0.0f, -0.0f,
            1.0f, -1.0f,
            0.5f, -0.5f,
            2.0f, -2.0f,
            HUGE_VALF, -HUGE_VALF,
            from_f_bits(0x7FC00000u),           // +qNaN
            from_f_bits(0xFFC00000u),           // -qNaN
            from_f_bits(0x7F800001u),           // +sNaN
            from_f_bits(0x7FC0DEADu),           // qNaN with payload
            from_f_bits(0x00800000u),           // min normal float
            from_f_bits(0x7F7FFFFFu),           // max finite float
            from_f_bits(0xFF7FFFFFu),           // -max finite float
            from_f_bits(0x00000001u),           // min positive subnormal
            from_f_bits(0x80000001u),           // -min positive subnormal
            from_f_bits(0x007FFFFFu),           // max subnormal
            from_f_bits(0x00400000u),           // subnormal with 1 bit
            from_f_bits(0x00000100u),           // small subnormal
            3.14159265f,
            1.0e38f, -1.0e38f,
            1.0e-38f, -1.0e-38f,
            1.0e-45f,                           // near min subnormal
        };
        errors += test_float_to_double("boundary",
            vals, (int)(sizeof(vals)/sizeof(vals[0])));
    }

    // Random normal floats
    {
        constexpr int N = 65536;
        float* vals = new float[N];
        std::mt19937 gen(42);
        std::uniform_int_distribution<uint32_t> dist(0x00800000u, 0x7F7FFFFFu);
        for (int i = 0; i < N; i++) {
            uint32_t bits = dist(gen);
            if (gen() & 1) bits |= 0x80000000u;
            vals[i] = from_f_bits(bits);
        }
        errors += test_float_to_double("random normal", vals, N);
        delete[] vals;
    }

    // Random subnormal floats
    {
        constexpr int N = 65536;
        float* vals = new float[N];
        std::mt19937 gen(123);
        std::uniform_int_distribution<uint32_t> dist(1u, 0x007FFFFFu);
        for (int i = 0; i < N; i++) {
            uint32_t bits = dist(gen);
            if (gen() & 1) bits |= 0x80000000u;
            vals[i] = from_f_bits(bits);
        }
        errors += test_float_to_double("random subnormal", vals, N);
        delete[] vals;
    }

    return errors;
}

// ============================================================================
// Test data: double → float (narrowing, round-to-nearest-even)
// ============================================================================

static int run_d2f_tests() {
    printf("\n  --- double → float (narrowing, rounding) ---\n");
    int errors = 0;

    // Boundary values
    {
        double vals[] = {
            0.0, -0.0,
            1.0, -1.0,
            0.5, -0.5,
            2.0, -2.0,
            HUGE_VAL, -HUGE_VAL,
            from_d_bits(0x7FF8000000000000ULL),  // +qNaN
            from_d_bits(0xFFF8000000000000ULL),  // -qNaN
            from_d_bits(0x7FF0000000000001ULL),  // +sNaN
            from_d_bits(0x7FF80000DEADBEEFull),  // qNaN with payload
            // Max finite float as double
            (double)from_f_bits(0x7F7FFFFFu),
            -(double)from_f_bits(0x7F7FFFFFu),
            // Min normal float as double
            (double)from_f_bits(0x00800000u),
            // Just below min normal float → subnormal
            from_d_bits(0x3800000000000000ULL),  // 2^(-127)
            from_d_bits(0x3690000000000000ULL),  // 2^(-150)
            // Float subnormal range
            from_d_bits(0x36A0000000000000ULL),  // 2^(-149) = min subnormal float
            from_d_bits(0x380FFFFFFFFFE000ULL),  // max subnormal float
            // Overflow to float Inf
            1.0e39, -1.0e39,
            3.5e38, -3.5e38,
            // Near-zero underflow
            1.0e-300, -1.0e-300,
            5.0e-324,
            // Exact conversions
            0.25, 0.125, 0.0625,
            100.0, -100.0,
            3.14159265358979,
        };
        errors += test_double_to_float("boundary",
            vals, (int)(sizeof(vals)/sizeof(vals[0])));
    }

    // Random normal doubles in float range
    {
        constexpr int N = 65536;
        double* vals = new double[N];
        std::mt19937 gen(99);
        std::normal_distribution<double> dist(0.0, 1.0e10);
        for (int i = 0; i < N; i++)
            vals[i] = dist(gen);
        errors += test_double_to_float("random normal", vals, N);
        delete[] vals;
    }

    // Random doubles that produce subnormal floats (exponent near -126)
    {
        constexpr int N = 65536;
        double* vals = new double[N];
        std::mt19937_64 gen(777);
        // Range: [2^(-149), 2^(-126)) = subnormal float region
        for (int i = 0; i < N; i++) {
            uint64_t r = gen();
            int32_t exp_d = 874 + (int32_t)(r % 23);  // biased double exp for subnormal float output
            uint64_t frac = gen() & 0x000FFFFFFFFFFFFFull;
            uint64_t sign = (gen() & 1) ? (1ULL << 63) : 0;
            vals[i] = from_d_bits(sign | ((uint64_t)exp_d << 52) | frac);
        }
        errors += test_double_to_float("subnormal output", vals, N);
        delete[] vals;
    }

    // Rounding tie cases: doubles where the 29 dropped bits = 0x10000000 (half)
    {
        constexpr int N = 8192;
        double* vals = new double[N];
        std::mt19937_64 gen(555);
        for (int i = 0; i < N; i++) {
            uint64_t r = gen();
            int32_t exp_d = 897 + (int32_t)(r % 200);
            uint32_t upper23 = (uint32_t)(r >> 32) & 0x7FFFFF;
            // Set exactly the halfway point: bit 28 = 1, bits 27:0 = 0
            uint64_t frac = ((uint64_t)upper23 << 29) | (1ULL << 28);
            uint64_t sign = (gen() & 1) ? (1ULL << 63) : 0;
            vals[i] = from_d_bits(sign | ((uint64_t)exp_d << 52) | frac);
        }
        errors += test_double_to_float("rounding ties", vals, N);
        delete[] vals;
    }

    // Overflow region: doubles just around float max
    {
        constexpr int N = 4096;
        double* vals = new double[N];
        std::mt19937_64 gen(333);
        for (int i = 0; i < N; i++) {
            uint64_t r = gen();
            int32_t exp_d = 1149 + (int32_t)(r % 10);  // near float overflow
            uint64_t frac = gen() & 0x000FFFFFFFFFFFFFull;
            uint64_t sign = (gen() & 1) ? (1ULL << 63) : 0;
            vals[i] = from_d_bits(sign | ((uint64_t)exp_d << 52) | frac);
        }
        errors += test_double_to_float("overflow region", vals, N);
        delete[] vals;
    }

    return errors;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("  ** fpemu float ↔ double conversion unit test\n");
    printf("  ** Verifying bit-exact match against native casts\n");

    int errors = 0;

    errors += run_f2d_tests();
    errors += run_d2f_tests();

    printf("\nfpemu_float: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
