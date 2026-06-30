/**
 * @brief Unit test for integer-to-fp64 emulated conversions
 *
 * Validates that fpemu library integer-to-double conversions produce
 * bit-identical results to native casts for all four integer types:
 *   int32_t, uint32_t, int64_t, uint64_t
 *
 * The int32/uint32 conversions are always exact (all values fit in 52-bit mantissa).
 * The int64/uint64 conversions require round-to-nearest-even for values with
 * more than 53 significant bits.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdint>
#include <cinttypes>
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

static uint64_t to_bits(double d) { uint64_t b; memcpy(&b, &d, 8); return b; }

// ============================================================================
// Device kernels: convert via fpemu, convert via native cast
// ============================================================================

TARGET void kern_from_int32(const int32_t* in, double* emu, double* ref, int n) {
    for (int i = 0; i < n; i++) {
        fp64emu e(in[i]);
        emu[i] = (double)e;
        ref[i] = (double)in[i];
    }
}

TARGET void kern_from_uint32(const uint32_t* in, double* emu, double* ref, int n) {
    for (int i = 0; i < n; i++) {
        fp64emu e(in[i]);
        emu[i] = (double)e;
        ref[i] = (double)in[i];
    }
}

TARGET void kern_from_int64(const int64_t* in, double* emu, double* ref, int n) {
    for (int i = 0; i < n; i++) {
        fp64emu e(in[i]);
        emu[i] = (double)e;
        ref[i] = (double)in[i];
    }
}

TARGET void kern_from_uint64(const uint64_t* in, double* emu, double* ref, int n) {
    for (int i = 0; i < n; i++) {
        fp64emu e(in[i]);
        emu[i] = (double)e;
        ref[i] = (double)in[i];
    }
}

// ============================================================================
// Generic test runner
// ============================================================================

template<typename T, typename KernelFn>
int run_test(const char* label, const T* vals, int n, KernelFn kernel) {
    T*      d_in;
    double* d_emu;
    double* d_ref;

    MALLOC(d_in,  n * sizeof(T));
    MALLOC(d_emu, n * sizeof(double));
    MALLOC(d_ref, n * sizeof(double));

    memcpy(d_in, vals, n * sizeof(T));

    LAUNCH(kernel, d_in, d_emu, d_ref, n);
    SYNC();

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (to_bits(d_emu[i]) != to_bits(d_ref[i])) {
            if (mismatches < 5) {
                printf("    MISMATCH [%d]: input=%" PRId64 "  emu=0x%016" PRIx64 "  ref=0x%016" PRIx64 "\n",
                       i, (int64_t)vals[i], to_bits(d_emu[i]), to_bits(d_ref[i]));
            }
            mismatches++;
        }
    }

    printf("  %-10s: %d values tested: %s", label, n,
           mismatches ? "FAIL" : "PASS");
    if (mismatches) printf(" (%d mismatches)", mismatches);
    printf("\n");

    FREE(d_in); FREE(d_emu); FREE(d_ref);
    return mismatches;
}

// ============================================================================
// Test data
// ============================================================================

static const int32_t int32_vals[] = {
    0, 1, -1, 2, -2, 100, -100, 1000000, -1000000,
    INT32_MAX, INT32_MIN, INT32_MIN + 1, INT32_MAX - 1,
    0x7FFFFFFF, (int32_t)0x80000000, 12345678, -12345678,
};

static const uint32_t uint32_vals[] = {
    0, 1, 2, 100, 1000000,
    0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0xFFFFFFFEu,
    42, 999999999, 0x12345678u, 0xDEADBEEFu,
};

static const int64_t int64_vals[] = {
    0, 1, -1, 2, -2,
    INT32_MAX, INT32_MIN, (int64_t)INT32_MAX + 1, (int64_t)INT32_MIN - 1,
    INT64_MAX, INT64_MIN, INT64_MIN + 1, INT64_MAX - 1,
    (1LL << 53), -(1LL << 53),
    (1LL << 53) + 1, (1LL << 53) - 1,
    (1LL << 53) + 2, (1LL << 53) + 3,
    (1LL << 54) - 1, -(1LL << 54) + 1,
    (1LL << 62), -(1LL << 62),
    0x100000000LL, -0x100000000LL,
    123456789012345LL, -123456789012345LL,
};

static const uint64_t uint64_vals[] = {
    0, 1, 2,
    (uint64_t)UINT32_MAX, (uint64_t)UINT32_MAX + 1,
    UINT64_MAX, UINT64_MAX - 1,
    (1ULL << 53), (1ULL << 53) + 1, (1ULL << 53) - 1,
    (1ULL << 53) + 2, (1ULL << 53) + 3,
    (1ULL << 54) - 1,
    (1ULL << 63), (1ULL << 63) + 1,
    0x8000000000000000ULL,
    0xFFFFFFFF00000000ULL,
    123456789012345ULL, 9999999999999999ULL,
};

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("  ** fpemu integer-to-double conversion unit test\n");
    printf("  ** Verifying bit-exact match against native casts\n\n");

    int errors = 0;

    errors += run_test("int32_t",  int32_vals,
                       (int)(sizeof(int32_vals)/sizeof(int32_vals[0])),
                       kern_from_int32);

    errors += run_test("uint32_t", uint32_vals,
                       (int)(sizeof(uint32_vals)/sizeof(uint32_vals[0])),
                       kern_from_uint32);

    errors += run_test("int64_t",  int64_vals,
                       (int)(sizeof(int64_vals)/sizeof(int64_vals[0])),
                       kern_from_int64);

    errors += run_test("uint64_t", uint64_vals,
                       (int)(sizeof(uint64_vals)/sizeof(uint64_vals[0])),
                       kern_from_uint64);

    printf("\nfpemu_from_int: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
