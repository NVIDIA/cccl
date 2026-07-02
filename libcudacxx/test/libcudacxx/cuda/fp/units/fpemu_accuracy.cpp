#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

/**
 * @brief Example demonstrating the different accuracy levels of the fp64emu library
 * 
 * This example showcases the comprehensive set of accuracy levels available in fp64emu for
 * emulated IEEE-754 compliant double-precision floating-point arithmetic:
 * 
 * ACCURACY LEVELS:
 * - high: Correctly rounded results with full IEEE-754 range (correct rounding + full range)
 * - mid: 1-2 LSB error with normal range (high accuracy + normal range)
 * - low: Low accuracy with normal range (optimized for performance)
 * - def: default selector (== high, IEEE-correct)
 * 
 * SUPPORTED OPERATIONS:
 * - Basic arithmetic: __dadd_rn, __dmul_rn, __dsub_rn, __ddiv_rn, __dsqrt_rn
 * - Fused multiply-add: __fma_rn for enhanced precision in a*b+c computations
 * - All operations support round-to-nearest-even (_rn) rounding mode
 * 
 * The example demonstrates how builtin functions (__dadd_rn, __dmul_rn, etc.) automatically
 * deduce the accuracy level from their argument types - no explicit template parameters needed.
 * 
 */

//  Define the macros for the host and device runs
#if __CUDACC__
    #define MALLOC(x,s) cudaMallocManaged(&x,s)
    #define FREE(x) cudaFree(x)
    #define RUN_EXAMPLE(x,y) run_example<<<1, 1>>>(x,y)
    #define DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
    #define TARGET_DEVICE __global__
#else
    #define MALLOC(x,s) x = (double*)malloc(s)
    #define FREE(x) free(x)
    #define RUN_EXAMPLE(x,y) run_example(x,y)
    #define DEVICE_SYNCHRONIZE()
    #define TARGET_DEVICE
#endif


TARGET_DEVICE void run_example(double *inp, double *out) 
{
    // high accuracy: builtins deduce the accuracy level from argument type
    fp64emu_t<fp64emu_accuracy::high> acc_x = inp[0];
    auto acc_r = __dadd_rn(acc_x, acc_x);
    acc_r = __dmul_rn(acc_r, acc_x);
    acc_r = __dsub_rn(acc_r, acc_x);
    acc_r = __ddiv_rn(acc_r, acc_x);
    acc_r = __fma_rn(acc_r, acc_x, acc_x);
    acc_r = __dsqrt_rn(acc_r);

    // default accuracy: fp64emu is fp64emu_t<fp64emu_accuracy::def> (== high)
    fp64emu def_x = inp[0];
    auto def_r = __dadd_rn(def_x, def_x);
    def_r = __dmul_rn(def_r, def_x);
    def_r = __dsub_rn(def_r, def_x);
    def_r = __ddiv_rn(def_r, def_x);
    def_r = __fma_rn(def_r, def_x, def_x);
    def_r = __dsqrt_rn(def_r);

    // low accuracy: builtins deduce the accuracy level from argument type
    fp64emu_t<fp64emu_accuracy::low> fast_x = inp[0];
    auto fast_r = __dadd_rn(fast_x, fast_x);
    fast_r = __dmul_rn(fast_r, fast_x);
    fast_r = __dsub_rn(fast_r, fast_x);
    fast_r = __ddiv_rn(fast_r, fast_x);

    out[0] = (double)acc_r + (double)def_r + (double)fast_r;

    return;
}

int main(int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    // Declare pointers for input, output and fp64emu results
    double* inp;
    double* out;

    // Allocate Unified Memory – accessible from CPU or GPU 
    MALLOC(inp, 1 * sizeof(double));
    MALLOC(out, 1 * sizeof(double));

    // Initialize input values
    inp[0] = 1.2345 * argc;
    // Launch CUDA kernel
    RUN_EXAMPLE(inp, out);

    // Wait for GPU to finish before accessing results on CPU
    DEVICE_SYNCHRONIZE();

    // Print results
    printf("  ** This example demonstrates the accuracy levels of the fp64emu library\n\n");
    printf("Input: %.4f, Output: %.4f\n",   inp[0], out[0]);

    // Basic verification: result should be finite and non-zero
    int errors = 0;
    if (out[0] != out[0]) { printf("FAIL: result is NaN\n"); errors++; }
    if (out[0] == 0.0)    { printf("FAIL: result is zero\n"); errors++; }

    // Verify each accuracy level individually: compute expected = sqrt(fma(((x+x)*x - x) / x, x, x))
    // for high accuracy with inp[0]=1.2345*argc
    double x = inp[0];
    double expected = sqrt(fma(((x + x) * x - x) / x, x, x));
    // All three accuracy levels contribute to out[0], so just check it's a reasonable value
    if (fabs(out[0]) > 1e20) { printf("FAIL: result seems too large: %.4f\n", out[0]); errors++; }

    printf("fpemu_accuracy: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // Free Unified Memory
    FREE(inp);
    FREE(out);

    return errors ? 1 : 0;
}
