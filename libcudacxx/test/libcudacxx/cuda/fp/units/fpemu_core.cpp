#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)
/**
 * @brief Example demonstrating the usage of fp64emu library for emulated floating-point arithmetic
 * 
 * This example compares native double-precision arithmetic operations with fp64emu library functions:
 * - Basic arithmetic: multiplication, addition, division, and subtraction
 * - Fused multiply-add (FMA) operations
 * 
 * The example performs computations using:
 * 1. Native double-precision arithmetic
 * 2. fp64emu builtin functions with different accuracy levels:
 *    - Multiplication: def accuracy
 *    - Addition: high accuracy
 *    - Division: low accuracy
 *    - Subtraction: low accuracy
 *    - FMA: low accuracy
 * 
 * For each operation type, the results are compared between:
 * - Host computation using native double precision
 * - Device computation using native double precision 
 * - Device computation using fp64emu library
 * 
 * This demonstrates the emulated precision capabilities of fp64emu compared to standard
 * floating-point arithmetic.
 */


//  Define the macros for the host and device runs
#if __CUDACC__
    #define MALLOC(x,s) cudaMallocManaged(&x,s)
    #define FREE(x) cudaFree(x)
    #define RUN_EXAMPLE(x,y,z) run_example<<<1, 1>>>(x,y,z)
    #define DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
    #define TARGET_DEVICE __global__
#else
    #define MALLOC(x,s) x = (double*)malloc(s)
    #define FREE(x) free(x)
    #define RUN_EXAMPLE(x,y,z) run_example(x,y,z)
    #define DEVICE_SYNCHRONIZE()
    #define TARGET_DEVICE
#endif


TARGET_DEVICE void run_example(double *inp, double *out, double* eout) 
{
    // Input values
    double dx = inp[0];
    double dy = inp[1];
    double dz = inp[2];

    // Convert double to fpbits64
    fpbits64_t ex = __nv_fp64emu_from_double(dx);
    fpbits64_t ey = __nv_fp64emu_from_double(dy);
    fpbits64_t ez = __nv_fp64emu_from_double(dz);

    // Perform native double-precision arithmetic
    out[0] = dx * dy;
    out[1] = dx + dy;
    out[2] = dx / dy;
    out[3] = dx - dy;
    out[4] = dx * dy + dz;

    // Perform fp64emu arithmetic using builtins: high for add/sub, def for others
    eout[0] = __nv_fp64emu_to_double(__nv_fp64emu_mid_dmul_rn(ex, ey));
    eout[1] = __nv_fp64emu_to_double(__nv_fp64emu_high_dadd_rn(ex, ey));
    eout[2] = __nv_fp64emu_to_double(__nv_fp64emu_mid_ddiv_rn(ex, ey));
    eout[3] = __nv_fp64emu_to_double(__nv_fp64emu_high_dsub_rn(ex, ey));
    eout[4] = __nv_fp64emu_to_double(__nv_fp64emu_mid_fma_rn(ex, ey, ez));
    return;
}

int main(int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    // Declare pointers for input, output and fp64emu results
    double* inp;
    double* out;
    double* eout;

    // Allocate Unified Memory – accessible from CPU or GPU 
    MALLOC(inp,  3 * sizeof(double));
    MALLOC(out,  5 * sizeof(double));
    MALLOC(eout, 5 * sizeof(double));

    // Initialize input values
    inp[0] = 1.2345 * argc;
    inp[1] = 2.3456 * argc;
    inp[2] = 3.4567 * argc;

    // Launch CUDA kernel
    RUN_EXAMPLE(inp, out, eout);

    // Wait for GPU to finish before accessing results on CPU
    DEVICE_SYNCHRONIZE();

    // Print results
    printf("  ** This example demonstrates the usage of fp64emu library core builtins \n     which have control over the accuracy and range\n\n");

    printf("                          Host: %.4f * %.4f = %.8f\n",   inp[0], inp[1], inp[0] * inp[1]);
    printf("               Device (native): %.4f * %.4f = %.8f\n",   inp[0], inp[1], out[0]);
    printf("           Device (fp64emu core): %.4f * %.4f = %.8f\n\n", inp[0], inp[1], eout[0]);

    printf("                          Host: %.4f + %.4f = %.8f\n",   inp[0], inp[1], inp[0] + inp[1]);
    printf("               Device (native): %.4f + %.4f = %.8f\n",   inp[0], inp[1], out[1]);
    printf("           Device (fp64emu core): %.4f + %.4f = %.8f\n\n", inp[0], inp[1], eout[1]);

    printf("                          Host: %.4f / %.4f = %.8f\n",   inp[0], inp[1], inp[0] / inp[1]);
    printf("               Device (native): %.4f / %.4f = %.8f\n",   inp[0], inp[1], out[2]);
    printf("           Device (fp64emu core): %.4f / %.4f = %.8f\n\n", inp[0], inp[1], eout[2]);

    printf("                          Host: %.4f - %.4f = %.8f\n",   inp[0], inp[1], inp[0] - inp[1]);
    printf("               Device (native): %.4f - %.4f = %.8f\n",   inp[0], inp[1], out[3]);
    printf("           Device (fp64emu core): %.4f - %.4f = %.8f\n\n", inp[0], inp[1], eout[3]);

    printf("                          Host: fma(%.4f, %.4f, %.4f) = %.8f\n",   inp[0], inp[1], inp[2], std::fma(inp[0],inp[1],inp[2]));
    printf("               Device (native): fma(%.4f, %.4f, %.4f) = %.8f\n",   inp[0], inp[1], inp[2], out[4]);
    printf("           Device (fp64emu core): fma(%.4f, %.4f, %.4f) = %.8f\n\n", inp[0], inp[1], inp[2], eout[4]);

    // Basic verification: fp64emu core results should match native double within tolerance
    // All operations use high or def accuracy — tight tolerance expected
    const double tol = 1e-10;
    const double tols[] = {tol, tol, tol, tol, tol};
    int errors = 0;
    const char* op_names[] = {"mul(def)", "add(high)", "div(def)", "sub(high)", "fma(def)"};
    for (int i = 0; i < 5; i++) {
        double err = fabs(eout[i] - out[i]);
        if (err > tols[i]) { printf("FAIL: %s err=%.2e\n", op_names[i], err); errors++; }
    }
    printf("fpemu_core: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // Free Unified Memory
    FREE(inp);
    FREE(out);
    FREE(eout);

    return errors ? 1 : 0;
}