#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

/**
 * @brief Example demonstrating mixed arithmetic operations using fp64emu library
 * 
 * This example showcases:
 * 1. Complex arithmetic expressions combining multiple operations
 *    - Basic arithmetic, FMA, conditional expressions, mixed operations with constants
 * 2. Mixed-type builtin calls (__dadd_rn, __dmul_rn, etc.) where one argument
 *    is fp64emu and the other is a plain arithmetic type (double or int)
 * 
 * Results are compared between native and fp64emu implementations to demonstrate
 * the enhanced precision and control provided by fp64emu.
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
    // constants
   constexpr double c1 = 9.876;
   constexpr int    c2 = -6;

    // Input values
    double dx = inp[0];
    double dy = inp[1];
    double dz = inp[2];

    // Create fp64emu objects for input values
    fp64emu ex = dx;
    fp64emu ey = dy;
    fp64emu ez = dz;

    // Perform native double-precision arithmetic
    *out  = (dx < dy) ? c2 + (dx * dy + dz) * c2 + fma(dz, dy, dx) / (dz - dx) + c1: 
                        c1 + (dx * dz - dy) * c1 + fma(dx, dz, dy) / (dx - dz) + c2;
    // Perform fp64emu arithmetic
    *eout = (ex < ey) ? c2 + (ex * ey + ez) * c2 + fma(ez, ey, ex) / (ez - ex) + c1: 
                        c1 + (ex * ez - ey) * c1 + fma(ex, ez, ey) / (ex - ez) + c2;

    // --- Mixed-type builtin calls ---
    // Builtins support mixed types: one fp64emu and one arithmetic type (double, int, etc.)
    // The arithmetic argument is implicitly converted to fp64emu before the operation.
    eout[1] = __dadd_rn(ex, 2.5);        // fp64emu + double constant
    eout[2] = __dadd_rn(2.5, ex);        // double constant + fp64emu
    eout[3] = __dmul_rn(ex, c2);         // fp64emu * int constant
    eout[4] = __dsub_rn(ex, 1.0);        // fp64emu - double constant
    eout[5] = __dadd_rn(c2, ey);         // int constant + fp64emu

    // Native equivalents for comparison
    out[1] = dx + 2.5;
    out[2] = 2.5 + dx;
    out[3] = dx * c2;
    out[4] = dx - 1.0;
    out[5] = c2 + dy;

    return;
}

int main(int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    // constants
    constexpr double c1 = 9.876;
    constexpr int    c2 = -6;

    // Declare pointers for input, output and fp64emu results
    double* inp;
    double* out;
    double* eout;

    // Allocate Unified Memory – accessible from CPU or GPU 
    MALLOC(inp,  3 * sizeof(double));
    MALLOC(out,  6 * sizeof(double));
    MALLOC(eout, 6 * sizeof(double));

    // Initialize input values
    inp[0] = 1.2345 * argc;
    inp[1] = 2.3456 * argc;
    inp[2] = 3.4567 * argc;

    // Perform native double-precision arithmetic on host
    double host_res = (inp[0] < inp[1]) ? c2 + (inp[0] * inp[1] + inp[2]) * c2 + fma(inp[2], inp[1], inp[0]) / (inp[2] - inp[0]) + c1: 
                                          c1 + (inp[0] * inp[2] - inp[1]) * c1 + fma(inp[0], inp[2], inp[1]) / (inp[0] - inp[2]) + c2;
    // Launch CUDA kernel
    RUN_EXAMPLE(inp, out, eout);

    // Wait for GPU to finish before accessing results on CPU
    DEVICE_SYNCHRONIZE();

    // Print results
    printf("  ** This example demonstrates mixed arithmetic operations using fp64emu library\n\n");
    printf("--- Complex expression ---\n");
    printf("         Host result:  %.8f\n",   host_res);
    printf("       Device result:  %.8f\n",   out[0]);
    printf(" Device fp64emu result:  %.8f\n\n", eout[0]);

    printf("--- Mixed-type builtin calls ---\n");
    printf("  __dadd_rn(ex, 2.5):   native=%.8f  fp64emu=%.8f\n", out[1], eout[1]);
    printf("  __dadd_rn(2.5, ex):   native=%.8f  fp64emu=%.8f\n", out[2], eout[2]);
    printf("  __dmul_rn(ex, %d):    native=%.8f  fp64emu=%.8f\n", c2, out[3], eout[3]);
    printf("  __dsub_rn(ex, 1.0):   native=%.8f  fp64emu=%.8f\n", out[4], eout[4]);
    printf("  __dadd_rn(%d, ey):    native=%.8f  fp64emu=%.8f\n", c2, out[5], eout[5]);


    // Basic verification: fp64emu results should match native double within tolerance
    const double tol = 1e-10;
    int errors = 0;
    for (int i = 0; i < 6; i++) {
        double err = fabs(eout[i] - out[i]);
        if (err > tol) { printf("FAIL: result[%d] err=%.2e (native=%.8f fp64emu=%.8f)\n", i, err, out[i], eout[i]); errors++; }
    }
    printf("fpemu_mixed: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // Free Unified Memory
    FREE(inp);
    FREE(out);
    FREE(eout);

    return errors ? 1 : 0;
}