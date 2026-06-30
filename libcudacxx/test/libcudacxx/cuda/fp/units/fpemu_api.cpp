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
 * - Different accuracy levels (high, mid, low)
 * 
 * The example performs computations using:
 * 1. Native double-precision arithmetic
 * 2. fp64emu default functions
 * 3. fp64emu builtin functions with specific accuracy levels
 * 
 * Results are compared to demonstrate the enhanced precision and control provided by fp64emu.
 */


//  Define the macros for the host and device runs
#if __CUDACC__
    #define MALLOC(x,s) cudaMallocManaged(&x,s)
    #define FREE(x) cudaFree(x)
    #define RUN_EXAMPLE(x,y,z,r) run_example<<<1, 1>>>(x,y,z,r)
    #define DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
    #define TARGET_DEVICE __global__
#else
    #define MALLOC(x,s) x = (double*)malloc(s)
    #define FREE(x) free(x)
    #define RUN_EXAMPLE(x,y,z,r) run_example(x,y,z,r)
    #define DEVICE_SYNCHRONIZE()
    #define TARGET_DEVICE
#endif


TARGET_DEVICE void run_example(double *inp, double *out, double* eout, double* eout_builtin) 
{
    // Input values
    double dx = inp[0];
    double dy = inp[1];
    double dz = inp[2];

    // Create fp64emu objects for input values
    fp64emu ex = dx;
    fp64emu ey = dy;
    fp64emu ez = dz;

    // Perform native double-precision arithmetic
    out[0] = dx * dy;
    out[1] = dx + dy;
    out[2] = dx / dy;
    out[3] = dx - dy;
    out[4] = dx * dy + dz;

    // Perform fp64emu arithmetic with default accuracy and range
    eout[0] = ex * ey;
    eout[1] = ex + ey;
    eout[2] = ex / ey;
    eout[3] = ex - ey;
    eout[4] = fma(ex, ey, ez);

    // Perform fp64emu arithmetic using builtin functions with accuracy-specific variants
    eout_builtin[0] = __dmul_rn(ex, ey);
    eout_builtin[1] = __dadd_rn(ex, ey);
    eout_builtin[2] = __ddiv_rn(ex, ey);
    eout_builtin[3] = __dsub_rn(ex, ey);    
    eout_builtin[4] = __fma_rn(ex, ey, ez);

    return;
}

int main(int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    // Declare pointers for input, output and fp64emu results
    double* inp;
    double* out;
    double* eout;
    double* eout_builtin;

    // Allocate Unified Memory – accessible from CPU or GPU 
    MALLOC(inp,          3 * sizeof(double));
    MALLOC(out,          5 * sizeof(double));
    MALLOC(eout,         5 * sizeof(double));
    MALLOC(eout_builtin, 5 * sizeof(double));

    // Initialize input values
    inp[0] = 1.2345 * argc;
    inp[1] = 2.3456 * argc;
    inp[2] = 3.4567 * argc;

    // Launch CUDA kernel
    RUN_EXAMPLE(inp, out, eout, eout_builtin);

    // Wait for GPU to finish before accessing results on CPU
    DEVICE_SYNCHRONIZE();

    // Print results
    printf("  ** This example demonstrates the usage of fp64emu library C++ API\n     with fp64emu type and accuracy parameters to control the accuracy level\n\n");
    
    printf("                      Host:           %.4f * %.4f  = %.8f\n",   inp[0], inp[1], inp[0] * inp[1]);
    printf("           Device (native):           %.4f * %.4f  = %.8f\n",   inp[0], inp[1], out[0]);
    printf("        Device (fp64emu C++):           %.4f * %.4f  = %.8f\n",   inp[0], inp[1], eout[0]);
    printf("Device (fp64emu C++ builtin): __dmul_rn(%.4f,  %.4f) = %.8f\n\n", inp[0], inp[1], eout_builtin[0]);

    printf("                      Host:           %.4f + %.4f  = %.8f\n",   inp[0], inp[1], inp[0] + inp[1]);
    printf("           Device (native):           %.4f + %.4f  = %.8f\n",   inp[0], inp[1], out[1]);
    printf("        Device (fp64emu C++):           %.4f + %.4f  = %.8f\n",   inp[0], inp[1], eout[1]);
    printf("Device (fp64emu C++ builtin): __dadd_rn(%.4f,  %.4f) = %.8f\n\n", inp[0], inp[1], eout_builtin[1]);

    printf("                      Host:            %.4f / %.4f = %.8f\n",   inp[0], inp[1], inp[0] / inp[1]);
    printf("           Device (native):            %.4f / %.4f = %.8f\n",   inp[0], inp[1], out[2]);
    printf("        Device (fp64emu C++):            %.4f / %.4f = %.8f\n",   inp[0], inp[1], eout[2]);
    printf("Device (fp64emu C++ builtin): __ddiv_rn(%.4f, %.4f)  = %.8f\n\n", inp[0], inp[1], eout_builtin[2]);

    printf("                      Host:            %.4f - %.4f = %.8f\n",   inp[0], inp[1], inp[0] - inp[1]);
    printf("           Device (native):            %.4f - %.4f = %.8f\n",   inp[0], inp[1], out[3]);
    printf("        Device (fp64emu C++):            %.4f - %.4f = %.8f\n",   inp[0], inp[1], eout[3]);
    printf("Device (fp64emu C++ builtin): __dsub_rn(%.4f, %.4f)  = %.8f\n\n", inp[0], inp[1], eout_builtin[3]);

    printf("                      Host:      fma(%.4f, %.4f, %.4f) = %.8f\n",   inp[0], inp[1], inp[2], std::fma(inp[0],inp[1],inp[2]));
    printf("           Device (native):      fma(%.4f, %.4f, %.4f) = %.8f\n",   inp[0], inp[1], inp[2], out[4]);
    printf("        Device (fp64emu C++):      fma(%.4f, %.4f, %.4f) = %.8f\n",   inp[0], inp[1], inp[2], eout[4]);
    printf("Device (fp64emu C++ builtin): __fma_rn(%.4f, %.4f, %.4f) = %.8f\n\n", inp[0], inp[1], inp[2], eout_builtin[4]);

    // Basic verification: fp64emu results should match native double within tolerance
    const double tol = 1e-10;
    int errors = 0;
    const char* op_names[] = {"mul", "add", "div", "sub", "fma"};
    for (int i = 0; i < 5; i++) {
        double err_cpp = fabs(eout[i] - out[i]);
        double err_bi  = fabs(eout_builtin[i] - out[i]);
        if (err_cpp > tol) { printf("FAIL: %s C++ API err=%.2e\n", op_names[i], err_cpp); errors++; }
        if (err_bi  > tol) { printf("FAIL: %s builtin err=%.2e\n", op_names[i], err_bi);  errors++; }
    }
    printf("fpemu_api: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // Free Unified Memory
    FREE(inp);
    FREE(out);
    FREE(eout);
    FREE(eout_builtin);

    return errors ? 1 : 0;
}