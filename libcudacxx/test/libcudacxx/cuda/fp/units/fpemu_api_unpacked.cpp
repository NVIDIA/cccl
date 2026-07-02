#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if __FPEMU_UNPACKED__ == 1

/*
 * @brief Example demonstrating the usage of fp64emu library unpacked C++ API for emulated floating-point arithmetic
 * 
 * This example compares native double-precision arithmetic operations with fp64emu unpacked functions:
 * - Basic arithmetic: addition and multiplication
 * - Multiply-add operations (simulated FMA)
 * 
 * The example demonstrates the unpacked floating-point workflow:
 * 1. Convert double values to unpacked format using fpbits64_unpacked_t
 * 2. Perform arithmetic operations using unpacked C++ API:
 *    - Addition: High accuracy with normal range (ha_normal_dadd_unpacked_rn)
 *    - Multiplication: High accuracy with normal range (ha_normal_dmul_unpacked_rn)
 * 3. Convert results back to double format for comparison
 * 
 * For each operation type, the results are compared between:
 * - Host computation using native double precision
 * - Device computation using native double precision 
 * - Device computation using fp64emu unpacked C++ API
 * 
 * This demonstrates the emulated precision capabilities of fp64emu unpacked types compared to standard
 * floating-point arithmetic, showing how unpacked representation enables higher precision operations.
 */


//  Define the macros for the host and device runs
#if __CUDACC__
    #define MALLOC(x,s) cudaMallocManaged(&x,s)
    #define FREE(x) cudaFree(x)
    #define RUN_EXAMPLE(x,y,z,w) run_example<<<1, 1>>>(x,y,z,w)
    #define DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
    #define TARGET_DEVICE __global__
#else
    #define MALLOC(x,s) x = (double*)malloc(s)
    #define FREE(x) free(x)
    #define RUN_EXAMPLE(x,y,z,w) run_example(x,y,z,w)
    #define DEVICE_SYNCHRONIZE()
    #define TARGET_DEVICE
#endif

#define C0 (1.0)
#define C1 (1.0/2.0)
#define C2 (1.0/6.0)
#define C3 (1.0/24.0)
#define C4 (1.0/120.0)
#define C5 (1.0/720.0)
#define C6 (1.0/5040.0)
#define C7 (1.0/40320.0)

TARGET_DEVICE void run_example(double *inp, double *out, double* eout, double* euout) 
{
    // Input values
    auto dx = inp[0];
    auto dy = inp[1];
    auto dz = inp[2];
    auto dw = inp[3];

    // Perform native double-precision arithmetic
    out[0] = dx * dy * dz * dw;
    out[1] = dx + dy + dz + dw;
    out[2] = dx * dy + dz;
    out[3] = dx * dy + dz * dw;
    out[4] = C0 + dx * (C1 + dx * (C2 + dx * (C3 + dx * (C4 + dx * (C5 + dx * (C6 + dx * C7))))));

    // packed emulated doubles (implicit conversion to packed type)
    fp64emu ex = dx, ey = dy, ez = dz, ew = dw;
    // packed multiply
    eout[0] = __dmul_rn(ex, ey) * ez * ew;
    // packed add
    eout[1] = __dadd_rn(ex, ey) + ez + ew;
    // packed mad
    eout[2] = mad(ex, ey, ez);
    // packed dot
    eout[3] = dot(ex, ez, ey, ew);    
    // packed poly
    eout[4] = C0 + ex * (C1 + ex * (C2 + ex * (C3 + ex * (C4 + ex * (C5 + ex * (C6 + ex * C7))))));

    // unpacked emulated doubles (explicit conversion to unpacked type to avoid ambiguity with packed type!)
    fp64emu_unpacked ux = (fp64emu_unpacked)dx, 
                        uy = (fp64emu_unpacked)dy, 
                        uz = (fp64emu_unpacked)dz, 
                        uw = (fp64emu_unpacked)dw;
    // unpacked multiply
    euout[0] = __dmul_rn(ex, ey) * ez * ew;
    // unpacked add
    euout[1] = __dadd_rn(ux, uy) + uz + uw;
    // unpacked mad
    euout[2] = mad(ux, uy, uz);
    // unpacked dot
    euout[3] = dot(ux, uz, uy, uw);
    // unpacked poly
    euout[4] = C0 + ux * (C1 + ux * (C2 + ux * (C3 + ux * (C4 + ux * (C5 + ux * (C6 + ux * C7))))));

    return;
}

int main (int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    // Declare pointers for input, output and fp64emu results
    double* inp;
    double* out;
    double* eout;
    double* euout;

    // Allocate Unified Memory – accessible from CPU or GPU 
    MALLOC(inp,  4 * sizeof(double));
    MALLOC(out,  5 * sizeof(double));
    MALLOC(eout, 5 * sizeof(double));
    MALLOC(euout, 5 * sizeof(double));

    // Initialize input values
    inp[0] =  0.23451432345642 * argc;
    inp[1] = -2.34561234567899 * argc;
    inp[2] =  3.45678726352678 * argc;
    inp[3] = -4.56787263526789 * argc;

    // Launch CUDA kernel
    RUN_EXAMPLE(inp, out, eout, euout);

    // Wait for GPU to finish before accessing results on CPU
    DEVICE_SYNCHRONIZE();

    // Print results
    printf("  ** This example demonstrates the usage of fp64emu library unpacked C++ API\n\n");

    printf("                  Host MUL:  %.3f * %.3f * %.3f * %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], inp[0] * inp[1] * inp[2] * inp[3]);
    printf("       Device MUL (native):  %.3f * %.3f * %.3f * %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], out[0]);
    printf("        Device MUL (fp64emu):  %.3f * %.3f * %.3f * %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], eout[0]);
    printf("Device MUL unpacked(fp64emu):  %.3f * %.3f * %.3f * %.3f = %la\n\n", inp[0], inp[1], inp[2], inp[3], euout[0]);

    printf("                  Host ADD:  %.3f + %.3f + %.3f + %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], inp[0] + inp[1] + inp[2] + inp[3]);
    printf("       Device ADD (native):  %.3f + %.3f + %.3f + %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], out[1]);
    printf("        Device ADD (fp64emu):  %.3f + %.3f + %.3f + %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], eout[1]);
    printf("Device ADD unpacked(fp64emu):  %.3f + %.3f + %.3f + %.3f = %la\n\n", inp[0], inp[1], inp[2], inp[3], euout[1]);

    printf("                  Host MAD:  %.3f * %.3f + %.3f = %la\n",   inp[0], inp[1], inp[2], inp[0] * inp[1] + inp[2]);
    printf("       Device MAD (native):  %.3f * %.3f + %.3f = %la\n",   inp[0], inp[1], inp[2], out[2]);
    printf("        Device MAD (fp64emu):  %.3f * %.3f + %.3f = %la\n",   inp[0], inp[1], inp[2], eout[2]);
    printf("Device MAD unpacked(fp64emu):  %.3f * %.3f + %.3f = %la\n\n", inp[0], inp[1], inp[2], euout[2]);

    printf("                  Host DOT:  %.3f * %.3f + %.3f * %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], inp[0] * inp[1] + inp[2] * inp[3]);
    printf("       Device DOT (native):  %.3f * %.3f + %.3f * %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], out[3]);
    printf("        Device DOT (fp64emu):  %.3f * %.3f + %.3f * %.3f = %la\n",   inp[0], inp[1], inp[2], inp[3], eout[3]);
    printf("Device DOT unpacked(fp64emu):  %.3f * %.3f + %.3f * %.3f = %la\n\n", inp[0], inp[1], inp[2], inp[3], euout[3]);

    printf("                  Host POLY: POLY(%.3f) = %la\n",   inp[0], C0 + inp[0] * (C1 + inp[0] * (C2 + inp[0] * (C3 + inp[0] * (C4 + inp[0] * (C5 + inp[0] * (C6 + inp[0] * C7)))))));
    printf("       Device POLY (native): POLY(%.3f) = %la\n",   inp[0], out[4]);
    printf("        Device POLY (fp64emu): POLY(%.3f) = %la\n",   inp[0], eout[4]);
    printf("Device POLY unpacked(fp64emu): POLY(%.3f) = %la\n\n", inp[0], euout[4]);    
    
    // Basic verification: fp64emu results should match native double within tolerance
    const double tol = 1e-10;
    int errors = 0;
    const char* op_names[] = {"mul", "add", "mad", "dot", "poly"};
    for (int i = 0; i < 5; i++) {
        double err_packed   = fabs(eout[i] - out[i]);
        double err_unpacked = fabs(euout[i] - out[i]);
        if (err_packed   > tol) { printf("FAIL: packed %s err=%.2e\n", op_names[i], err_packed); errors++; }
        if (err_unpacked > tol) { printf("FAIL: unpacked %s err=%.2e\n", op_names[i], err_unpacked); errors++; }
    }
    printf("fpemu_api_unpacked: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // Free Unified Memory
    FREE(inp);
    FREE(out);
    FREE(eout);
    FREE(euout);

    return errors ? 1 : 0;
}
#else
int main (int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    printf("FPEMU unpacked api is not enabled\n");
    return 0;
}
#endif // __FPEMU_UNPACKED__ == 1