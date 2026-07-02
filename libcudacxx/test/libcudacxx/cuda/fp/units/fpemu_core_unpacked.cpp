#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if __FPEMU_UNPACKED__ == 1

/**
 * @brief Example demonstrating the usage of fp64emu library unpacked core builtins for emulated floating-point arithmetic
 * 
 * This example compares native double-precision arithmetic operations with fp64emu unpacked functions:
 * - Basic arithmetic: addition and multiplication
 * - Multiply-add operations (simulated FMA)
 * 
 * The example demonstrates the unpacked floating-point workflow:
 * 1. Convert double values to unpacked format using fpbits64_unpacked_t
 * 2. Perform arithmetic operations using unpacked core builtins:
 *    - Addition: mid accuracy (__nv_fp64emu_unpacked_mid_dadd)
 *    - Multiplication: mid accuracy (__nv_fp64emu_unpacked_mid_dmul)
 * 3. Convert results back to double format for comparison
 * 
 * For each operation type, the results are compared between:
 * - Host computation using native double precision
 * - Device computation using native double precision 
 * - Device computation using fp64emu unpacked core builtins
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

    // packed emulated doubles
    fpbits64_t ex = __nv_fp64emu_from_double(dx);
    fpbits64_t ey = __nv_fp64emu_from_double(dy);
    fpbits64_t ez = __nv_fp64emu_from_double(dz);
    fpbits64_t ew = __nv_fp64emu_from_double(dw);
    // packed multiply
    fpbits64_t eres_mul = __nv_fp64emu_mid_dmul_rn(ex, ey);
    eres_mul            = __nv_fp64emu_mid_dmul_rn(eres_mul, ez);
    eres_mul            = __nv_fp64emu_mid_dmul_rn(eres_mul, ew);
    eout[0]             = __nv_fp64emu_to_double(eres_mul);
    // packed add
    fpbits64_t eres_add = __nv_fp64emu_mid_dadd_rn(ex, ey);
    eres_add            = __nv_fp64emu_mid_dadd_rn(eres_add, ez);
    eres_add            = __nv_fp64emu_mid_dadd_rn(eres_add, ew);
    eout[1]             = __nv_fp64emu_to_double(eres_add);
    // packed mad
    fpbits64_t eres_m1  = __nv_fp64emu_mid_mad_rn(ex, ey, ez);
    eout[2]             = __nv_fp64emu_to_double(eres_m1);
    // packed dot
    fpbits64_t eres_dot  = __nv_fp64emu_mid_dot_rn(ex, ez, ey, ew);
    eout[3]              = __nv_fp64emu_to_double(eres_dot);
    // packed poly
    fpbits64_t poly = __nv_fp64emu_dmul_rn(ex,__nv_fp64emu_from_double(C7));
    poly    = __nv_fp64emu_dadd_rn(poly,__nv_fp64emu_from_double(C6));
    poly    = __nv_fp64emu_dmul_rn(poly,ex);
    poly    = __nv_fp64emu_dadd_rn(poly,__nv_fp64emu_from_double(C5));
    poly    = __nv_fp64emu_dmul_rn(poly,ex);
    poly    = __nv_fp64emu_dadd_rn(poly,__nv_fp64emu_from_double(C4));
    poly    = __nv_fp64emu_dmul_rn(poly,ex);
    poly    = __nv_fp64emu_dadd_rn(poly,__nv_fp64emu_from_double(C3));
    poly    = __nv_fp64emu_dmul_rn(poly,ex);
    poly    = __nv_fp64emu_dadd_rn(poly,__nv_fp64emu_from_double(C2));
    poly    = __nv_fp64emu_dmul_rn(poly,ex);
    poly    = __nv_fp64emu_dadd_rn(poly,__nv_fp64emu_from_double(C1));
    poly    = __nv_fp64emu_dmul_rn(poly,ex);
    poly    = __nv_fp64emu_dadd_rn(poly,__nv_fp64emu_from_double(C0));
    eout[4] = __nv_fp64emu_to_double(poly);

    // unpacked emulated doubles
    fpbits64_unpacked_t eux = __nv_fp64emu_unpacked_from_double(dx);
    fpbits64_unpacked_t euy = __nv_fp64emu_unpacked_from_double(dy);
    fpbits64_unpacked_t euz = __nv_fp64emu_unpacked_from_double(dz);
    fpbits64_unpacked_t euw = __nv_fp64emu_unpacked_from_double(dw);
    // unpacked multiply
    fpbits64_unpacked_t eures_mul = __nv_fp64emu_unpacked_mid_dmul(eux, euy);
    eures_mul                     = __nv_fp64emu_unpacked_mid_dmul(eures_mul, euz);
    eures_mul                     = __nv_fp64emu_unpacked_mid_dmul(eures_mul, euw);
    euout[0]                      = __nv_fp64emu_unpacked_to_double(eures_mul);
    // unpacked add
    fpbits64_unpacked_t eures_add = __nv_fp64emu_unpacked_mid_dadd(eux, euy);
    eures_add                     = __nv_fp64emu_unpacked_mid_dadd(eures_add, euz);
    eures_add                     = __nv_fp64emu_unpacked_mid_dadd(eures_add, euw);
    euout[1]                      = __nv_fp64emu_unpacked_to_double(eures_add);
    // unpacked mad
    fpbits64_unpacked_t eures_m1  = __nv_fp64emu_unpacked_mid_mad(eux, euy, euz);
    euout[2]                      = __nv_fp64emu_unpacked_to_double(eures_m1);
    // unpacked dot
    fpbits64_unpacked_t eures_dot = __nv_fp64emu_unpacked_mid_dot(eux, euz, euy, euw);
    euout[3]                      = __nv_fp64emu_unpacked_to_double(eures_dot);
    
    fpbits64_unpacked_t upoly = __nv_fp64emu_unpacked_mid_dmul(eux,__nv_fp64emu_unpacked_from_double(C7));
    upoly    = __nv_fp64emu_unpacked_mid_dadd(upoly,__nv_fp64emu_unpacked_from_double(C6));
    upoly    = __nv_fp64emu_unpacked_mid_dmul(upoly,eux);
    upoly    = __nv_fp64emu_unpacked_mid_dadd(upoly,__nv_fp64emu_unpacked_from_double(C5));
    upoly    = __nv_fp64emu_unpacked_mid_dmul(upoly,eux);
    upoly    = __nv_fp64emu_unpacked_mid_dadd(upoly,__nv_fp64emu_unpacked_from_double(C4));
    upoly    = __nv_fp64emu_unpacked_mid_dmul(upoly,eux);
    upoly    = __nv_fp64emu_unpacked_mid_dadd(upoly,__nv_fp64emu_unpacked_from_double(C3));
    upoly    = __nv_fp64emu_unpacked_mid_dmul(upoly,eux);
    upoly    = __nv_fp64emu_unpacked_mid_dadd(upoly,__nv_fp64emu_unpacked_from_double(C2));
    upoly    = __nv_fp64emu_unpacked_mid_dmul(upoly,eux);
    upoly    = __nv_fp64emu_unpacked_mid_dadd(upoly,__nv_fp64emu_unpacked_from_double(C1));
    upoly    = __nv_fp64emu_unpacked_mid_dmul(upoly,eux);
    upoly    = __nv_fp64emu_unpacked_mid_dadd(upoly,__nv_fp64emu_unpacked_from_double(C0));
    euout[4] = __nv_fp64emu_unpacked_to_double(upoly);

    return;
}

int main(int argc, char** argv) 
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
    printf("  ** This example demonstrates the usage of fp64emu library unpacked core builtins\n\n");

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
    printf("fpemu_core_unpacked: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // Free Unified Memory
    FREE(inp);
    FREE(out);
    FREE(eout);
    FREE(euout);

    return errors ? 1 : 0;
}
#else
int main(int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    printf("FPEMU unpacked core is not enabled\n");
    return 0;
}
#endif // __FPEMU_UNPACKED__ == 1