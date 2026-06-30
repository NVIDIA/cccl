/*
    fpemu_copyable.cpp - Unit Test for Trivial Copyability and Volatile Value Preservation
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2026

    This test verifies that fp64emu_t and fp64emu_unpacked_t types are
    trivially copyable and that volatile reads/writes preserve values correctly.
    A compile-time static_assert checks trivial copyability, and a runtime check
    confirms that a value survives a round-trip through a volatile object.
*/

#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

int main()
{
    static_assert(std::is_trivially_copyable<fp64emu>::value, "fp64emu must be trivially copyable");

#if __FPEMU_UNPACKED__ == 1
    static_assert(std::is_trivially_copyable<fp64emu_unpacked>::value, "fp64emu_unpacked must be trivially copyable");
#endif

    // === Test packed type (fp64emu) ===
    {
        // Create a volatile object
        volatile fp64emu vx[1];

        // Create a non-volatile object and initialize it with a value
        fp64emu x[1] = { fp64emu(1.0e+20) };

        // Assign the non-volatile object to the volatile object
        vx[0] = x[0];

        // Read back from volatile (uses template volatile copy constructor)
        fp64emu readback(vx[0]);

        // Print both objects
        printf("fp64emu:\n");
        printf("  x[0]      = %f\n", (double)x[0]);
        printf("  vx[0]     = %f\n", (double)readback);

        // Check if the volatile round-trip preserved the value
        if (readback != x[0])
        {
            printf("  ERROR: vx[0] != x[0]\n");
            return 1;
        }
        else
        {
            printf("  PASS: vx[0] == x[0]\n");
        }
    }

#if __FPEMU_UNPACKED__ == 1
    // === Test unpacked type (fp64emu_unpacked) ===
    {
        // Create a volatile object
        volatile fp64emu_unpacked vx[1];

        // Create a non-volatile object and initialize it with a value
        fp64emu_unpacked x[1] = { fp64emu_unpacked(1.0e+20) };

        // Assign the non-volatile object to the volatile object
        vx[0] = x[0];

        // Read back from volatile (uses template volatile copy constructor)
        fp64emu_unpacked readback(vx[0]);

        // Print both objects
        printf("fp64emu_unpacked:\n");
        printf("  x[0]      = %f\n", (double)x[0]);
        printf("  vx[0]     = %f\n", (double)readback);

        // Check if the volatile round-trip preserved the value
        if (readback != x[0])
        {
            printf("  ERROR: vx[0] != x[0]\n");
            return 1;
        }
        else
        {
            printf("  PASS: vx[0] == x[0]\n");
        }
    }
#endif

    return 0;
}
