FPEMU Examples
==============

Examples demonstrating **FPEMU** — emulated IEEE-754 double-precision arithmetic
via integer compute units — for hardware where native FP64 is limited, absent,
or slow.

For build/run instructions and the wider examples landscape, see the
[examples README](README.md).

Available Examples
------------------

### fpemu.cpp — FP64 Emulation Demo

Demonstrates emulated double-precision floating-point arithmetic:

- **Precision**: configurable via the `accuracy` template parameter
- **Use case**: Double-precision on hardware where native FP64 is limited, absent, or slow

Features demonstrated:
- Construction and assignment of `fp64emu` values
- Using `fp64emu_accuracy::high` for addition (correctly rounded, full IEEE-754 range)
- Using `fp64emu_accuracy::low` for multiplication (relaxed precision, higher throughput)
- Default accuracy (`fp64emu_accuracy::def`, == high) for subtraction, division, sqrt, fma
- Comparison operators
- Compound assignment operators (+=, *=)
- CPU/GPU host/device compatibility
- Accuracy comparison with native `double`
