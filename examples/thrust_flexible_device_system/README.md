# Thrust Flexible Device System Example

This example illustrates best practices for writing generic CMake that supports user-configuration of the Thrust device system via the `CCCL_THRUST_DEVICE_SYSTEM` CMake option.

Valid values for this option are:

- `CUDA`
- `OMP` (OpenMP)
- `TBB` (Intel Thread Building Blocks)
- `CPP` (Serial C++ backend)

The CMakeLists.txt file for this example is annotated to show how to achieve a generic build system that supports any of device system.

## How To Use This Example

Configure and build this example as follows:

```
# Checkout example and prepare build directory:
git clone https://github.com/NVIDIA/cccl.git
cd cccl/thrust_flexible_device_system
mkdir build
cd build

# Configure:
cmake .. -DCCCL_THRUST_DEVICE_SYSTEM=CUDA # or TBB, OMP, CPP

# Build:
cmake --build .

# Run:
ctest -V
```

## Advanced Thrust Usecases

For more control over the Thrust configuration, see the Thrust CMake package's [README.md](../../lib/cmake/thrust/README.md).
This details how to use the `thrust_create_target` function to generate Thrust interface targets in CMake.

If using `thrust_create_target` directly, you may also want to set the CMake option `CCCL_ENABLE_DEFAULT_THRUST_TARGET=OFF` to prevent the default `CCCL::Thrust` target from being initialized.
This will avoid checking for any dependencies required for the default target that may be unnecessary for your project.

## Further Reading About CCCL + CMake

The `basic` example's [README.md](../basic/README.md) has additional information that you may find useful for using CCCL with CPM and CMake.
