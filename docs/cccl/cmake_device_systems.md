# Using Thrust with CMake: Selecting Device Backends

This guide shows how to compile a Thrust-based project using different device backends (`CPP`, `OMP`, `CUDA`) using CMake.

## Example Project

See: [`examples/device_system_selector`](../examples/device_system_selector)

### Build Instructions

**CPP Backend**

```bash
cmake -B build-cpp -DTHRUST_BACKEND=CPP
cmake --build build-cpp
```

**OMP Backend**
```bash
cmake -B build-omp -DTHRUST_BACKEND=OMP
cmake --build build-omp
```

**CUDA Backend**
```bash
cmake -B build-cuda -DTHRUST_BACKEND=CUDA
cmake --build build-cuda
```
