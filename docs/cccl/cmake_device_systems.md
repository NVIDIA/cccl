### Build Instructions

To choose a device backend (`CPP`, `OMP`, `CUDA`), set the `THRUST_BACKEND` variable when running CMake.

#### CPP Backend

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

