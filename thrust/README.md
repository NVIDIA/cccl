# Thrust: The C++ Parallel Algorithms Library

Thrust is the C++ parallel algorithms library which inspired the introduction
  of parallel algorithms to the C++ Standard Library.
Thrust's **high-level** interface greatly enhances programmer **productivity**
  while enabling performance portability between GPUs and multicore CPUs.
It builds on top of established parallel programming frameworks (such as CUDA,
  TBB, and OpenMP).
It also provides a number of general-purpose facilities similar to those found
  in the C++ Standard Library.

Thrust is an open source project; it is available on
  [GitHub] and included in the NVIDIA HPC SDK and CUDA Toolkit.
If you have one of those SDKs installed, no additional installation or compiler
  flags are needed to use Thrust.

## Examples

Thrust is best learned through examples.

The following example generates random numbers serially and then transfers them
  to a parallel device where they are sorted.

```cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

int main() {
  // Generate 32M random numbers serially.
  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Transfer data to the device.
  thrust::device_vector<int> d_vec = h_vec;

  // Sort data on the device.
  thrust::sort(d_vec.begin(), d_vec.end());

  // Transfer data back to host.
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
```

[See it on Godbolt](https://godbolt.org/z/GeWEd8Er9)

This example demonstrates computing the sum of some random numbers in parallel:

```cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

int main() {
  // Generate random data serially.
  thrust::default_random_engine rng(1337);
  thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
  thrust::host_vector<double> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Transfer to device and compute the sum.
  thrust::device_vector<double> d_vec = h_vec;
  double x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
}
```

[See it on Godbolt](https://godbolt.org/z/cnsbWWME7)

This example show how to perform such a reduction asynchronously:

```cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/async/copy.h>
#include <thrust/async/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <numeric>

int main() {
  // Generate 32M random numbers serially.
  thrust::default_random_engine rng(123456);
  thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
  thrust::host_vector<double> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Asynchronously transfer to the device.
  thrust::device_vector<double> d_vec(h_vec.size());
  thrust::device_event e = thrust::async::copy(h_vec.begin(), h_vec.end(),
                                               d_vec.begin());

  // After the transfer completes, asynchronously compute the sum on the device.
  thrust::device_future<double> f0 = thrust::async::reduce(thrust::device.after(e),
                                                           d_vec.begin(), d_vec.end(),
                                                           0.0, thrust::plus<double>());

  // While the sum is being computed on the device, compute the sum serially on
  // the host.
  double f1 = std::accumulate(h_vec.begin(), h_vec.end(), 0.0, thrust::plus<double>());
}
```

[See it on Godbolt](https://godbolt.org/z/be54efaKj)

## Getting The Thrust Source Code

Thrust is a header-only library; there is no need to build or install the project unless you want to run the Thrust unit tests.
The CUDA Toolkit provides a recent release of the Thrust source code in `include/thrust`. This will be suitable for most users.
Users that wish to contribute to Thrust or try out newer features should recursively clone the Thrust Github repository:

```bash
git clone --recursive https://github.com/NVIDIA/thrust.git
```

## Using Thrust From Your Project

For CMake-based projects, we provide a CMake package for use with `find_package`. See :ref:`CMake Options <cmake-options>` for more information.
Thrust can also be added via `add_subdirectory` or tools like the [CMake Package Manager](https://github.com/cpm-cmake/CPM.cmake).

For non-CMake projects, compile with:

- The Thrust include path (`-I<thrust repo root>`)
- The libcu++ include path (`-I<thrust repo root>/dependencies/libcudacxx/`)
- The CUB include path, if using the CUDA device system (`-I<thrust repo root>/dependencies/cub/`)
- By default, the CPP host system and CUDA device system are used.
  These can be changed using compiler definitions:
  - `-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_XXX`,
     where `XXX` is `CPP` (serial, default), `OMP` (OpenMP), or `TBB` (Intel TBB)
  - `-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_XXX`, where `XXX` is
    `CPP`, `OMP`, `TBB`, or `CUDA` (default).

## Developing Thrust

Thrust uses the [CMake build system] to build unit tests, examples, and header tests.
To build Thrust as a developer, it is recommended that you use our containerized development system:

```bash
# Clone Thrust and CUB repos recursively:
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust

# Build and run tests and examples:
ci/local/build.bash
```

That does the equivalent of the following, but in a clean containerized environment which has all dependencies installed:

```bash
# Clone Thrust and CUB repos recursively:
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust

# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake ..   # Command line interface.
ccmake ..  # ncurses GUI (Linux only).
cmake-gui  # Graphical UI, set source/build directories in the app.

# Build:
cmake --build . -j ${NUM_JOBS} # Invokes make (or ninja, etc).

# Run tests and examples:
ctest
```

By default, a serial `CPP` host system, `CUDA` accelerated device system, and C++14 standard are used.
This can be changed in CMake and via flags to `ci/local/build.bash`

More information on configuring your Thrust build and creating a pull request can be found in the [contributing section].

## Licensing

Thrust is an open source project developed on [GitHub].
Thrust is distributed under the [Apache License v2.0 with LLVM Exceptions].
Some parts are distributed under the [Apache License v2.0] and the [Boost License v1.0].

[GitHub]: https://github.com/NVIDIA/cccl/tree/main/thrust

[contributing section]: https://nvidia.github.io/cccl/thrust/contributing.html

[CMake build system]: https://cmake.org

[Apache License v2.0 with LLVM Exceptions]: https://llvm.org/LICENSE.txt
[Apache License v2.0]: https://www.apache.org/licenses/LICENSE-2.0.txt
[Boost License v1.0]: https://www.boost.org/LICENSE_1_0.txt
