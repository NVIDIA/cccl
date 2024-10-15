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

## Getting The Thrust Source Code & Developing Thrust

Thrust started as a stand-alone project, but as of March 2024 Thrust is a part of the
CUDA Core Compute Libraries (CCCL). Please refer to the [CCCL Getting Started section] and the
[Contributing Guide] for instructions on how to get started developing the CCCL sources.

[CCCL Getting Started]: https://github.com/NVIDIA/cccl?tab=readme-ov-file#getting-started
[Contributing Guide]: https://github.com/NVIDIA/cccl/blob/main/CONTRIBUTING.md
