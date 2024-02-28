## CUDA Next: Library for experimental features in CUDA C++ Core Libraries.
CUDA Next serves as a distribution channel for features that are considered experimental in the CUDA C++ Core Libraries.
Some of them are still activelly designed or developed and their API is evolving.
Some of them are specific to one hardware architecture and are still looking for a generic and forward compatible exposure.
Finally, some of them need to prove useful enough to deserve long term support.

**All APIs available in CUDA Next are not considered stable and can change without a notice.** It can also be deprecated or removed on a much faster cadence than in other CCCL libraries. 

Features are exposed here for the CUDA C++ community to experiment with and provide to provide feedback on how to shape it to best fit their use cases.
Once we become confident a feature is ready and would be a great permement addition in CCCL, it will become a part of some other CCCL library with a stable API.

## Installation
CUDA Next library is **not** distributed with the CUDA Toolkit like the rest of CCCL. It is only avaiable on the [CCCL GitHub repository](https://github.com/NVIDIA/cccl).

Everything in CUDA Next is header-only, so cloning and including it in a simple project is as easy as the following:
```bash
git clone https://github.com/NVIDIA/cccl.git
# Note:
nvcc -Icccl/cuda_next/include main.cu -o main
```