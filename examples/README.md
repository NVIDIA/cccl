
This directory contains examples of how to use CCCL in your project.

See the `README.md` in each subdirectory for more information.

To build and run only the examples, run the following commands from the root directory of the repository:

```bash
cmake -S . -B build -DCCCL_ENABLE_EXAMPLES=ON -DCCCL_ENABLE_THRUST=OFF -DCCCL_ENABLE_CUB=OFF -DCCCL_ENABLE_LIBCUDACXX=OFF -DCCCL_ENABLE_TESTING=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```
