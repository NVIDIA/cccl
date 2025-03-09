Once Thrust has been installed, these example programs can be compiled
directly with nvcc.  For example, the following command will compile the
`norm` example.

```bash
$ nvcc norm.cu -o norm
```

These examples are also available online:
https://github.com/NVIDIA/cccl/tree/main/thrust/examples

For any serious experimentation, we recommend using CMake and [CCCL from GitHub](https://github.com/NVIDIA/cccl).
We also provide consistent and convenient development environments as [devcontainers](../../.devcontainers/README.md).
