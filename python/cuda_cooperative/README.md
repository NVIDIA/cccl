# `cuda.cooperative`: Experimental CUDA Core Compute Library for Python

## Documentation

Please visit the documentation here: https://nvidia.github.io/cccl/python.html.

## Local development

First-time installation:

```bash
pip3 install ./cuda_cccl
pip3 install ./cuda_cooperative[test]
pytest -v ./cuda_cooperative/tests/
```

For faster iterative development:

```bash
pip3 install -e ./cuda_cooperative[test]
```
