
# CUDA CCCL Python Package

## ✅ Installation Instructions for WSL (Ubuntu on Windows)

> **Note:** We recommend using **Python 3.12**. Python 3.13 is **not yet officially supported** and may not work with `cuda-cccl`.

### 1. Install the Python Package

Run the following **inside your WSL Ubuntu environment** (on the Linux VM drive, e.g., `/home/your_username`):

```bash
pip install cuda-cccl
```

---

### 2. Install the CUDA Toolkit

Download and install the CUDA Toolkit 12.9.1:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run
```

> **Tip:** On the [official CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads), choose:
> - **Operating System:** Linux
> - **Architecture:** x86_64
> - **Distribution:** WSL-Ubuntu
> - **Version:** 2.0 (latest)
> - **Installer Type:** runfile (local)

---

### 3. Install the NVIDIA Driver (if needed)

```bash
sudo sh cuda_12.9.1_575.57.08_linux.run --silent --driver
```

---

### 4. Set Environment Variables

Add the following lines to your `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### 5. Restart WSL

To apply all changes:

```bash
wsl --shutdown
```

Then reopen your WSL terminal.

---

### 6. Validate the Setup

Check if CUDA and the GPU driver are correctly installed:

```bash
nvcc --version
nvidia-smi
```


You can also test the package by running one of our [examples]((https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/parallel/examples/reduction/basic_reduce.py):

```bash
python basic_reduce.py
```

Expected Output:

```bash
Running basic reduction examples...
Sum: 15
All examples completed successfully!
```
