.. _libcudacxx-setup-building:

Building & Testing libcu++
==========================

\*nix Systems, Native Build/Test
--------------------------------

The procedure is demonstrated for NVCC + GCC in C++11 mode on a
Debian-like Linux systems; the same basic steps are required on all
other platforms.

Step 0: Install Build Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell:

.. code:: bash

   # Install LLVM (needed for LLVM's CMake modules)
   apt-get -y install llvm

   # Install CMake
   apt-get -y install cmake

   # Install the LLVM Integrated Tester (`lit`)
   apt-get -y install python-pip
   pip install lit

   # Env vars that should be set, or kept in mind for use later
   export LIBCUDACXX_ROOT=/path/to/libcudacxx # Git repo root.

Step 1: Generate the Build Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell:

.. code:: bash

   cd ${LIBCUDACXX_ROOT}
   cmake \
       -S ./ \
       -B build \
       -DCMAKE_CXX_COMPILER=$CXX \
       -DCMAKE_CUDA_COMPILER=$TOOLKIT/bin/nvcc \
       -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON

Step 2: Build & Run the Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell:

.. code:: bash

   cd ${LIBCUDACXX_ROOT}/build # build directory of this repo
   ../utils/nvidia/linux/perform_tests.bash --skip-libcxx-tests

\*nix Systems, Cross Build/Test
-------------------------------

The procedure is demonstrated for NVCC + GCC cross compiler in C++14
mode on a Debian-like Linux systems targeting an aarch64 L4T system; the
same basic steps are required on all other platforms.

Step 0: Install Build Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow Step 0 for \*nix native builds/tests.

.. _step-1-generate-the-build-files-1:

Step 1: Generate the Build Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell:

.. code:: bash

   export HOST=executor.nvidia.com
   export USERNAME=ubuntu

   cd ${LIBCUDACXX_ROOT}
   cmake \
     -S ./ \
     -B build \
     -DCMAKE_CUDA_COMPILER=$TOOLKIT/bin/nvcc \
     -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
     -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON \
     -DLIBCXX_EXECUTOR="SSHExecutor(host='${HOST}', username='${USERNAME}')"

Ensure that you can SSH to the target system from the host system
without inputing a password (e.g. use SSH keys).

.. _step-2-build-run-the-tests-1:

Step 2: Build & Run the Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow Step 2 for \*nix native builds/tests.

\*nix Systems, NVRTC Build/Test
-------------------------------

The procedure is demonstrated for NVRTC in C++11 mode on a Debian-like
Linux systems; the same basic steps are required on all other platforms.

.. _step-0-install-build-prerequisites-1:

Step 0: Install Build Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow Step 0 for \*nix native builds/tests.

.. _step-1-generate-the-build-files-2:

Step 1: Generate the Build Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell:

.. code:: bash

   cd ${LIBCUDACXX_ROOT}
   cmake \
     -S ./ \
     -B build \
     -DCMAKE_CXX_COMPILER=$CC \
     -DCMAKE_CUDA_COMPILER=$TOOLKIT/bin/nvcc \
     -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON \
     -DLIBCUDACXX_TEST_WITH_NVRTC=ON

.. _step-2-build-run-the-tests-2:

Step 2: Build & Run the Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow Step 2 for \*nix native builds/tests.

Windows, Native Build/Test
--------------------------

.. _step-0-install-build-requirements-1:

Step 0: Install Build Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Install Python <https://www.python.org/downloads/windows>`_.

Download `the get-pip.py bootstrap
script <https://bootstrap.pypa.io/get-pip.py>`_ and run it.

Install the LLVM Integrated Tester (``lit``) using a Visual Studio
command prompt:

.. code:: bat

   pip install lit

Step 0.5: Launching a Build Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visual Studio comes with a few build environments that are appropriate
to use.

The ``x64 Native Tools Command Prompt`` and other similarly named
environments will work.

If Powershell is desired, it would be best to launch it from within the
native tools. This helps avoid configuration step issues.

.. _step-1-generate-the-build-files-3:

Step 1: Generate the Build Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Visual Studio command prompt:

.. code:: bat

   set LIBCUDACXX_ROOT=\path\to\libcudacxx # Helpful env var pointing to the git repo root.
   cd %LIBCUDACXX_ROOT%

   cmake ^
     -S ./ ^
     -B build ^
     -G "Ninja" ^
     -DCMAKE_CXX_COMPILER=cl ^
     -DCMAKE_CUDA_COMPILER=nvcc ^
     -DCMAKE_CUDA_COMPILER_FORCED=ON ^
     -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON

.. _step-2-build-run-the-tests-3:

Step 2: Build & Run the Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SM_ARCH`` can be set to any integer value (Ex: “80”, “86”)

.. code:: bat

   set LIBCUDACXX_SITE_CONFIG=%LIBCUDACXX_ROOT%\build\test\lit.site.cfg
   lit %LIBCUDACXX_ROOT%\test -Dcompute_archs=%SM_ARCH% -sv --no-progress-bar
