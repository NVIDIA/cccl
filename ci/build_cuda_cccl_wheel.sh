#!/usr/bin/env bash
set -euo pipefail

# Target script for the per-CUDA builds in build_cuda_cccl_python.sh.
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
pip_config_args=()

if cccl_python_is_windows; then
  # Prefer the newest installed minor for the requested CUDA major. The outer
  # container builds CUDA 12 and the nested native container builds CUDA 13.
  case "${CCCL_WHEEL_CUDA_MAJOR:-}" in
    12 | 13) ;;
    *) echo "CCCL_WHEEL_CUDA_MAJOR must be 12 or 13" >&2; exit 1 ;;
  esac
  cuda_path=""
  while IFS= read -r variable; do
    if [[ "$variable" =~ ^CUDA_PATH_V${CCCL_WHEEL_CUDA_MAJOR}_[0-9]+$ ]]; then
      cuda_path="${!variable}"
    fi
  done < <(compgen -e | sort -V)
  cuda_path="${cuda_path:-${CUDA_PATH:-}}"
  if [[ -z "$cuda_path" ]]; then
    echo "CUDA Toolkit ${CCCL_WHEEL_CUDA_MAJOR} not found" >&2
    exit 1
  fi
  cuda_path=$(cygpath -am "$cuda_path")
  nvcc_path="${cuda_path}/bin/nvcc.exe"
  test -f "$(cygpath -u "$nvcc_path")"
  command -v cl
  export CC=cl.exe CXX=cl.exe CUDAHOSTCXX=cl.exe
  export CUDACXX="$nvcc_path" CUDA_PATH="$cuda_path"
  PATH="$(cygpath -u "$cuda_path")/bin:$PATH"
  export PATH
  export CMAKE_GENERATOR=Ninja
  pip_config_args=(
    -C cmake.define.CMAKE_C_COMPILER=cl.exe
    -C cmake.define.CMAKE_CXX_COMPILER=cl.exe
    -C "cmake.define.CMAKE_CUDA_COMPILER=${nvcc_path}"
    -C "cmake.define.CUDAToolkit_ROOT=${cuda_path}"
  )
else

  # Toolchain version, pinned in one place. Every gcc-toolset package name and the
  # /opt/rh enable path below derive from it — the -fsanitize=thread libtsan must
  # come from the same toolset as the compiler, so these cannot drift apart.
  readonly gcc_toolset_version=13

  # Install the GCC toolset (needed for the build) and ccache (shared between
  # cu12 and cu13 builds via /root/.ccache bind-mount from the host).
  /workspace/ci/util/retry.sh 5 30 dnf -y install \
    "gcc-toolset-${gcc_toolset_version}-gcc" "gcc-toolset-${gcc_toolset_version}-gcc-c++" ccache

  # ThreadSanitizer builds link libcccl.c.parallel.so with -fsanitize=thread, which
  # the linker resolves via the toolset's libtsan.so. That runtime lives in a
  # separate package not pulled in by -gcc-c++, so install it for the TSan lane.
  if [[ "${CCCL_C_PARALLEL_SANITIZE_THREAD:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
    /workspace/ci/util/retry.sh 5 30 dnf -y install "gcc-toolset-${gcc_toolset_version}-libtsan-devel"
  fi

  # When the caller bind-mounts a ccache dir, wire it through to CMake. This
  # transparently caches every compile, so the second wheel build (cu13 after
  # cu12, or vice versa) reuses the entire LLVM/clang object tree.
  if [[ -n "${CCACHE_DIR:-}" ]]; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    echo "ccache enabled: CCACHE_DIR=${CCACHE_DIR}"
    ccache --version 2>&1 | head -1 || true
    ccache --show-stats 2>&1 | head -5 || true
  fi
  echo -e "#!/usr/bin/env bash\nsource /opt/rh/gcc-toolset-${gcc_toolset_version}/enable" >/etc/profile.d/enable_devtools.sh
  # shellcheck disable=SC1091
  source /etc/profile.d/enable_devtools.sh

  # Check what's available
  command -v gcc
  gcc --version
  command -v nvcc
  nvcc --version
fi

# Set up Python environment
# shellcheck disable=SC2154
setup_python_env "${py_version}"
command -v python
python --version
echo "Done setting up python env"

# Keep the existing manylinux version override. Native Windows wheels use the
# version from setuptools-scm, just as the previous Windows builder did.
if ! cccl_python_is_windows; then
  # Figure out the version to use for the package, we need repo history
  if "$(git rev-parse --is-shallow-repository)"; then
    git fetch --unshallow
  fi
  export PACKAGE_VERSION_PREFIX="0.1."
  package_version=$(/workspace/ci/generate_version.sh)
  echo "Using package version ${package_version}"
  # Override the version used by setuptools_scm to the custom version
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_CCCL="${package_version}"
fi

cd "$repo_root/python/cuda_cccl"

# Git Bash's native locale may not support grep's PCRE mode. Capture only the
# version digits, which also discards native nvcc's CRLF line ending.
cuda_version=$(nvcc --version | sed -nE 's/.*release ([0-9]+)\.[0-9]+.*/\1/p')
echo "Detected CUDA version: ${cuda_version}"
if cccl_python_is_windows && [[ "$cuda_version" != "$CCCL_WHEEL_CUDA_MAJOR" ]]; then
  echo "Expected CUDA ${CCCL_WHEEL_CUDA_MAJOR}, found ${cuda_version}" >&2
  exit 1
fi

# Configure compilers:
if ! cccl_python_is_windows; then
  CXX="$(command -v g++)"
  export CXX
  CUDACXX="$(command -v nvcc)"
  export CUDACXX
  CUDAHOSTCXX="$(command -v g++)"
  export CUDAHOSTCXX
fi

# When CCCL_PYTHON_USE_V2 is set (=1/true/on), build the wheel against the
# HostJIT-based cccl.c.parallel.v2 library instead of the default v1.
if [[ "${CCCL_PYTHON_USE_V2:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCCCL_PYTHON_USE_V2=ON"
  echo "Building wheel with CCCL v2 backend: CMAKE_ARGS=${CMAKE_ARGS}"

  if ! cccl_python_is_windows; then
    # v2's hostjit links against libnvJitLink and libnvfatbin, which aren't in
    # the base rapidsai/ci-wheel image. Install the matching CTK devel packages
    # so CMake's FindCUDAToolkit picks them up. nvcc is on PATH; derive the
    # version (e.g. "13-0") from it.
    ctk_pkg_ver=$(nvcc --version 2>/dev/null \
      | sed -nE 's/.*release ([0-9]+)\.([0-9]+).*/\1-\2/p')
    if [[ -n "${ctk_pkg_ver}" ]]; then
      echo "Installing libnvjitlink-devel-${ctk_pkg_ver} libnvfatbin-devel-${ctk_pkg_ver}..."
      /workspace/ci/util/retry.sh 5 30 dnf -y install \
        "libnvjitlink-devel-${ctk_pkg_ver}" \
        "libnvfatbin-devel-${ctk_pkg_ver}"
    else
      echo "WARNING: could not derive CTK version from nvcc; skipping nvJitLink/nvfatbin install"
    fi

    # FindCUDAToolkit learned about CUDA::nvfatbin only in CMake 3.27. The base
    # rapidsai/ci-wheel image ships an older CMake; install a newer one into
    # the active venv so scikit-build-core picks it up over the system cmake.
    echo "Pinning cmake>=3.27 for FindCUDAToolkit nvfatbin support..."
    python -m pip install --upgrade 'cmake>=3.27'
  fi
fi

# When CCCL_C_PARALLEL_SANITIZE_THREAD is set (=1/true/on), instrument the
# c.parallel (v1) host code with ThreadSanitizer for the free-threaded TSan
# nightly lane. Host-only; the libtsan runtime stays external (the shared
# build_cuda_cccl_python.sh --excludes it from auditwheel).
if [[ "${CCCL_C_PARALLEL_SANITIZE_THREAD:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCCCL_C_PARALLEL_SANITIZE_THREAD=ON"
  echo "Building wheel with ThreadSanitizer-instrumented c.parallel: CMAKE_ARGS=${CMAKE_ARGS}"
fi

# Build the wheel
python -m pip wheel --no-deps --verbose --wheel-dir dist "${pip_config_args[@]}" .

if cccl_python_is_windows; then
  built_wheels=(dist/cuda_cccl-*.whl)
  if [[ ${#built_wheels[@]} != 1 || ! -f "${built_wheels[0]}" ]]; then
    echo "Error: expected exactly one built CUDA ${CCCL_WHEEL_CUDA_MAJOR} wheel" >&2
    exit 1
  fi
fi

# Rename wheel to include CUDA version suffix
for wheel in dist/cuda_cccl-*.whl; do
    if [[ -f "$wheel" ]]; then
        base_name=$(basename "$wheel" .whl)
        new_name="${base_name}.cu${cuda_version}.whl"
        mv "$wheel" "dist/${new_name}"
        echo "Renamed wheel to: ${new_name}"
    fi
done

# Move wheel to output directory
mkdir -p "$repo_root/wheelhouse"
mv dist/cuda_cccl-*.cu*.whl "$repo_root/wheelhouse/"
