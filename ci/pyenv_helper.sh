setup_python_env() {
    local py_version=$1

    # Source pretty_printing.sh for begin_group/end_group helpers
    local script_dir
    script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # shellcheck source=ci/pretty_printing.sh
    source "${script_dir}/pretty_printing.sh"

    begin_group "🐍 Setting up Python ${py_version} (uv)"

    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Create a venv with the requested Python version.
    # uv downloads a pre-built CPython binary automatically — no compilation needed.
    uv venv --seed --python "${py_version}" "${HOME}/.cccl-venv"

    # Windows venvs use Scripts/, Linux/macOS use bin/
    if [[ -f "${HOME}/.cccl-venv/Scripts/activate" ]]; then
        #shellcheck disable=SC1091
        source "${HOME}/.cccl-venv/Scripts/activate"
    else
        #shellcheck disable=SC1091
        source "${HOME}/.cccl-venv/bin/activate"
    fi

    end_group "🐍 Setting up Python ${py_version} (uv)"
}

# Pin the cuda-toolkit wheels to the container's CTK major.minor (read from nvcc)
# via PIP_CONSTRAINT, unless the lane opts out with CCCL_PYTHON_TEST_LATEST_CTK=1
# -- in which case pip resolves whatever the latest minor is (what a plain
# `pip install` with no lockfile would get). The sysctk lanes
# (CCCL_PYTHON_TEST_SYSCTK=1) skip pinning entirely: they install no cuda-toolkit
# wheel and rely on the system-provided toolkit instead. Sets and exports
# cuda_version and cuda_major_version for the caller (e.g. the [test-cuNN] extra).
pin_cuda_toolkit() {
    cuda_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1-2 | cut -d 'V' -f 2)
    cuda_major_version=$(echo "$cuda_version" | cut -d '.' -f 1)
    export cuda_version cuda_major_version

    if [[ "${CCCL_PYTHON_TEST_SYSCTK:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
        # sysctk lanes use the system-provided CUDA toolkit; no cuda-toolkit wheel
        # is pip-installed, so there is nothing to pin. Clear any inherited
        # constraint so it cannot affect the sysctk resolve.
        unset PIP_CONSTRAINT
        return 0
    fi

    if [[ "${CCCL_PYTHON_TEST_LATEST_CTK:-}" != "1" ]]; then
        export PIP_CONSTRAINT="${TMPDIR:-/tmp}/ctk-constraint.txt"
        echo "cuda-toolkit==${cuda_version}.*" > "${PIP_CONSTRAINT}"
    else
        # Clear any inherited constraint so this lane truly tests the latest minor.
        unset PIP_CONSTRAINT
    fi
}

# Echoes the pip-extra toolkit "flavor" used to build extra names: "sysctk" when
# CCCL_PYTHON_TEST_SYSCTK=1 (rely on the system-provided CUDA toolkit) or "cu"
# otherwise (pip-installed toolkit). Combine with the CUDA major, e.g.
# "minimal-$(ctk_extra_flavor)${cuda_major_version}" -> minimal-sysctk12 / minimal-cu12.
ctk_extra_flavor() {
    if [[ "${CCCL_PYTHON_TEST_SYSCTK:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
        echo "sysctk"
    else
        echo "cu"
    fi
}
