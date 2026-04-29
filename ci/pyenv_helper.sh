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
