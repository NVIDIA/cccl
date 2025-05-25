#!/bin/bash

setup_python_env() {
    local py_version=$1

    # check if pyenv is installed
    if ! command -v pyenv &> /dev/null; then
        rm -f /pyenv
        curl -fsSL https://pyenv.run | bash
    fi

    # Install the build dependencies, check /etc/os-release to see if we are on ubuntu or rocky
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        if [ "$ID" = "ubuntu" ]; then
            # Use the retry helper to mitigate issues with apt network errors:
            script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
            retry() {
              "${script_dir}/util/retry.sh" 5 30 "$@"
            }

            retry sudo apt update
            retry sudo apt install -y make libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev curl git \
            libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
        elif [ "$ID" = "rocky" ]; then
            # we're inside the rockylinux container, sudo not required/available
            dnf install -y make patch zlib-devel bzip2 bzip2-devel readline-devel \
            sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel libuuid-devel \
            gdbm-libs libnsl2
        else
            echo "Unsupported Linux distribution"
            exit 1
        fi
    fi

    # Always set up pyenv environment
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"

    # Using pyenv, install the Python version
    PYENV_DEBUG=1 pyenv install -v "${py_version}"
    pyenv local "${py_version}"

    pip install --upgrade pip
}
