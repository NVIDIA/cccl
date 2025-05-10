#!/bin/bash

setup_python_env() {
    local py_version=$1

    # check if pyenv is installed
    if ! command -v pyenv &> /dev/null; then
        rm -f /pyenv
        curl -fsSL https://pyenv.run | bash
        echo "Installing pyenv"
    fi

    # Always set up pyenv environment
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"

    # Using pyenv, install the Python version
    echo $py_version
    which pyenv
    PYENV_DEBUG=1 pyenv install -v "${py_version}"
    echo "Done installing pyenv"
    pyenv local "${py_version}"
}
