#!/usr/bin/env bash

set -euo pipefail

# Ensure the script is being executed in the cccl/ root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

print_help() {
    echo "Usage: $0 [-c|--cuda <CUDA version>] [-H|--host <Host compiler>] [-d|--docker]"
    echo "Launch a development container. If no CUDA version or Host compiler are specified,"
    echo "the top-level devcontainer in .devcontainer/devcontainer.json will be used."
    echo ""
    echo "Options:"
    echo "  -c, --cuda       Specify the CUDA version. E.g., 12.2"
    echo "  -H, --host       Specify the host compiler. E.g., gcc12"
    echo "  -d, --docker     Launch the development environment in Docker directly without using VSCode."
    echo "  -h, --help       Display this help message and exit."
}

_upvar() {
    if unset -v "$1"; then
        if (( $# == 2 )); then
            eval $1=\"\$2\";
        else
            eval $1=\(\"\${@:2}\"\);
        fi;
    fi
}

parse_options() {
    local -;
    set -euo pipefail;

    local UNPARSED="${!#}";
    set -- "${@:1:$#-1}";

    local OPTIONS=c:H:dh
    local LONG_OPTIONS=cuda:,host:,docker,help
    # shellcheck disable=SC2155
    local PARSED_OPTIONS="$(getopt -n "$0" -o "${OPTIONS}" --long "${LONG_OPTIONS}" -- "$@")"

    # shellcheck disable=SC2181
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    eval set -- "${PARSED_OPTIONS}"

    while true; do
        case "$1" in
            -c|--cuda)
                cuda_version="$2"
                shift 2
                ;;
            -H|--host)
                host_compiler="$2"
                shift 2
                ;;
            -d|--docker)
                docker_mode=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            --)
                shift
                _upvar "${UNPARSED}" "${@}"
                break
                ;;
            *)
                echo "Invalid option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

# shellcheck disable=SC2155
launch_docker() {
    local -;
    set -euo pipefail;

    # Read image
    local DOCKER_IMAGE="$(
        grep '"image":' "${path}/devcontainer.json" \
      | sed 's/.*: "\(.*\)",/\1/'
    )"

    # Read workspaceFolder
    local WORKSPACE_FOLDER="$(
        grep '"workspaceFolder":' "${path}/devcontainer.json" \
      | sed 's/.*: "\(.*\)",/\1/' \
      | sed "s@\${localWorkspaceFolderBasename}@$(basename "$(pwd)")@g"
    )"

    # Read runArgs
    local -a RUN_ARGS="($(
        grep -Pzo '(?s)"runArgs": \[(.*?)\s*\]' "${path}/devcontainer.json" \
      | grep -Pzo '(?s)\[(.*?)\s*\]' \
      | cut -d'[' -f1 --complement | rev | cut -d']' -f1 --complement | rev \
      | tr -s '[:blank:]' \
      | tr '\n' '\0' \
      | xargs -0 -r -n1 echo -n \
      | sed -r 's/", "/" "/g' \
      | sed "s@\${localWorkspaceFolder}@$(pwd)@g" \
      | sed "s@\${localWorkspaceFolderBasename}@$(basename "$(pwd)")@g" \
      | sed -r 's/\$\{localEnv:(.*)\}/${\1-}/' \
      | xargs -0 -r -n1 bash -c "eval echo \$0"
    ))"

    if [[ " ${RUN_ARGS[*]} " != *" --rm "* ]]; then
        RUN_ARGS+=(--rm)
    fi

    # Read initializeCommand
    local INITIALIZE_COMMAND="$(
        grep -Pzo '(?s)"initializeCommand": \[(.*?)\s*\]' "${path}/devcontainer.json" \
      | grep -Pzo '(?s)\[(.*?)\s*\]' \
      | cut -d'[' -f1 --complement | rev | cut -d']' -f1 --complement | rev \
      | tr -s '[:blank:]' \
      | tr '\n' '\0' \
      | xargs -0 -r -n1 echo -n \
      | sed -r 's/", "/" "/g' \
      | sed "s@\${localWorkspaceFolder}@$(pwd)@g" \
      | sed "s@\${localWorkspaceFolderBasename}@$(basename "$(pwd)")@g"
    )"

    # shellcheck disable=SC2207
    local -a ENVVARS="($(
        grep -Pzo '(?s)"containerEnv": {(.*?)\s+}' "${path}/devcontainer.json" \
      | head -n-1 | tail -n+2 \
      | rev | cut -d',' -f1 --complement | rev \
      | sed -r 's/\$\{localEnv:(.*)\}/${\1-}/' \
      | sed -r 's/": /=/' \
      | cut -d'"' -f1 --complement \
      | tr -d '[:blank:]' \
      | tr '\n' '\0' \
      | xargs -0 -r -n1 bash -c "echo \$(eval echo \${0})" \
      | xargs -0 -r -n1 bash -c "sed -r \"s/(.*)=(.*)/\1='\2'/\" <<< \$0" \
      | head -n-1 \
      | tr '\n' '\0' \
      | xargs -0 -r -n1 bash -c "echo '--env' \$0"
    ))"

    ENVVARS+=(--env REMOTE_USER=coder)
    ENVVARS+=(--env NEW_UID="$(id -u)")
    ENVVARS+=(--env NEW_GID="$(id -g)")

    # shellcheck disable=SC2207
    local -a MOUNTS=($(
        grep 'type=bind' "${path}/devcontainer.json" \
      | sed -r 's/.*: "(.*)",/\1/' \
      | tr -d \" | tr -d '[:blank:]' \
      | rev | cut -d',' -f1 --complement | rev \
      | sed "s@\${localWorkspaceFolder}@$(pwd)@g" \
      | sed "s@\${localWorkspaceFolderBasename}@$(basename "$(pwd)")@g" \
      | xargs -r -n1 echo --mount
    ))

    if test -n "${SSH_AUTH_SOCK:-}"; then
        ENVVARS+=(--env "SSH_AUTH_SOCK=/tmp/ssh-auth-sock");
        MOUNTS+=(--mount "source=${SSH_AUTH_SOCK},target=/tmp/ssh-auth-sock,type=bind");
    fi

    if test "${#INITIALIZE_COMMAND}" -gt 0; then
        eval "${INITIALIZE_COMMAND}"
    fi

    echo "Found image: ${DOCKER_IMAGE}"
    docker pull "${DOCKER_IMAGE}"
    docker run \
        -it --init \
        -u root:root \
        --workdir "${WORKSPACE_FOLDER}" \
        --entrypoint /home/coder/cccl/.devcontainer/docker-entrypoint.sh \
        "${RUN_ARGS[@]}" \
        "${ENVVARS[@]}" \
        "${MOUNTS[@]}" \
        "${DOCKER_IMAGE}" \
        "$@"
}

launch_vscode() {
    local -;
    set -euo pipefail;
    # Since Visual Studio Code allows only one instance per `devcontainer.json`,
    # this code prepares a unique temporary directory structure for each launch of a devcontainer.
    # By doing so, it ensures that multiple instances of the same environment can be run
    # simultaneously. The script replicates the `devcontainer.json` from the desired CUDA
    # and compiler environment into this temporary directory, adjusting paths to ensure the
    # correct workspace is loaded. A special URL is then generated to instruct VSCode to
    # launch the development container using this temporary configuration.
    local workspace="$(basename "$(pwd)")"
    local tmpdir="$(mktemp -d)/${workspace}"
    mkdir -p "${tmpdir}"
    mkdir -p "${tmpdir}/.devcontainer"
    cp -arL "${path}/devcontainer.json" "${tmpdir}/.devcontainer"
    sed -i "s@\${localWorkspaceFolder}@$(pwd)@g" "${tmpdir}/.devcontainer/devcontainer.json"
    local path="${tmpdir}"
    local hash="$(echo -n "${path}" | xxd -pu - | tr -d '[:space:]')"
    local url="vscode://vscode-remote/dev-container+${hash}/home/coder/cccl"

    local launch=""
    if type open >/dev/null 2>&1; then
        launch="open"
    elif type xdg-open >/dev/null 2>&1; then
        launch="xdg-open"
    fi

    if [ -n "${launch}" ]; then
        echo "Launching VSCode Dev Container URL: ${url}"
        code --new-window "${tmpdir}"
        exec "${launch}" "${url}" >/dev/null 2>&1
    fi
}

main() {
    local -a unparsed;
    parse_options "$@" unparsed;
    set -- "${unparsed[@]}";

    # If no CTK/Host compiler are provided, just use the default environment
    if [[ -z ${cuda_version:-} ]] && [[ -z ${host_compiler:-} ]]; then
        path=".devcontainer"
    else
        path=".devcontainer/cuda${cuda_version}-${host_compiler}"
        if [[ ! -f "${path}/devcontainer.json" ]]; then
            echo "Unknown CUDA [${cuda_version}] compiler [${host_compiler}] combination"
            echo "Requested devcontainer ${path}/devcontainer.json does not exist"
            exit 1
        fi
    fi

    if ${docker_mode:-'false'}; then
        launch_docker "$@"
    else
        launch_vscode
    fi
}

main "$@"
