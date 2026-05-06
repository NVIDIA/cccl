#!/usr/bin/env python3

###
# Parses and merges the devcontainer.json metadata with the feature metadata
# from the devcontainer.json image's "devcontainer.metadata" label.
#
# Prints the relevant fields as eval-friendly shell variable declarations.
#
# Example:
# python3 .devcontainer/launch.py .devcontainer/devcontainer.json
###

import functools
import json
import os
import pathlib
import re
import subprocess
import sys

localWorkspaceFolder = str(
    pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent
)
localWorkspaceFolderBasename = os.path.basename(localWorkspaceFolder)
containerWorkspaceFolder = ""
containerWorkspaceFolderBasename = ""


def inline_vars(val):
    val = inline_local_workspace_folder(val)
    val = inline_local_workspace_folder_basename(val)
    val = inline_container_workspace_folder(val)
    val = inline_container_workspace_folder_basename(val)
    val = inline_shell_variables(val)
    return val


def inline_local_workspace_folder(val):
    return re.sub(r"\$\{localWorkspaceFolder\}", localWorkspaceFolder, val)


def inline_local_workspace_folder_basename(val):
    return re.sub(
        r"\$\{localWorkspaceFolderBasename\}", localWorkspaceFolderBasename, val
    )


def inline_container_workspace_folder(val):
    return re.sub(r"\$\{containerWorkspaceFolder\}", containerWorkspaceFolder, val)


def inline_container_workspace_folder_basename(val):
    return re.sub(
        r"\$\{containerWorkspaceFolderBasename\}", containerWorkspaceFolderBasename, val
    )


def inline_shell_variables(val):
    return re.sub(r"\$\{localEnv:([^\:]*):?(.*)\}", r"${\1:-\2}", val)


def flat_dict(list_of_dicts):
    return functools.reduce(lambda x, y: {**x, **y}, list_of_dicts, dict())


def flat_list(list_of_lists):
    return functools.reduce(list.__add__, list_of_lists, list())


def dict_list(map_):
    return [f"{k}={v}" for k, v in map_.items()]


def json_lookup(map_, path):
    for key in path:
        if key in map_:
            map_ = map_[key]
        else:
            return None
    return map_


def load_devcontainer_json(path):
    with open(path, "r") as devcontainer_file:
        return json.load(devcontainer_file)


def load_devcontainer_meta(tag):
    import time

    # Always pull the latest copy of the image
    if os.environ.get("GITHUB_ACTIONS", None) is not None:
        print(f"::group::Pulling Docker image {tag}", file=sys.stderr)

    attempts = 0
    while True:
        try:
            subprocess.run(
                ["docker", "pull", tag], stdout=sys.stderr
            ).check_returncode()
            break
        except Exception as e:
            if attempts < 10:
                attempts += 1
                time.sleep(5)
            else:
                raise e

    if os.environ.get("GITHUB_ACTIONS", None) is not None:
        print("::endgroup::", file=sys.stderr)

    return json.loads(
        json.loads(
            subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    "--type",
                    "image",
                    "--format",
                    r"{{json .Config.Labels}}",
                    tag,
                ],
                text=True,
            )
        )["devcontainer.metadata"]
    )


def normalize_mount(mount):
    if isinstance(mount, dict):
        # {"source": "/var/run/docker.sock", "target": "/var/run/docker-host.sock", "type": "bind"} ->
        # 'source="/var/run/docker.sock",target="/var/run/docker-host.sock",type="bind"'
        mount_ary = []
        for k, v in mount.items():
            mount_ary += [f"{k}={v}"]
        return r",".join(mount_ary)
    return mount


def quote_ary(ary, quote=r'"'):
    if len(ary) == 0:
        ary = ""
    else:
        ary = f"{quote} {quote}".join(ary)
        ary = f"{quote}{ary}{quote}"
    return ary


def bash_str(name, string):
    return f'declare {name}="{inline_vars(string)}"'


def bash_list(name, ary):
    return f"declare -a {name}=({quote_ary([inline_vars(x) for x in ary])})"


def bash_dict(name, ary):
    return f'declare -A {name}="({quote_ary([inline_vars(x) for x in ary])})"'


def bash_list_of_commands(name, arys):
    # Wrap each initializeCommand in quotes so they're not all
    # expanded into elements of the bash array
    return bash_list(name, [quote_ary(xs, "'") for xs in arys])


# Load the devcontainer.json
devcontainer_json = load_devcontainer_json(sys.argv[1])
containerWorkspaceFolder = inline_vars(
    devcontainer_json.get("workspaceFolder", "/home/coder/cccl")
)
containerWorkspaceFolderBasename = os.path.basename(containerWorkspaceFolder)
# The feature metadata first, then devcontainer.json
devcontainer_meta = load_devcontainer_meta(devcontainer_json["image"]) + [
    devcontainer_json
]

print(bash_str("DOCKER_IMAGE", devcontainer_json["image"]))
print(bash_str("WORKSPACE_FOLDER", containerWorkspaceFolder))
print(
    bash_str(
        "REMOTE_USER",
        [x["remoteUser"] for x in devcontainer_meta if "remoteUser" in x][-1],
    )
)

gpu_request = json_lookup(devcontainer_json, ["hostRequirements", "gpu"])
if gpu_request is None or gpu_request is False:
    print(bash_list("GPU_REQUEST", []))
if gpu_request is True:
    print(bash_list("GPU_REQUEST", ["--gpus", "all"]))
elif gpu_request == "optional":
    try:
        subprocess.check_output("command -v nvidia-container-runtime", shell=True)
        print(bash_list("GPU_REQUEST", ["--gpus", "all"]))
    except Exception:
        print(bash_list("GPU_REQUEST", []))

print(
    bash_list(
        "ENTRYPOINTS", [x["entrypoint"] for x in devcontainer_meta if "entrypoint" in x]
    )
)

cap_add = flat_list(
    [
        ["--cap-add", cap]
        for cap in list(
            set(flat_list([x["capAdd"] for x in devcontainer_meta if "capAdd" in x]))
        )
    ]
)

secopts = flat_list(
    [
        ["--security-opt", opt]
        for opt in list(
            set(
                flat_list(
                    [x["securityOpt"] for x in devcontainer_meta if "securityOpt" in x]
                )
            )
        )
    ]
)

print(
    bash_list(
        "RUN_ARGS",
        flat_list([x["runArgs"] for x in devcontainer_meta if "runArgs" in x])
        + cap_add
        + secopts
        + ["--workdir", containerWorkspaceFolder],
    )
)

print(
    bash_list_of_commands(
        "INITIALIZE_COMMANDS",
        [x["initializeCommand"] for x in devcontainer_meta if "initializeCommand" in x],
    )
)

print(
    bash_list(
        "ENV_VARS",
        flat_list(
            [
                ["--env", x]
                for x in dict_list(
                    flat_dict(
                        [
                            x["containerEnv"]
                            for x in devcontainer_meta
                            if "containerEnv" in x
                        ]
                    )
                )
            ]
        ),
    )
)

print(
    bash_list(
        "MOUNTS",
        flat_list(
            [
                ["--mount", normalize_mount(m)]
                for m in flat_list(
                    [x["mounts"] for x in devcontainer_meta if "mounts" in x]
                    + [
                        [devcontainer_json["workspaceMount"]]
                        if "workspaceMount" in devcontainer_json
                        else []
                    ]
                )
            ]
        ),
    )
)
