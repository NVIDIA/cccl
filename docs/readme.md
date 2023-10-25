# Building docs

Most docs can be *prepared* locally, but the build process is partially invested in the github workflows.

## First steps

Prepare docker container for local builds

```bash
make_env.bash $TAG
```

## Build Thrust

```bash
build_docs.bash $TAG /cccl/thrust/docs/build_docs_locally.bash
```

## Build CUB

```bash
build_docs.bash $TAG /cccl/cub/docs/gen_docs.bash
```
