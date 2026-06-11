import json
import os
import subprocess


def get_alg_base(algname: str):
    return os.path.join(".", "bin", algname + ".base")


class JsonCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.bench_cache = {}
            cls._instance.device_cache = {}
        return cls._instance

    def get_jsonlist(self, algname, listname):
        benchmark_bin = os.path.join(".", "bin", algname + ".base")
        if not os.path.exists(benchmark_bin):
            raise Exception(f"Benchmark binary not found: {benchmark_bin}")
        return subprocess.check_output([benchmark_bin, f"--jsonlist-{listname}"])

    def get_bench(self, algname):
        if algname not in self.bench_cache:
            result = self.get_jsonlist(algname, "benches")
            self.bench_cache[algname] = json.loads(result)
        return self.bench_cache[algname]

    def get_device(self, algname):
        if algname not in self.device_cache:
            result = self.get_jsonlist(algname, "devices")
            data = json.loads(result)
            if "devices" not in data:
                raise Exception(
                    "JSON returned from --jsonlist-devices does not contain 'devices' key"
                )
            devices = data["devices"]
            if len(devices) != 1:
                raise Exception(
                    "NVBench doesn't work well with multiple GPUs, use `CUDA_VISIBLE_DEVICES`"
                )

            self.device_cache[algname] = devices[0]

        return self.device_cache[algname]


def json_benches(algname):
    return JsonCache().get_bench(algname)


def device_json(algname):
    return JsonCache().get_device(algname)
