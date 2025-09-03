# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import ctypes
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys

import libcudacxx.util
from libcudacxx.compiler import CXXCompiler

# The wildcard import is to support `eval(exec_str)` in
# `Configuration.configure_executor()` below.
from libcudacxx.test.executor import *  # noqa: F403
from libcudacxx.test.executor import LocalExecutor, NoopExecutor
from libcudacxx.test.target_info import make_target_info


def loadSiteConfig(lit_config, config, param_name, env_name):
    # We haven't loaded the site specific configuration (the user is
    # probably trying to run on a test file directly, and either the site
    # configuration hasn't been created by the build system, or we are in an
    # out-of-tree build situation).
    site_cfg = lit_config.params.get(param_name, os.environ.get(env_name))
    if not site_cfg:
        lit_config.warning(
            "No site specific configuration file found!"
            " Running the tests in the default configuration."
        )
    elif not os.path.isfile(site_cfg):
        lit_config.fatal(
            "Specified site configuration file does not exist: '%s'" % site_cfg
        )
    else:
        lit_config.note("using site specific configuration at %s" % site_cfg)
        ld_fn = lit_config.load_config

        # Null out the load_config function so that lit.site.cfg doesn't
        # recursively load a config even if it tries.
        # TODO: This is one hell of a hack. Fix it.
        def prevent_reload_fn(*args, **kwargs):
            pass

        lit_config.load_config = prevent_reload_fn
        ld_fn(config, site_cfg)
        lit_config.load_config = ld_fn


# Extract the value of a numeric macro such as __cplusplus or a feature-test
# macro.
def intMacroValue(token):
    return int(token.rstrip("LlUu"))


class Configuration(object):
    # pylint: disable=redefined-outer-name
    def __init__(self, lit_config, config):
        self.lit_config = lit_config
        self.config = config
        self.is_windows = platform.system() == "Windows"
        self.cxx = None
        self.cxx_is_clang_cl = None
        self.cxx_stdlib_under_test = None
        self.project_obj_root = None
        self.libcudacxx_src_root = None
        self.libcudacxx_obj_root = None
        self.cxx_library_root = None
        self.cxx_runtime_root = None
        self.abi_library_root = None
        self.link_shared = self.get_lit_bool("enable_shared", default=True)
        self.debug_build = self.get_lit_bool("debug_build", default=False)
        self.exec_env = dict(os.environ)
        self.exec_env["CUDA_MODULE_LOADING"] = "EAGER"
        self.use_target = False
        self.use_system_cxx_lib = False
        self.use_clang_verify = False
        self.long_tests = None
        self.execute_external = False

    def get_lit_conf(self, name, default=None):
        val = self.lit_config.params.get(name, None)
        if val is None:
            val = getattr(self.config, name, None)
            if val is None:
                val = default
        return val

    def get_lit_bool(self, name, default=None, env_var=None):
        def check_value(value, var_name):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if not isinstance(value, str):
                raise TypeError("expected bool or string")
            if value.lower() in ("1", "true"):
                return True
            if value.lower() in ("", "0", "false"):
                return False
            self.lit_config.fatal(
                "parameter '{}' should be true or false".format(var_name)
            )

        conf_val = self.get_lit_conf(name)
        if (
            env_var is not None
            and env_var in os.environ
            and os.environ[env_var] is not None
        ):
            val = os.environ[env_var]
            if conf_val is not None:
                self.lit_config.warning(
                    "Environment variable %s=%s is overriding explicit "
                    "--param=%s=%s" % (env_var, val, name, conf_val)
                )
            return check_value(val, env_var)
        return check_value(conf_val, name)

    def get_compute_capabilities(self):
        deduced_compute_archs = []
        libnames = ("libcuda.so", "libcuda.dylib", "nvcuda.dll", "cuda.dll")
        for libname in libnames:
            try:
                cuda = ctypes.CDLL(libname)
            except OSError:
                continue
            else:
                break
        else:
            raise OSError("could not load any of: " + " ".join(libnames))

        self.lit_config.note('compute_archs set to "native", computing available archs')
        CUDA_SUCCESS = 0
        nGpus = ctypes.c_int()
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()

        result = ctypes.c_int()
        device = ctypes.c_int()
        error_str = ctypes.c_char_p()

        result = cuda.cuInit(0)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            self.lit_config.note(
                "cuInit failed with error code %d: %s"
                % (result, error_str.value.decode())
            )
            return "native"

        result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            self.lit_config.note(
                "cuDeviceGetCount failed with error code %d: %s"
                % (result, error_str.value.decode())
            )
            return "native"
        self.lit_config.note("Found %d device(s)." % nGpus.value)
        for i in range(nGpus.value):
            result = cuda.cuDeviceGet(ctypes.byref(device), i)
            if result != CUDA_SUCCESS:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                self.lit_config.note(
                    "cuDeviceGet failed with error code %d: %s"
                    % (result, error_str.value.decode())
                )
                return "native"
            if (
                cuda.cuDeviceComputeCapability(
                    ctypes.byref(cc_major), ctypes.byref(cc_minor), device
                )
                == CUDA_SUCCESS
            ):
                self.lit_config.note(
                    "Deduced compute capability of device %d to: %d%d"
                    % (i + 1, cc_major.value, cc_minor.value)
                )
                deduced_compute_archs.append(cc_major.value * 10 + cc_minor.value)

        self.lit_config.note(
            "Deduced compute capabilities are: %s" % deduced_compute_archs
        )
        deduced_comput_archs_str = ", ".join(
            [str(element) for element in deduced_compute_archs]
        )
        return deduced_comput_archs_str

    def _get_nvcc_archs(self):
        if self.cxx.type != "nvcc":
            self.lit_config.fatal(
                "Retrieving compute capabilities is only supported for nvcc compiler type"
            )
            return []

        cmd = (
            f"{self.cxx.path} --help | grep -oE 'compute_[0-9]+' | "
            "sed -E 's/compute_//g' | sort -ug"
        )
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        archs = result.stdout.strip().splitlines()

        if not archs:
            self.lit_config.fatal(
                "Failed to retrieve compute capabilities or no capabilities found."
            )
            return []

        return sorted(set(int(arch) for arch in archs))

    def get_all_major_compute_capabilities(self):
        archs = self._get_nvcc_archs()
        if not archs:
            return ""

        # Build the same list used by --arch=all-major:

        # Handle special case where the first architecture is not a round decade (e.g., first arch is 75, not 70).
        oldest = archs[0]
        archs = sorted(set((arch // 10 * 10) for arch in archs))
        archs[0] = oldest
        last_arch = archs[-1]
        archs = [f"{arch}-real" for arch in archs]
        archs.append(f"{last_arch}-virtual")

        archs = ";".join(archs)

        self.lit_config.note("Deduced major compute capabilities are: %s" % archs)

        return archs

    def get_all_compute_capabilities(self):
        archs = self._get_nvcc_archs()
        if not archs:
            return ""
        last_arch = archs[-1]
        archs = [f"{arch}-real" for arch in archs]
        archs.append(f"{last_arch}-virtual")

        archs = ";".join(archs)

        self.lit_config.note("Deduced compute capabilities are: %s" % archs)

        return archs

    def get_modules_enabled(self):
        return self.get_lit_bool(
            "enable_modules", default=False, env_var="LIBCUDACXX_ENABLE_MODULES"
        )

    def make_static_lib_name(self, name):
        """Return the full filename for the specified library name"""
        if self.is_windows:
            # Only allow libc++ to use this function for now.
            assert name == "c++"
            return "lib" + name + ".lib"
        else:
            return "lib" + name + ".a"

    def configure(self):
        self.configure_executor()
        self.configure_use_system_cxx_lib()
        self.configure_target_info()
        self.configure_cxx()
        self.configure_triple()
        self.configure_deployment()
        self.configure_src_root()
        self.configure_obj_root()
        self.configure_cxx_stdlib_under_test()
        self.configure_cxx_library_root()
        self.configure_use_clang_verify()
        self.configure_use_thread_safety()
        self.configure_no_execute()
        self.configure_execute_external()
        self.configure_ccache()
        self.configure_compile_flags()
        self.configure_filesystem_compile_flags()
        self.configure_link_flags()
        self.configure_env()
        self.configure_color_diagnostics()
        if self.cxx.type != "nvrtcc":
            self.configure_debug_mode()
            self.configure_warnings()
            self.configure_sanitizer()
            self.configure_coverage()
            self.configure_modules()
            self.configure_coroutines()
            self.configure_substitutions()
            self.configure_features()

    def print_config_info(self):
        # Print the final compile and link flags.
        self.lit_config.note(
            "Using compiler: %s %s" % (self.cxx.path, self.cxx.first_arg)
        )
        self.lit_config.note("Using flags: %s" % self.cxx.flags)
        if self.cxx.use_modules:
            self.lit_config.note("Using modules flags: %s" % self.cxx.modules_flags)
        self.lit_config.note("Using compile flags: %s" % self.cxx.compile_flags)
        if len(self.cxx.warning_flags):
            self.lit_config.note("Using warnings: %s" % self.cxx.warning_flags)
        self.lit_config.note("Using link flags: %s" % self.cxx.link_flags)
        # Print as list to prevent "set([...])" from being printed.
        self.lit_config.note(
            "Using available_features: %s" % list(self.config.available_features)
        )
        show_env_vars = {}
        for k, v in self.exec_env.items():
            if k not in os.environ or os.environ[k] != v:
                show_env_vars[k] = v
        self.lit_config.note("Adding environment variables: %r" % show_env_vars)
        sys.stderr.flush()  # Force flushing to avoid broken output on Windows

    def get_test_format(self):
        from libcudacxx.test.format import LibcxxTestFormat

        return LibcxxTestFormat(
            self.cxx,
            self.use_clang_verify,
            self.execute_external,
            self.executor,
            exec_env=self.exec_env,
        )

    def configure_executor(self):
        exec_str = self.get_lit_conf("executor", "None")
        exec_timeout = self.get_lit_conf("maxIndividualTestTime", "None")
        te = eval(exec_str)
        if te:
            self.lit_config.note("Using executor: %r" % exec_str)
            if self.lit_config.useValgrind:
                # We have no way of knowing where in the chain the
                # ValgrindExecutor is supposed to go. It is likely
                # that the user wants it at the end, but we have no
                # way of getting at that easily.
                self.lit_config.fatal(
                    "Cannot infer how to create a Valgrind  executor."
                )
        else:
            te = LocalExecutor()
            te.timeout = 0
            if exec_timeout:
                te.timeout = exec_timeout
            if self.lit_config.useValgrind:
                # te = ValgrindExecutor(self.lit_config.valgrindArgs, te)
                self.lit_config.fatal("ValgrindExecutor never existed in CCCL.")
        self.executor = te

    def configure_target_info(self):
        self.target_info = make_target_info(self)

    def configure_cxx(self):
        # Gather various compiler parameters.
        cxx = self.get_lit_conf("cxx_under_test")
        cxx_first_arg = self.get_lit_conf("cxx_first_arg")
        nvrtc = self.get_lit_bool("is_nvrtc", False)

        self.cxx_is_clang_cl = (
            cxx is not None and os.path.basename(cxx) == "clang-cl.exe"
        )

        # Build CXXCompiler manually for NVRTCC
        if nvrtc is True:
            cxx_type = "nvrtcc"
            self.cxx = CXXCompiler(
                path=cxx,
                first_arg=cxx_first_arg,
                cxx_type=cxx_type,
                cxx_version=("1", "1", "1"),
            )

            self.cxx.default_dialect = "c++11"
            self.cxx.source_lang = "cu"
            maj_v, min_v, patch_v = self.cxx.version
            self.config.available_features.add("nvrtc")
            self.config.available_features.add("%s-%s" % (self.cxx.type, maj_v))
            self.config.available_features.add(
                "%s-%s.%s" % (self.cxx.type, maj_v, min_v)
            )
            self.config.available_features.add(
                "%s-%s.%s.%s" % (self.cxx.type, maj_v, min_v, patch_v)
            )
            self.lit_config.note("detected cxx.type as: {}".format(self.cxx.type))
            self.lit_config.note("detected cxx.version as: {}".format(self.cxx.version))
            self.lit_config.note(
                "detected cxx.default_dialect as: {}".format(self.cxx.default_dialect)
            )
            self.cxx.compile_env = dict(os.environ)
        # If compiler is *not* NVRTCC
        else:
            # If no specific cxx_under_test was given, attempt to infer it as
            # clang++.
            if cxx is None or self.cxx_is_clang_cl:
                search_paths = self.config.environment["PATH"]
                if cxx is not None and os.path.isabs(cxx):
                    search_paths = os.path.dirname(cxx)
                clangxx = libcudacxx.util.which("clang++", search_paths)
                if clangxx:
                    cxx = clangxx
                    self.lit_config.note("inferred cxx_under_test as: %r" % cxx)
                elif self.cxx_is_clang_cl:
                    self.lit_config.fatal(
                        "Failed to find clang++ substitution for clang-cl"
                    )
            if not cxx:
                self.lit_config.fatal(
                    "must specify user parameter cxx_under_test "
                    "(e.g., --param=cxx_under_test=clang++)"
                )
            if self.cxx_is_clang_cl:
                self.cxx = self._configure_clang_cl(cxx)
            else:
                self.cxx = CXXCompiler(
                    cxx,
                    cxx_first_arg,
                    compile_flags=self.get_lit_conf("cmake_cxx_flags"),
                    cuda_path=self.get_lit_conf("cuda_path"),
                )
            cxx_type = self.cxx.type
            if cxx_type is not None:
                assert self.cxx.version is not None
                maj_v, min_v, patch_v = self.cxx.version
                self.config.available_features.add(cxx_type)
                self.config.available_features.add("%s-%s" % (cxx_type, maj_v))
                self.config.available_features.add(
                    "%s-%s.%s" % (cxx_type, maj_v, min_v)
                )
                self.config.available_features.add(
                    "%s-%s.%s.%s" % (cxx_type, maj_v, min_v, patch_v)
                )
            self.lit_config.note("detected cxx.type as: {}".format(self.cxx.type))
            self.lit_config.note("detected cxx.version as: {}".format(self.cxx.version))
            self.lit_config.note(
                "detected cxx.default_dialect as: {}".format(self.cxx.default_dialect)
            )
            self.cxx.compile_env = dict(os.environ)
            # 'CCACHE_CPP2' prevents ccache from stripping comments while
            # preprocessing. This is required to prevent stripping of '-verify'
            # comments.
            self.cxx.compile_env["CCACHE_CPP2"] = "1"

            if self.cxx.type == "nvcc":
                nvcc_host_compiler = self.get_lit_conf("nvcc_host_compiler")
                if len(nvcc_host_compiler.strip()) == 0:
                    if platform.system() == "Darwin":
                        nvcc_host_compiler = "clang"
                    elif platform.system() == "Windows":
                        nvcc_host_compiler = "cl.exe"
                    else:
                        nvcc_host_compiler = "gcc"

                self.cxx.host_cxx = CXXCompiler(nvcc_host_compiler, None)
                self.host_cxx_type = self.cxx.host_cxx.type
                if self.host_cxx_type is not None:
                    assert self.cxx.host_cxx.version is not None
                    maj_v, min_v, _ = self.cxx.host_cxx.version
                    self.config.available_features.add(self.host_cxx_type)
                    self.config.available_features.add(
                        "%s-%s" % (self.host_cxx_type, maj_v)
                    )
                    self.config.available_features.add(
                        "%s-%s.%s" % (self.host_cxx_type, maj_v, min_v)
                    )
                self.lit_config.note(
                    "detected host_cxx.type as: {}".format(self.cxx.host_cxx.type)
                )
                self.lit_config.note(
                    "detected host_cxx.version as: {}".format(self.cxx.host_cxx.version)
                )
                self.lit_config.note(
                    "detected host_cxx.default_dialect as: {}".format(
                        self.cxx.host_cxx.default_dialect
                    )
                )

    def _configure_clang_cl(self, clang_path):
        def _split_env_var(var):
            return [p.strip() for p in os.environ.get(var, "").split(";") if p.strip()]

        def _prefixed_env_list(var, prefix):
            from itertools import chain

            return list(
                chain.from_iterable((prefix, path) for path in _split_env_var(var))
            )

        assert self.cxx_is_clang_cl
        flags = []
        compile_flags = _prefixed_env_list("INCLUDE", "-isystem")
        link_flags = _prefixed_env_list("LIB", "-L")
        for path in _split_env_var("LIB"):
            self.add_path(self.exec_env, path)
        return CXXCompiler(
            clang_path, flags=flags, compile_flags=compile_flags, link_flags=link_flags
        )

    def _dump_macros_verbose(self, *args, **kwargs):
        if self.cxx.type == "nvrtcc":
            return None
        macros_or_error = self.cxx.dumpMacros(*args, **kwargs)
        if isinstance(macros_or_error, tuple):
            cmd, out, err, rc = macros_or_error
            report = libcudacxx.util.makeReport(cmd, out, err, rc)
            report += "Compiler failed unexpectedly when dumping macros!"
            self.lit_config.fatal(report)
            return None
        assert isinstance(macros_or_error, dict)
        return macros_or_error

    def configure_src_root(self):
        self.libcudacxx_src_root = self.get_lit_conf(
            "libcudacxx_src_root", os.path.dirname(self.config.test_source_root)
        )

    def configure_obj_root(self):
        self.project_obj_root = self.get_lit_conf("project_obj_root")
        self.libcudacxx_obj_root = self.get_lit_conf("libcudacxx_obj_root")
        if not self.libcudacxx_obj_root and self.project_obj_root is not None:
            possible_roots = [
                os.path.join(self.project_obj_root, "libcudacxx"),
                os.path.join(self.project_obj_root, "projects", "libcudacxx"),
                os.path.join(self.project_obj_root, "runtimes", "libcudacxx"),
            ]
            for possible_root in possible_roots:
                if os.path.isdir(possible_root):
                    self.libcudacxx_obj_root = possible_root
                    break
            else:
                self.libcudacxx_obj_root = self.project_obj_root

    def configure_cxx_library_root(self):
        self.cxx_library_root = self.get_lit_conf(
            "cxx_library_root", self.libcudacxx_obj_root
        )
        self.cxx_runtime_root = self.get_lit_conf(
            "cxx_runtime_root", self.cxx_library_root
        )

    def configure_use_system_cxx_lib(self):
        # This test suite supports testing against either the system library or
        # the locally built one; the former mode is useful for testing ABI
        # compatibility between the current headers and a shipping dynamic
        # library.
        # Default to testing against the locally built libc++ library.
        self.use_system_cxx_lib = self.get_lit_conf("use_system_cxx_lib")
        if self.use_system_cxx_lib == "true":
            self.use_system_cxx_lib = True
        elif self.use_system_cxx_lib == "false":
            self.use_system_cxx_lib = False
        elif self.use_system_cxx_lib:
            assert os.path.isdir(self.use_system_cxx_lib), (
                "the specified use_system_cxx_lib parameter (%s) is not a valid directory"
                % self.use_system_cxx_lib
            )
            self.use_system_cxx_lib = os.path.abspath(self.use_system_cxx_lib)
        self.lit_config.note(
            "inferred use_system_cxx_lib as: %r" % self.use_system_cxx_lib
        )

    def configure_cxx_stdlib_under_test(self):
        self.cxx_stdlib_under_test = self.get_lit_conf(
            "cxx_stdlib_under_test", "libc++"
        )
        if self.cxx_stdlib_under_test not in [
            "libc++",
            "libstdc++",
            "msvc",
            "cxx_default",
        ]:
            self.lit_config.fatal(
                'unsupported value for "cxx_stdlib_under_test": %s'
                % self.cxx_stdlib_under_test
            )
        self.config.available_features.add(self.cxx_stdlib_under_test)
        if self.cxx_stdlib_under_test == "libstdc++":
            self.config.available_features.add("libstdc++")
            # Manually enable the experimental and filesystem tests for libstdc++
            # if the options aren't present.
            # FIXME this is a hack.
            if self.get_lit_conf("enable_experimental") is None:
                self.config.enable_experimental = "true"

    def configure_use_clang_verify(self):
        if self.cxx.type == "nvrtcc":
            return
        """If set, run clang with -verify on failing tests."""
        self.use_clang_verify = self.get_lit_bool("use_clang_verify")
        if self.use_clang_verify is None:
            # NOTE: We do not test for the -verify flag directly because
            #   -verify will always exit with non-zero on an empty file.
            self.use_clang_verify = self.cxx.isVerifySupported()
            self.lit_config.note(
                "inferred use_clang_verify as: %r" % self.use_clang_verify
            )
        if self.use_clang_verify:
            self.config.available_features.add("verify-support")

    def configure_use_thread_safety(self):
        """If set, run clang with -verify on failing tests."""
        has_thread_safety = self.cxx.hasCompileFlag("-Werror=thread-safety")
        if has_thread_safety:
            self.cxx.compile_flags += ["-Werror=thread-safety"]
            self.config.available_features.add("thread-safety")
            self.lit_config.note("enabling thread-safety annotations")

    def configure_execute_external(self):
        # Choose between lit's internal shell pipeline runner and a real shell.
        # If LIT_USE_INTERNAL_SHELL is in the environment, we use that as the
        # default value. Otherwise we ask the target_info.
        use_lit_shell_default = os.environ.get("LIT_USE_INTERNAL_SHELL")
        if use_lit_shell_default is not None:
            use_lit_shell_default = use_lit_shell_default != "0"
        else:
            use_lit_shell_default = self.target_info.use_lit_shell_default()
        # Check for the command line parameter using the default value if it is
        # not present.
        use_lit_shell = self.get_lit_bool("use_lit_shell", use_lit_shell_default)
        self.execute_external = not use_lit_shell

    def configure_no_execute(self):
        if isinstance(self.executor, NoopExecutor):
            self.config.available_features.add("no_execute")

    def configure_ccache(self):
        use_ccache_default = os.environ.get("CMAKE_CUDA_COMPILER_LAUNCHER") is not None
        use_ccache = self.get_lit_bool("use_ccache", use_ccache_default)
        if use_ccache and not self.cxx.type == "nvrtcc":
            self.cxx.use_ccache = True
            self.lit_config.note("enabling ccache")

    def add_deployment_feature(self, feature):
        (arch, name, version) = self.config.deployment
        self.config.available_features.add("%s=%s-%s" % (feature, arch, name))
        self.config.available_features.add("%s=%s" % (feature, name))
        self.config.available_features.add("%s=%s%s" % (feature, name, version))

    def configure_features(self):
        additional_features = self.get_lit_conf("additional_features")
        if additional_features:
            for f in additional_features.split(","):
                self.config.available_features.add(f.strip())
        self.target_info.add_locale_features(self.config.available_features)

        target_platform = self.target_info.platform()

        # Write an "available feature" that combines the triple when
        # use_system_cxx_lib is enabled. This is so that we can easily write
        # XFAIL markers for tests that are known to fail with versions of
        # libc++ as were shipped with a particular triple.
        if self.use_system_cxx_lib:
            self.config.available_features.add("with_system_cxx_lib")
            self.config.available_features.add(
                "with_system_cxx_lib=%s" % self.config.target_triple
            )

            # Add subcomponents individually.
            target_components = self.config.target_triple.split("-")
            for component in target_components:
                self.config.available_features.add("with_system_cxx_lib=%s" % component)

            # Add available features for more generic versions of the target
            # triple attached to  with_system_cxx_lib.
            if self.use_deployment:
                self.add_deployment_feature("with_system_cxx_lib")

        # Configure the availability feature. Availability is only enabled
        # with libc++, because other standard libraries do not provide
        # availability markup.
        if self.use_deployment and self.cxx_stdlib_under_test == "libc++":
            self.config.available_features.add("availability")
            self.add_deployment_feature("availability")

        if platform.system() == "Darwin":
            self.config.available_features.add("apple-darwin")

        # Insert the platform name into the available features as a lower case.
        self.config.available_features.add(target_platform)

        # Simulator testing can take a really long time for some of these tests
        # so add a feature check so we can REQUIRES: long_tests in them
        self.long_tests = self.get_lit_bool("long_tests")
        if self.long_tests is None:
            # Default to running long tests.
            self.long_tests = True
            self.lit_config.note("inferred long_tests as: %r" % self.long_tests)

        if self.long_tests:
            self.config.available_features.add("long_tests")

        if not self.get_lit_bool("enable_filesystem", default=True):
            self.config.available_features.add("c++filesystem-disabled")
            self.config.available_features.add("dylib-has-no-filesystem")

        # Run a compile test for the -fsized-deallocation flag. This is needed
        # in test/std/language.support/support.dynamic/new.delete
        if self.cxx.hasCompileFlag("-fsized-deallocation"):
            self.config.available_features.add("-fsized-deallocation")

        if self.cxx.hasCompileFlag("-faligned-allocation"):
            self.config.available_features.add("-faligned-allocation")
        else:
            # FIXME remove this once more than just clang-4.0 support
            # C++17 aligned allocation.
            self.config.available_features.add("no-aligned-allocation")

        if self.cxx.hasCompileFlag("-fdelayed-template-parsing"):
            self.config.available_features.add("fdelayed-template-parsing")

        if self.get_lit_bool("has_libatomic", False):
            self.config.available_features.add("libatomic")

        if "msvc" not in self.config.available_features:
            macros = self._dump_macros_verbose()
            if "__cpp_if_constexpr" not in macros:
                self.config.available_features.add("libcpp-no-if-constexpr")

            if "__cpp_structured_bindings" not in macros:
                self.config.available_features.add("libcpp-no-structured-bindings")

            if (
                "__cpp_deduction_guides" not in macros
                or intMacroValue(macros["__cpp_deduction_guides"]) < 201611
            ):
                self.config.available_features.add("libcpp-no-deduction-guides")

        if self.is_windows:
            self.config.available_features.add("windows")
            if self.cxx_stdlib_under_test == "libc++":
                # LIBCXX-WINDOWS-FIXME is the feature name used to XFAIL the
                # initial Windows failures until they can be properly diagnosed
                # and fixed. This allows easier detection of new test failures
                # and regressions. Note: New failures should not be suppressed
                # using this feature. (Also see llvm.org/PR32730)
                self.config.available_features.add("LIBCUDACXX-WINDOWS-FIXME")

        if "msvc" not in self.config.available_features:
            # Attempt to detect the glibc version by querying for __GLIBC__
            # in 'features.h'.
            macros = self.cxx.dumpMacros(flags=["-include", "features.h"])
            if isinstance(macros, dict) and "__GLIBC__" in macros:
                maj_v, min_v = (macros["__GLIBC__"], macros["__GLIBC_MINOR__"])
                self.config.available_features.add("glibc")
                self.config.available_features.add("glibc-%s" % maj_v)
                self.config.available_features.add("glibc-%s.%s" % (maj_v, min_v))

        libcudacxx_gdb = self.get_lit_conf("libcudacxx_gdb")
        if libcudacxx_gdb and "NOTFOUND" not in libcudacxx_gdb:
            self.config.available_features.add("libcudacxx_gdb")
            self.cxx.libcudacxx_gdb = libcudacxx_gdb

    def configure_compile_flags(self):
        self.configure_default_compile_flags()
        # Configure extra flags
        compile_flags_str = self.get_lit_conf("compile_flags", "")
        self.cxx.compile_flags += shlex.split(compile_flags_str)
        self.cxx.compile_flags += ["-D_CCCL_NO_SYSTEM_HEADER"]
        if self.is_windows:
            # FIXME: Can we remove this?
            self.cxx.compile_flags += ["-D_CRT_SECURE_NO_WARNINGS"]
            self.cxx.compile_flags += ["--use-local-env"]
            # Required so that tests using min/max don't fail on Windows,
            # and so that those tests don't have to be changed to tolerate
            # this insanity.
            self.cxx.compile_flags += ["-DNOMINMAX"]
            if "msvc" in self.config.available_features:
                if self.cxx.type == "nvcc":
                    self.cxx.compile_flags += ["-Xcompiler"]
                self.cxx.compile_flags += ["/bigobj"]
        additional_flags = self.get_lit_conf("test_compiler_flags")
        if additional_flags:
            self.cxx.compile_flags += shlex.split(additional_flags)
        compute_archs = self.get_lit_conf("compute_archs")
        if self.cxx.type == "nvrtcc":
            self.config.available_features.add("nvrtc")
        if self.cxx.type == "nvcc":
            self.cxx.compile_flags += ["--extended-lambda"]
        real_arch_format = "-gencode=arch=compute_{0},code=sm_{0}"
        virt_arch_format = "-gencode=arch=compute_{0},code=compute_{0}"
        if self.cxx.type == "clang":
            real_arch_format = "--cuda-gpu-arch=sm_{0}"
            virt_arch_format = "--cuda-gpu-arch=compute_{0}"
            self.cxx.compile_flags += ["-O1"]
        pre_sm_32 = True
        pre_sm_60 = True
        pre_sm_70 = True
        pre_sm_80 = True
        pre_sm_90 = True
        pre_sm_90a = True
        if compute_archs and (
            self.cxx.type == "nvcc"
            or self.cxx.type == "clang"
            or self.cxx.type == "nvrtcc"
        ):
            pre_sm_32 = False
            pre_sm_60 = False
            pre_sm_70 = False
            pre_sm_80 = False
            pre_sm_90 = False
            pre_sm_90a = False

            self.lit_config.note("Compute Archs: %s" % compute_archs)
            if compute_archs == "native":
                compute_archs = self.get_compute_capabilities()
            elif compute_archs == "all":
                compute_archs = self.get_all_compute_capabilities()
            elif compute_archs == "all-major":
                compute_archs = self.get_all_major_compute_capabilities()

            compute_archs = sorted(set(re.split("\\s|;|,", compute_archs)))
            for s in compute_archs:
                # Split arch and mode i.e. 80-virtual -> 80, virtual
                arch, *mode = re.split("-", s)

                # With Hopper there are new subarchitectures like 90a we need to handle those
                subarchitecture = ""
                if not arch.isnumeric():
                    subarchitecture = arch[-1]
                    arch = arch[:-1]
                arch = int(arch)
                if arch < 32:
                    pre_sm_32 = True
                if arch < 60:
                    pre_sm_60 = True
                if arch < 70:
                    pre_sm_70 = True
                if arch < 80:
                    pre_sm_80 = True
                if arch < 90:
                    pre_sm_90 = True
                if arch < 90 or (arch == 90 and subarchitecture < "a"):
                    pre_sm_90a = True
                arch_flag = real_arch_format.format(str(arch) + subarchitecture)
                if mode.count("virtual"):
                    arch_flag = virt_arch_format.format(str(arch) + subarchitecture)
                self.cxx.compile_flags += [arch_flag]
        if pre_sm_32:
            self.config.available_features.add("pre-sm-32")
        if pre_sm_60:
            self.config.available_features.add("pre-sm-60")
        if pre_sm_70:
            self.config.available_features.add("pre-sm-70")
        if pre_sm_80:
            self.config.available_features.add("pre-sm-80")
        if pre_sm_90:
            self.config.available_features.add("pre-sm-90")
        if pre_sm_90a:
            self.config.available_features.add("pre-sm-90a")

    def configure_default_compile_flags(self):
        nvcc_host_compiler = self.get_lit_conf("nvcc_host_compiler")

        if nvcc_host_compiler and self.cxx.type == "nvcc":
            self.cxx.compile_flags += ["-ccbin={0}".format(nvcc_host_compiler)]

        # Try and get the std version from the command line. Fall back to
        # default given in lit.site.cfg is not present. If default is not
        # present then force c++11.
        std = self.get_lit_conf("std")
        if not std:
            # Choose the newest possible language dialect if none is given.
            possible_stds = [
                "c++20",
                "c++2a",
                "c++17",
                "c++1z",
                "c++14",
                "c++11",
                "c++03",
            ]
            if self.cxx.type == "gcc":
                maj_v, _, _ = self.cxx.version
                maj_v = int(maj_v)
                if maj_v < 6:
                    possible_stds.remove("c++1z")
                    possible_stds.remove("c++17")
                # FIXME: How many C++14 tests actually fail under GCC 5 and 6?
                # Should we XFAIL them individually instead?
                if maj_v < 6:
                    possible_stds.remove("c++14")
            for s in possible_stds:
                cxx = self.cxx
                success = True

                if self.cxx.type == "nvcc":
                    # NVCC warns, but doesn't error, if the host compiler
                    # doesn't support the dialect. It's also possible that the
                    # host compiler supports the dialect, but NVCC doesn't.

                    # So, first we need to check if NVCC supports the dialect...
                    if not self.cxx.hasCompileFlag("-std=%s" % s):
                        # If it doesn't, give up on this dialect.
                        success = False

                    # ... then we need to check if host compiler supports the
                    # dialect.
                    cxx = self.cxx.host_cxx

                if cxx.type == "msvc":
                    if not cxx.hasCompileFlag("/std:%s" % s):
                        success = False
                else:
                    if not cxx.hasCompileFlag("-std=%s" % s):
                        success = False

                if success:
                    std = s
                    self.lit_config.note("inferred language dialect as: %s" % std)
                    break

        if std:
            # We found a dialect flag.
            stdflag = "-std={0}".format(std)
            if self.cxx.type == "msvc":
                stdflag = "/std:{0}".format(std)

            extraflags = []
            if self.cxx.type == "clang":
                extraflags = ["-Wno-unknown-cuda-version", "--no-cuda-version-check"]

            # Do a check with the user/config flag to ensure that the flag is supported.
            if not self.cxx.hasCompileFlag([stdflag] + extraflags):
                raise OSError(
                    "Configured compiler does not support flag {0}".format(stdflag)
                )

            self.cxx.flags += [stdflag]

        if not std:
            # There is no dialect flag. This happens with older MSVC.
            if self.cxx.type == "nvcc":
                std = self.cxx.host_cxx.default_dialect
            else:
                std = self.cxx.default_dialect
            self.lit_config.note("using default language dialect: %s" % std)

        std_feature = std.replace("gnu++", "c++")
        std_feature = std.replace("1z", "17")
        std_feature = std.replace("2a", "20")
        self.config.available_features.add(std_feature)
        # Configure include paths
        self.configure_compile_flags_header_includes()
        self.target_info.add_cxx_compile_flags(self.cxx.compile_flags)
        # Configure feature flags.
        self.configure_compile_flags_exceptions()
        self.configure_compile_flags_rtti()
        self.configure_compile_flags_abi_version()
        enable_32bit = self.get_lit_bool("enable_32bit", False)
        if enable_32bit:
            self.cxx.flags += ["-m32"]
        # Use verbose output for better errors
        if not self.cxx.use_ccache or self.cxx.type == "msvc":
            self.cxx.flags += ["-v"]
        sysroot = self.get_lit_conf("sysroot")
        if sysroot:
            self.cxx.flags += ["--sysroot=" + sysroot]
        gcc_toolchain = self.get_lit_conf("gcc_toolchain")
        if gcc_toolchain:
            self.cxx.flags += ["--gcc-toolchain=" + gcc_toolchain]
        # NOTE: the _DEBUG definition must precede the triple check because for
        # the Windows build of libc++, the forced inclusion of a header requires
        # that _DEBUG is defined.  Incorrect ordering will result in -target
        # being elided.
        if self.is_windows and self.debug_build:
            self.cxx.compile_flags += ["-D_DEBUG"]
        if self.use_target:
            if not self.cxx.addFlagIfSupported(
                ["--target=" + self.config.target_triple]
            ):
                self.lit_config.warning(
                    "use_target is true but --target is not supported by the compiler"
                )
        if self.use_deployment:
            arch, name, version = self.config.deployment
            self.cxx.flags += ["-arch", arch]
            self.cxx.flags += ["-m" + name + "-version-min=" + version]

        # Add includes for support headers used in the tests.
        support_path = os.path.join(self.libcudacxx_src_root, "test", "support")
        self.cxx.compile_flags += ["-I" + support_path]

        # Add includes for the PSTL headers
        pstl_src_root = self.get_lit_conf("pstl_src_root")
        pstl_obj_root = self.get_lit_conf("pstl_obj_root")
        if pstl_src_root is not None and pstl_obj_root is not None:
            self.cxx.compile_flags += ["-I" + os.path.join(pstl_src_root, "include")]
            self.cxx.compile_flags += [
                "-I" + os.path.join(pstl_obj_root, "generated_headers")
            ]
            self.cxx.compile_flags += ["-I" + os.path.join(pstl_src_root, "test")]
            self.config.available_features.add("parallel-algorithms")

        # FIXME(EricWF): variant_size.pass.cpp requires a slightly larger
        # template depth with older Clang versions.
        self.cxx.addFlagIfSupported("-ftemplate-depth=270")

        # If running without execution we need to mark tests that only fail at runtime as unsupported
        if self.lit_config.noExecute:
            self.config.available_features.add("no_execute")

    def configure_compile_flags_header_includes(self):
        support_path = os.path.join(self.libcudacxx_src_root, "test", "support")
        self.configure_config_site_header()
        if self.cxx_stdlib_under_test != "libstdc++" and not self.is_windows:
            self.cxx.compile_flags += [
                "-include",
                os.path.join(support_path, "nasty_macros.h"),
            ]
        if self.cxx_stdlib_under_test == "msvc":
            self.cxx.compile_flags += [
                "-include",
                os.path.join(support_path, "msvc_stdlib_force_include.h"),
            ]
            pass
        if (
            self.is_windows
            and self.debug_build
            and self.cxx_stdlib_under_test != "msvc"
        ):
            self.cxx.compile_flags += [
                "-include",
                os.path.join(support_path, "set_windows_crt_report_mode.h"),
            ]
        cxx_headers = self.get_lit_conf("cxx_headers")
        if cxx_headers == "" or (
            cxx_headers is None and self.cxx_stdlib_under_test != "libc++"
        ):
            self.lit_config.note("using the system cxx headers")
            return
        # I don't think this is required, since removing it helps clang-cuda compile and libcudacxx only supports building in CUDA modes?
        # if self.cxx.type != 'nvcc' and self.cxx.type != 'pgi':
        #    self.cxx.compile_flags += ['-nostdinc++']
        if cxx_headers is None:
            cxx_headers = os.path.join(self.libcudacxx_src_root, "include")
        if not os.path.isdir(cxx_headers):
            self.lit_config.fatal("cxx_headers='%s' is not a directory." % cxx_headers)
        self.cxx.compile_flags += ["-I" + cxx_headers]
        if self.libcudacxx_obj_root is not None:
            cxxabi_headers = os.path.join(
                self.libcudacxx_obj_root, "include", "c++build"
            )
            if os.path.isdir(cxxabi_headers):
                self.cxx.compile_flags += ["-I" + cxxabi_headers]

    def configure_config_site_header(self):
        # Check for a possible __config_site in the build directory. We
        # use this if it exists.
        if self.libcudacxx_obj_root is None:
            return
        config_site_header = os.path.join(self.libcudacxx_obj_root, "__config_site")
        if not os.path.isfile(config_site_header):
            return
        contained_macros = self.parse_config_site_and_add_features(config_site_header)
        self.lit_config.note(
            "Using __config_site header %s with macros: %r"
            % (config_site_header, contained_macros)
        )
        # FIXME: This must come after the call to
        # 'parse_config_site_and_add_features(...)' in order for it to work.
        self.cxx.compile_flags += ["-include", config_site_header]

    def parse_config_site_and_add_features(self, header):
        """parse_config_site_and_add_features - Deduce and add the test
        features that that are implied by the #define's in the __config_site
        header. Return a dictionary containing the macros found in the
        '__config_site' header.
        """
        # MSVC can't dump macros, so we just give up.
        if "msvc" in self.config.available_features:
            return {}
        # Parse the macro contents of __config_site by dumping the macros
        # using 'c++ -dM -E' and filtering the predefines.
        predefines = self._dump_macros_verbose()
        macros = self._dump_macros_verbose(header)
        feature_macros_keys = set(macros.keys()) - set(predefines.keys())
        feature_macros = {}
        for k in feature_macros_keys:
            feature_macros[k] = macros[k]
        # We expect the header guard to be one of the definitions
        assert "_LIBCUDACXX_CONFIG_SITE" in feature_macros
        del feature_macros["_LIBCUDACXX_CONFIG_SITE"]
        # The __config_site header should be non-empty. Otherwise it should
        # have never been emitted by CMake.
        assert len(feature_macros) > 0
        # FIXME: This is a hack that should be fixed using module maps.
        # If modules are enabled then we have to lift all of the definitions
        # in __config_site onto the command line.
        for m in feature_macros:
            define = "-D%s" % m
            if feature_macros[m]:
                define += "=%s" % (feature_macros[m])
            self.cxx.modules_flags += [define]
        if self.cxx.hasCompileFlag("-Wno-macro-redefined"):
            self.cxx.compile_flags += ["-Wno-macro-redefined"]
        # Transform each macro name into the feature name used in the tests.
        # Ex. _LIBCUDACXX_HAS_NO_THREADS -> libcpp-has-no-threads
        for m in feature_macros:
            assert m.startswith("_LIBCUDACXX_HAS_") or m.startswith("_LIBCUDACXX_ABI_")
            m = m.lower()[1:].replace("_", "-")
            self.config.available_features.add(m)
        return feature_macros

    def configure_compile_flags_exceptions(self):
        enable_exceptions = self.get_lit_bool("enable_exceptions", True)
        nvrtc = self.get_lit_bool("is_nvrtc", False)

        if not enable_exceptions:
            self.config.available_features.add("libcpp-no-exceptions")
            if "nvhpc" in self.config.available_features:
                # NVHPC reports all expressions as `noexcept(true)` with its
                # "no exceptions" mode. Override the setting from CMake as
                # a temporary workaround for that.
                pass
            # TODO: I don't know how to shut off exceptions with MSVC.
            elif "msvc" not in self.config.available_features:
                if self.cxx.type == "nvcc":
                    self.cxx.compile_flags += ["-Xcompiler"]
                self.cxx.compile_flags += ["-fno-exceptions"]
        elif nvrtc:
            self.config.available_features.add("libcpp-no-exceptions")

    def configure_compile_flags_rtti(self):
        enable_rtti = self.get_lit_bool("enable_rtti", True)
        if not enable_rtti:
            self.config.available_features.add("libcpp-no-rtti")
            if self.cxx.type == "nvcc":
                self.cxx.compile_flags += ["-Xcompiler"]
            if "nvhpc" in self.config.available_features:
                self.cxx.compile_flags += ["--no_rtti"]
            elif "msvc" in self.config.available_features:
                self.cxx.compile_flags += ["/GR-"]
                self.cxx.compile_flags += ["-D_SILENCE_CXX20_CISO646_REMOVED_WARNING"]
            else:
                self.cxx.compile_flags += ["-fno-rtti"]

    def configure_compile_flags_abi_version(self):
        abi_unstable = self.get_lit_bool("abi_unstable")
        if abi_unstable:
            self.config.available_features.add("libcpp-abi-unstable")
            self.cxx.compile_flags += ["-D_LIBCUDACXX_ABI_UNSTABLE"]

    def configure_filesystem_compile_flags(self):
        if not self.get_lit_bool("enable_filesystem", default=True):
            return

        static_env = os.path.join(
            self.libcudacxx_src_root,
            "test",
            "libcudacxx",
            "std",
            "input.output",
            "filesystems",
            "Inputs",
            "static_test_env",
        )
        static_env = os.path.realpath(static_env)
        assert os.path.isdir(static_env)
        self.cxx.compile_flags += [
            '-DLIBCXX_FILESYSTEM_STATIC_TEST_ROOT="%s"' % static_env
        ]

        dynamic_env = os.path.join(
            self.config.test_exec_root, "filesystem", "Output", "dynamic_env"
        )
        dynamic_env = os.path.realpath(dynamic_env)
        if not os.path.isdir(dynamic_env):
            os.makedirs(dynamic_env)
        self.cxx.compile_flags += [
            '-DLIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT="%s"' % dynamic_env
        ]
        self.exec_env["LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT"] = "%s" % dynamic_env

        dynamic_helper = os.path.join(
            self.libcudacxx_src_root,
            "test",
            "support",
            "filesystem_dynamic_test_helper.py",
        )
        assert os.path.isfile(dynamic_helper)

        self.cxx.compile_flags += [
            '-DLIBCXX_FILESYSTEM_DYNAMIC_TEST_HELPER="%s %s"'
            % (sys.executable, dynamic_helper)
        ]

    def configure_link_flags(self):
        nvcc_host_compiler = self.get_lit_conf("nvcc_host_compiler")
        if nvcc_host_compiler and self.cxx.type == "nvcc":
            self.cxx.link_flags += ["-ccbin={0}".format(nvcc_host_compiler)]

        if self.is_windows:
            self.cxx.link_flags += ["--use-local-env"]

        # Configure library path
        self.configure_link_flags_cxx_library_path()
        self.configure_link_flags_abi_library_path()

        # Configure libraries
        if self.cxx_stdlib_under_test == "libc++":
            if self.get_lit_conf("name") != "libcu++":
                if (
                    "nvhpc" not in self.config.available_features
                    or not self.cxx.is_nvrtc
                ):
                    if self.cxx.type == "nvcc":
                        self.cxx.link_flags += ["-Xcompiler"]
                    self.cxx.link_flags += ["-nodefaultlibs"]

                    # FIXME: Handle MSVCRT as part of the ABI library handling.
                    if self.is_windows and "msvc" not in self.config.available_features:
                        if self.cxx.type == "nvcc":
                            self.cxx.link_flags += ["-Xcompiler"]
                        self.cxx.link_flags += ["-nostdlib"]
            self.configure_link_flags_cxx_library()
            self.configure_link_flags_abi_library()
            self.configure_extra_library_flags()
        elif self.cxx_stdlib_under_test == "libstdc++":
            self.config.available_features.add("c++experimental")
            self.cxx.link_flags += ["-lstdc++fs", "-lm", "-pthread"]
        elif self.cxx_stdlib_under_test == "msvc":
            # FIXME: Correctly setup debug/release flags here.
            pass
        elif self.cxx_stdlib_under_test == "cxx_default":
            self.cxx.link_flags += ["-pthread"]
        else:
            self.lit_config.fatal("invalid stdlib under test")

        link_flags_str = self.get_lit_conf("link_flags", "")
        self.cxx.link_flags += shlex.split(link_flags_str)

    def configure_link_flags_cxx_library_path(self):
        if not self.use_system_cxx_lib:
            if self.cxx_library_root:
                self.cxx.link_flags += ["-L" + self.cxx_library_root]
                if self.is_windows and self.link_shared:
                    self.add_path(self.cxx.compile_env, self.cxx_library_root)
            if self.cxx_runtime_root:
                if not self.is_windows:
                    if self.cxx.type == "nvcc":
                        self.cxx.link_flags += [
                            "-Xcompiler",
                            '"-Wl,-rpath,' + self.cxx_runtime_root + '"',
                        ]
                    else:
                        self.cxx.link_flags += ["-Wl,-rpath," + self.cxx_runtime_root]
                elif self.is_windows and self.link_shared:
                    self.add_path(self.exec_env, self.cxx_runtime_root)
        elif os.path.isdir(str(self.use_system_cxx_lib)):
            self.cxx.link_flags += ["-L" + self.use_system_cxx_lib]
            if not self.is_windows:
                if self.cxx.type == "nvcc":
                    self.cxx.link_flags += [
                        "-Xcompiler",
                        '"-Wl,-rpath,' + self.cxx_runtime_root + '"',
                    ]
                else:
                    self.cxx.link_flags += ["-Wl,-rpath," + self.use_system_cxx_lib]
            if self.is_windows and self.link_shared:
                self.add_path(self.cxx.compile_env, self.use_system_cxx_lib)
        additional_flags = self.get_lit_conf("test_linker_flags")
        if additional_flags:
            self.cxx.link_flags += shlex.split(additional_flags)

    def configure_link_flags_abi_library_path(self):
        # Configure ABI library paths.
        self.abi_library_root = self.get_lit_conf("abi_library_path")
        if self.abi_library_root:
            self.cxx.link_flags += ["-L" + self.abi_library_root]
            if not self.is_windows:
                if self.cxx.type == "nvcc":
                    self.cxx.link_flags += [
                        "-Xcompiler",
                        '"-Wl,-rpath,' + self.cxx_runtime_root + '"',
                    ]
                else:
                    self.cxx.link_flags += ["-Wl,-rpath," + self.abi_library_root]
            else:
                self.add_path(self.exec_env, self.abi_library_root)

    def configure_link_flags_cxx_library(self):
        libcxx_experimental = self.get_lit_bool("enable_experimental", default=False)
        if libcxx_experimental:
            self.config.available_features.add("c++experimental")
            self.cxx.link_flags += ["-lc++experimental"]
        if self.link_shared:
            self.cxx.link_flags += ["-lc++"]

    def configure_link_flags_abi_library(self):
        cxx_abi = self.get_lit_conf("cxx_abi", "libcxxabi")
        if cxx_abi == "libstdc++":
            self.cxx.link_flags += ["-lstdc++"]
        elif cxx_abi == "libsupc++":
            self.cxx.link_flags += ["-lsupc++"]
        elif cxx_abi == "libcxxabi":
            # If the C++ library requires explicitly linking to libc++abi, or
            # if we're testing libc++abi itself (the test configs are shared),
            # then link it.
            testing_libcxxabi = self.get_lit_conf("name", "") == "libc++abi"
            if self.target_info.allow_cxxabi_link() or testing_libcxxabi:
                libcxxabi_shared = self.get_lit_bool("libcxxabi_shared", default=True)
                if libcxxabi_shared:
                    self.cxx.link_flags += ["-lc++abi"]
                else:
                    cxxabi_library_root = self.get_lit_conf("abi_library_path")
                    if cxxabi_library_root:
                        libname = self.make_static_lib_name("c++abi")
                        abs_path = os.path.join(cxxabi_library_root, libname)
                        self.cxx.link_flags += [abs_path]
                    else:
                        self.cxx.link_flags += ["-lc++abi"]
        elif cxx_abi == "libcxxrt":
            self.cxx.link_flags += ["-lcxxrt"]
        elif cxx_abi == "vcruntime":
            debug_suffix = "d" if self.debug_build else ""
            self.cxx.link_flags += [
                "-l%s%s" % (lib, debug_suffix)
                for lib in ["vcruntime", "ucrt", "msvcrt"]
            ]
        elif cxx_abi == "none" or cxx_abi == "default":
            if self.is_windows:
                debug_suffix = "d" if self.debug_build else ""
                self.cxx.link_flags += ["-lmsvcrt%s" % debug_suffix]
        else:
            self.lit_config.fatal("C++ ABI setting %s unsupported for tests" % cxx_abi)

    def configure_extra_library_flags(self):
        if self.get_lit_bool("cxx_ext_threads", default=False):
            self.cxx.link_flags += ["-lc++external_threads"]
        self.target_info.add_cxx_link_flags(self.cxx.link_flags)

    def configure_color_diagnostics(self):
        use_color = self.get_lit_conf("color_diagnostics")
        if use_color is None:
            use_color = os.environ.get("LIBCXX_COLOR_DIAGNOSTICS")
        if use_color is None:
            return
        if use_color != "":
            self.lit_config.fatal(
                'Invalid value for color_diagnostics "%s".' % use_color
            )
        color_flag = "-fdiagnostics-color=always"
        # Check if the compiler supports the color diagnostics flag. Issue a
        # warning if it does not since color diagnostics have been requested.
        if not self.cxx.hasCompileFlag(color_flag):
            self.lit_config.warning(
                "color diagnostics have been requested but are not supported "
                "by the compiler"
            )
        else:
            self.cxx.flags += [color_flag]

    def configure_debug_mode(self):
        debug_level = self.get_lit_conf("debug_level", None)
        if not debug_level:
            return
        if debug_level not in ["0", "1"]:
            self.lit_config.fatal('Invalid value for debug_level "%s".' % debug_level)
        self.cxx.compile_flags += ["-D_LIBCUDACXX_DEBUG=%s" % debug_level]

    def configure_warnings(self):
        default_enable_warnings = (
            "clang" in self.config.available_features
            or "msvc" in self.config.available_features
            or "nvcc" in self.config.available_features
        )
        enable_warnings = self.get_lit_bool("enable_warnings", default_enable_warnings)
        self.cxx.useWarnings(enable_warnings)
        if "nvcc" in self.config.available_features:
            self.cxx.warning_flags += ["-Xcudafe", "--display_error_number"]
            self.cxx.warning_flags += ["-Werror=all-warnings"]
            if "msvc" in self.config.available_features:
                self.cxx.warning_flags += ["-Xcompiler", "/W4", "-Xcompiler", "/WX"]
                # warning C4100: 'quack': unreferenced formal parameter
                self.cxx.warning_flags += ["-Xcompiler", "-wd4100"]
                # warning C4127: conditional expression is constant
                self.cxx.warning_flags += ["-Xcompiler", "-wd4127"]
                # warning C4180: qualifier applied to function type has no meaning; ignored
                self.cxx.warning_flags += ["-Xcompiler", "-wd4180"]
                # warning C4309: 'moo': truncation of constant value
                self.cxx.warning_flags += ["-Xcompiler", "-wd4309"]
                # warning C4996: deprecation warnings
                self.cxx.warning_flags += ["-Xcompiler", "-wd4996"]
            else:
                # TODO: Re-enable soon.
                def addIfHostSupports(flag):
                    if hasattr(
                        self.cxx, "host_cxx"
                    ) and self.cxx.host_cxx.hasWarningFlag(flag):
                        self.cxx.warning_flags += ["-Xcompiler", flag]

                addIfHostSupports("-Wall")
                addIfHostSupports("-Wextra")
                addIfHostSupports("-Werror")
                if "gcc" in self.config.available_features:
                    addIfHostSupports(
                        "-Wno-literal-suffix"
                    )  # GCC warning about reserved UDLs
                addIfHostSupports(
                    "-Wno-user-defined-literals"
                )  # Clang warning about reserved UDLs
                addIfHostSupports("-Wno-unused-parameter")
                addIfHostSupports(
                    "-Wno-unused-local-typedefs"
                )  # GCC warning local typdefs
                addIfHostSupports("-Wno-deprecated-declarations")
                addIfHostSupports("-Wno-noexcept-type")
                addIfHostSupports("-Wno-unused-function")

                if "gcc-4.8" in self.config.available_features:
                    # GCC pre-GCC5 spuriously generates these on reasonable aggregate initialization.
                    addIfHostSupports("-Wno-missing-field-initializers")

                # TODO: port the warning disables from the non-NVCC path?

                self.cxx.warning_flags += [
                    "-D_LIBCUDACXX_DISABLE_PRAGMA_GCC_SYSTEM_HEADER"
                ]
                pass
        else:
            self.cxx.warning_flags += [
                "-D_LIBCUDACXX_DISABLE_PRAGMA_GCC_SYSTEM_HEADER",
                "-Wall",
                "-Wextra",
                "-Werror",
            ]
            if self.cxx.hasWarningFlag("-Wuser-defined-warnings"):
                self.cxx.warning_flags += ["-Wuser-defined-warnings"]
                self.config.available_features.add("diagnose-if-support")
            self.cxx.addWarningFlagIfSupported("-Wshadow")
            self.cxx.addWarningFlagIfSupported("-Wno-unused-command-line-argument")
            self.cxx.addWarningFlagIfSupported("-Wno-attributes")
            self.cxx.addWarningFlagIfSupported("-Wno-pessimizing-move")
            self.cxx.addWarningFlagIfSupported("-Wno-c++11-extensions")
            self.cxx.addWarningFlagIfSupported("-Wno-user-defined-literals")
            self.cxx.addWarningFlagIfSupported("-Wno-noexcept-type")
            self.cxx.addWarningFlagIfSupported("-Wno-aligned-allocation-unavailable")
            # These warnings should be enabled in order to support the MSVC
            # team using the test suite; They enable the warnings below and
            # expect the test suite to be clean.
            self.cxx.addWarningFlagIfSupported("-Wsign-compare")
            self.cxx.addWarningFlagIfSupported("-Wunused-variable")
            self.cxx.addWarningFlagIfSupported("-Wunused-parameter")
            self.cxx.addWarningFlagIfSupported("-Wunreachable-code")

        std = self.get_lit_conf("std", None)
        if std in ["c++98", "c++03"]:
            if "nvcc" not in self.config.available_features:
                # The '#define static_assert' provided by libc++ in C++03 mode
                # causes an unused local typedef whenever it is used.
                self.cxx.addWarningFlagIfSupported("-Wno-unused-local-typedef")

    def configure_sanitizer(self):
        san = self.get_lit_conf("use_sanitizer", "").strip()
        if san:
            self.target_info.add_sanitizer_features(san, self.config.available_features)
            # Search for llvm-symbolizer along the compiler path first
            # and then along the PATH env variable.
            symbolizer_search_paths = os.environ.get("PATH", "")
            cxx_path = libcudacxx.util.which(self.cxx.path)
            if cxx_path is not None:
                symbolizer_search_paths = (
                    os.path.dirname(cxx_path) + os.pathsep + symbolizer_search_paths
                )
            llvm_symbolizer = libcudacxx.util.which(
                "llvm-symbolizer", symbolizer_search_paths
            )

            def add_ubsan():
                self.cxx.flags += [
                    "-fsanitize=undefined",
                    "-fno-sanitize=float-divide-by-zero",
                    "-fno-sanitize-recover=all",
                ]
                self.exec_env["UBSAN_OPTIONS"] = "print_stacktrace=1"
                self.config.available_features.add("ubsan")

            # Setup the sanitizer compile flags
            self.cxx.flags += ["-g", "-fno-omit-frame-pointer"]
            if (
                san == "Address"
                or san == "Address;Undefined"
                or san == "Undefined;Address"
            ):
                self.cxx.flags += ["-fsanitize=address"]
                if llvm_symbolizer is not None:
                    self.exec_env["ASAN_SYMBOLIZER_PATH"] = llvm_symbolizer
                # FIXME: Turn ODR violation back on after PR28391 is resolved
                # https://bugs.llvm.org/show_bug.cgi?id=28391
                self.exec_env["ASAN_OPTIONS"] = "detect_odr_violation=0"
                self.config.available_features.add("asan")
                self.config.available_features.add("sanitizer-new-delete")
                self.cxx.compile_flags += ["-O1"]
                if san == "Address;Undefined" or san == "Undefined;Address":
                    add_ubsan()
            elif san == "Memory" or san == "MemoryWithOrigins":
                self.cxx.flags += ["-fsanitize=memory"]
                if san == "MemoryWithOrigins":
                    self.cxx.compile_flags += ["-fsanitize-memory-track-origins"]
                if llvm_symbolizer is not None:
                    self.exec_env["MSAN_SYMBOLIZER_PATH"] = llvm_symbolizer
                self.config.available_features.add("msan")
                self.config.available_features.add("sanitizer-new-delete")
                self.cxx.compile_flags += ["-O1"]
            elif san == "Undefined":
                add_ubsan()
                self.cxx.compile_flags += ["-O2"]
            elif san == "Thread":
                self.cxx.flags += ["-fsanitize=thread"]
                self.config.available_features.add("tsan")
                self.config.available_features.add("sanitizer-new-delete")
            else:
                self.lit_config.fatal(
                    "unsupported value for use_sanitizer: {0}".format(san)
                )
            san_lib = self.get_lit_conf("sanitizer_library")
            if san_lib:
                if self.cxx.type == "nvcc":
                    self.cxx.link_flags += [
                        "-Xcompiler",
                        '"-Wl,-rpath,' + os.path.dirname(san_lib) + '"',
                    ]
                else:
                    self.cxx.link_flags += ["-Wl,-rpath," + os.path.dirname(san_lib)]

    def configure_coverage(self):
        self.generate_coverage = self.get_lit_bool("generate_coverage", False)
        if self.generate_coverage:
            self.cxx.flags += ["-g", "--coverage"]
            self.cxx.compile_flags += ["-O0"]

    def configure_coroutines(self):
        if self.cxx.hasCompileFlag("-fcoroutines-ts"):
            macros = self._dump_macros_verbose(flags=["-fcoroutines-ts"])
            if "__cpp_coroutines" not in macros:
                self.lit_config.warning(
                    "-fcoroutines-ts is supported but __cpp_coroutines is not defined"
                )
            # Consider coroutines supported only when the feature test macro
            # reflects a recent value.
            if intMacroValue(macros["__cpp_coroutines"]) >= 201703:
                self.config.available_features.add("fcoroutines-ts")

    def configure_modules(self):
        modules_flags = ["-fmodules"]
        if platform.system() != "Darwin":
            modules_flags += ["-Xclang", "-fmodules-local-submodule-visibility"]
        supports_modules = self.cxx.hasCompileFlag(modules_flags)
        enable_modules = self.get_modules_enabled()
        if enable_modules and not supports_modules:
            self.lit_config.fatal(
                "-fmodules is enabled but not supported by the compiler"
            )
        if not supports_modules:
            return
        self.config.available_features.add("modules-support")
        module_cache = os.path.join(self.config.test_exec_root, "modules.cache")
        module_cache = os.path.realpath(module_cache)
        if os.path.isdir(module_cache):
            shutil.rmtree(module_cache)
        os.makedirs(module_cache)
        self.cxx.modules_flags += modules_flags + [
            "-fmodules-cache-path=" + module_cache
        ]
        if enable_modules:
            self.config.available_features.add("-fmodules")
            self.cxx.useModules()

    def configure_substitutions(self):
        sub = self.config.substitutions
        cxx_path = shlex.quote(self.cxx.path)
        # Configure compiler substitutions
        sub.append(("%cxx", cxx_path))
        sub.append(("%libcxx_src_root", self.libcudacxx_src_root))
        # Configure flags substitutions
        flags_str = " ".join([shlex.quote(f) for f in self.cxx.flags])
        compile_flags_str = " ".join([shlex.quote(f) for f in self.cxx.compile_flags])
        link_flags_str = " ".join([shlex.quote(f) for f in self.cxx.link_flags])
        all_flags = "%s %s %s" % (flags_str, compile_flags_str, link_flags_str)
        sub.append(("%flags", flags_str))
        sub.append(("%compile_flags", compile_flags_str))
        sub.append(("%link_flags", link_flags_str))
        sub.append(("%all_flags", all_flags))
        if self.cxx.isVerifySupported():
            verify_str = " " + " ".join(self.cxx.verify_flags) + " "
            sub.append(("%verify", verify_str))
        # Add compile and link shortcuts
        compile_str = cxx_path + " -o %t.o %s -c " + flags_str + " " + compile_flags_str
        link_str = cxx_path + " -o %t.exe %t.o " + flags_str + " " + link_flags_str
        assert type(link_str) is str
        build_str = cxx_path + " -o %t.exe %s " + all_flags
        if self.cxx.use_modules:
            sub.append(("%compile_module", compile_str))
            sub.append(("%build_module", build_str))
        elif self.cxx.modules_flags is not None:
            modules_str = " ".join(self.cxx.modules_flags) + " "
            sub.append(("%compile_module", compile_str + " " + modules_str))
            sub.append(("%build_module", build_str + " " + modules_str))
        sub.append(("%compile", compile_str))
        sub.append(("%link", link_str))
        sub.append(("%build", build_str))
        # Configure exec prefix substitutions.
        # Configure run env substitution.
        sub.append(("%run", "%t.exe"))
        # Configure not program substitutions
        not_py = os.path.join(self.libcudacxx_src_root, "test", "utils", "not.py")
        not_str = "%s %s " % (shlex.quote(sys.executable), shlex.quote(not_py))
        sub.append(("not ", not_str))
        if self.get_lit_conf("libcudacxx_gdb"):
            sub.append(("%libcxx_gdb", self.get_lit_conf("libcudacxx_gdb")))

    def can_use_deployment(self):
        # Check if the host is on an Apple platform using clang.
        if not self.target_info.platform() == "darwin":
            return False
        if not self.target_info.is_host_macosx():
            return False
        if not self.cxx.type.endswith("clang"):
            return False
        return True

    def configure_triple(self):
        # Get or infer the target triple.
        target_triple = self.get_lit_conf("target_triple")
        self.use_target = self.get_lit_bool("use_target", False)
        if self.use_target and target_triple:
            self.lit_config.warning("use_target is true but no triple is specified")

        # Use deployment if possible.
        self.use_deployment = not self.use_target and self.can_use_deployment()
        if self.use_deployment:
            return

        # Save the triple (and warn on Apple platforms).
        self.config.target_triple = target_triple
        if self.use_target and "apple" in target_triple:
            self.lit_config.warning(
                "consider using arch and platform instead"
                " of target_triple on Apple platforms"
            )

        # If no target triple was given, try to infer it from the compiler
        # under test.
        if not self.config.target_triple:
            target_triple = (
                self.cxx if self.cxx.type != "nvcc" else self.cxx.host_cxx
            ).getTriple()
            # Drop sub-major version components from the triple, because the
            # current XFAIL handling expects exact matches for feature checks.
            # Example: x86_64-apple-darwin14.0.0 -> x86_64-apple-darwin14
            # The 5th group handles triples greater than 3 parts
            # (ex x86_64-pc-linux-gnu).
            target_triple = re.sub(
                r"([^-]+)-([^-]+)-([^.]+)([^-]*)(.*)", r"\1-\2-\3\5", target_triple
            )
            # linux-gnu is needed in the triple to properly identify linuxes
            # that use GLIBC. Handle redhat and opensuse triples as special
            # cases and append the missing `-gnu` portion.
            if target_triple.endswith("redhat-linux") or target_triple.endswith(
                "suse-linux"
            ):
                target_triple += "-gnu"
            self.config.target_triple = target_triple
            self.lit_config.note(
                "inferred target_triple as: %r" % self.config.target_triple
            )

    def configure_deployment(self):
        assert self.use_deployment is not None
        assert self.use_target is not None
        if not self.use_deployment:
            # Warn about ignored parameters.
            if self.get_lit_conf("arch"):
                self.lit_config.warning("ignoring arch, using target_triple")
            if self.get_lit_conf("platform"):
                self.lit_config.warning("ignoring platform, using target_triple")
            return

        assert not self.use_target
        assert self.target_info.is_host_macosx()

        # Always specify deployment explicitly on Apple platforms, since
        # otherwise a platform is picked up from the SDK.  If the SDK version
        # doesn't match the system version, tests that use the system library
        # may fail spuriously.
        arch = self.get_lit_conf("arch")
        if not arch:
            arch = (
                (self.cxx if self.cxx.type != "nvcc" else self.cxx.host_cxx)
                .getTriple()
                .split("-", 1)[0]
            )
            self.lit_config.note("inferred arch as: %r" % arch)

        inferred_platform, name, version = self.target_info.get_platform()
        if inferred_platform:
            self.lit_config.note("inferred platform as: %r" % (name + version))
        self.config.deployment = (arch, name, version)

        # Set the target triple for use by lit.
        self.config.target_triple = arch + "-apple-" + name + version
        self.lit_config.note(
            "computed target_triple as: %r" % self.config.target_triple
        )

        # If we're testing a system libc++ as opposed to the upstream LLVM one,
        # take the version of the system libc++ into account to compute which
        # features are enabled/disabled. Otherwise, disable availability markup,
        # which is not relevant for non-shipped flavors of libc++.
        if self.use_system_cxx_lib:
            # Dylib support for shared_mutex was added in macosx10.12.
            if name == "macosx" and version in ("10.%s" % v for v in range(7, 12)):
                self.config.available_features.add("dylib-has-no-shared_mutex")
                self.lit_config.note(
                    "shared_mutex is not supported by the deployment target"
                )
            # Throwing bad_optional_access, bad_variant_access and bad_any_cast is
            # supported starting in macosx10.14.
            if name == "macosx" and version in ("10.%s" % v for v in range(7, 14)):
                self.config.available_features.add("dylib-has-no-bad_optional_access")
                self.lit_config.note(
                    "throwing bad_optional_access is not supported by the deployment target"
                )

                self.config.available_features.add("dylib-has-no-bad_variant_access")
                self.lit_config.note(
                    "throwing bad_variant_access is not supported by the deployment target"
                )

                self.config.available_features.add("dylib-has-no-bad_any_cast")
                self.lit_config.note(
                    "throwing bad_any_cast is not supported by the deployment target"
                )
            # Filesystem is support on Apple platforms starting with macosx10.15.
            if name == "macosx" and version in ("10.%s" % v for v in range(7, 15)):
                self.config.available_features.add("dylib-has-no-filesystem")
                self.lit_config.note(
                    "the deployment target does not support <filesystem>"
                )

    def configure_env(self):
        self.target_info.configure_env(self.exec_env)

    def add_path(self, dest_env, new_path):
        if "PATH" not in dest_env:
            dest_env["PATH"] = new_path
        else:
            split_char = ";" if self.is_windows else ":"
            dest_env["PATH"] = "%s%s%s" % (new_path, split_char, dest_env["PATH"])
