import os
import time
import signal
import subprocess

from .build import Build
from .config import Config
from .storage import Storage
from .logger import *


def create_builds_table(conn):
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS builds (
            ctk TEXT NOT NULL,
            cccl TEXT NOT NULL,
            bench TEXT NOT NULL,
            code TEXT NOT NULL,
            elapsed REAL
        );
        """)


class CMakeCache:
  _instance = None

  def __new__(cls, *args, **kwargs):
      if cls._instance is None:
          cls._instance = super().__new__(cls, *args, **kwargs)
          create_builds_table(Storage().connection())
      return cls._instance

  def pull_build(self, bench):
      config = Config()
      ctk = config.ctk
      cccl = config.cccl
      conn = Storage().connection()

      with conn:
          query = "SELECT code, elapsed FROM builds WHERE ctk = ? AND cccl = ? AND bench = ?;"
          result = conn.execute(query, (ctk, cccl, bench.label())).fetchone()

          if result:
              code, elapsed = result
              return Build(int(code), float(elapsed))

          return result

  def push_build(self, bench, build):
      config = Config()
      ctk = config.ctk
      cccl = config.cccl
      conn = Storage().connection()

      with conn:
          conn.execute("INSERT INTO builds (ctk, cccl, bench, code, elapsed) VALUES (?, ?, ?, ?, ?);",
                       (ctk, cccl, bench.label(), build.code, build.elapsed))


class CMake:
  def __init__(self):
    pass

  def do_build(self, bench, timeout):
      logger = Logger()

      try:
          if not bench.is_base():
              with open(bench.exe_name() + ".h", "w") as f:
                  f.writelines(bench.definitions())

          cmd = ["cmake", "--build", ".", "--target", bench.exe_name()]
          logger.info("starting build for {}: {}".format(bench.label(), " ".join(cmd)))

          begin = time.time()
          p = subprocess.Popen(cmd,
                              start_new_session=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
          p.wait(timeout=timeout)
          elapsed = time.time() - begin
          logger.info("finished build for {} (exit code: {}) in {:.3f}s".format(bench.label(), p.returncode, elapsed))

          return Build(p.returncode, elapsed)
      except subprocess.TimeoutExpired:
          logger.info("build for {} reached timeout of {}s".format(bench.label(), timeout))
          os.killpg(os.getpgid(p.pid), signal.SIGTERM)
          return Build(424242, float('inf'))

  def build(self, bench):
      logger = Logger()
      timeout = None

      cache = CMakeCache()

      if bench.is_base():
          # Only base build can be pulled from cache
          build = cache.pull_build(bench)

          if build:
              logger.info("found cached base build for {}".format(bench.label()))
              if bench.is_base():
                  if not os.path.exists("bin/{}".format(bench.exe_name())):
                      self.do_build(bench, None)

              return build
      else:
          base_build = self.build(bench.get_base())

          if base_build.code != 0:
              raise Exception("Base build failed")

          timeout = base_build.elapsed * 10

      build = self.do_build(bench, timeout)
      cache.push_build(bench, build)
      return build

  def clean():
      cmd = ["cmake", "--build", ".", "--target", "clean"]
      p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
      p.wait()

      if p.returncode != 0:
          raise Exception("Unable to clean build directory")
