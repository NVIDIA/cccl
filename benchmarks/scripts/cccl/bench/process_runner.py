import os
import signal
import subprocess


class ProcessRunner:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(ProcessRunner, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.process = None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def new_process(self, cmd):
        self.process = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self.process

    def signal_handler(self, signum, frame):
        self.kill_process()
        raise SystemExit("search was interrupted")

    def kill_process(self):
        if self.process is not None and self.process.poll is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
