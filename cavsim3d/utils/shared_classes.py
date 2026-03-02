import os
import sys


class suppress_c_stdout_stderr:
    def __enter__(self):
        self.stdout_fd = sys.__stdout__.fileno()
        self.stderr_fd = sys.__stderr__.fileno()

        # Save original fds
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)

        # Redirect stdout/stderr to /dev/null (or nul on Windows)
        devnull = os.devnull
        self.null_fd = os.open(devnull, os.O_RDWR)
        os.dup2(self.null_fd, self.stdout_fd)
        os.dup2(self.null_fd, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        os.close(self.null_fd)
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)