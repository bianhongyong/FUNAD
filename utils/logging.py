"""Logging utilities: stdout/stderr mirroring and log file setup."""

import os
import sys

_LOG_STREAM_HOLDER = []


class TeeStream:
    """Mirror writes to terminal and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def enable_print_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    _LOG_STREAM_HOLDER.append(log_f)
    sys.stdout = TeeStream(sys.__stdout__, log_f)
    sys.stderr = TeeStream(sys.__stderr__, log_f)
    print(f"[log] mirrored stdout/stderr to: {log_path}")
