import time
import logging

# ---------------------------------------------------------------------------

class FulcroFunctionLogger:

    # ---------------------------------------------------------------------------

    def __init__(self):

        self.start_time = time.time()

    # ---------------------------------------------------------------------------

    def info(self, message): self.log_core(message, logging.info)
    def warning(self, message): self.log_core(message, logging.warning)
    def error(self, message): self.log_core(message, logging.error)

    # ---------------------------------------------------------------------------

    def log_core(self, message, log_fn):
        elapsed_seconds = time.time() - self.start_time
        elapsed_seconds_formatted = (
            f"{time.time() - self.start_time:.1f}"
            if elapsed_seconds < 10
            else f"{round(time.time() - self.start_time)}"
        )
        log_fn(f"[fulcro, {elapsed_seconds_formatted} sec] {message}")
        return