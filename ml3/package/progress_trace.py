import time

# ---------------------------------------------------------------------------
# A basic tracer of progress in scripts that take long time. Consider using
# richer tqdm for production-grade tracing.
#
# Changes to interface or behavior of this class is discouraged since there
# are many non-production scripts that use this.
#

class ProgressTrace:

    # -----------------------------------------------------------------------

    def __init__(
        self,
        trace_every_seconds=2.0,
        check_every_ticks=100,
        intermediate_trace_template="  ..%s",
        final_trace_template="  ..%s - DONE",
    ):

        self.trace_threshold_ms = trace_every_seconds * 1000
        self.check_every_ticks = check_every_ticks
        self.intermediate_trace_template = intermediate_trace_template
        self.final_trace_template = final_trace_template

        self.i_tick = 0
        self.time_0 = time.time()

        self.trace_tick()

    # -----------------------------------------------------------------------

    def trace_tick(self, additional_info=None):
        s = self.intermediate_trace_template % str(self.i_tick)
        if additional_info:
            s += " " + additional_info
        print(s)

    # -----------------------------------------------------------------------

    def trace_final_tick(self, additional_info=None):
        s = self.final_trace_template % self.format_n(self.i_tick)
        if additional_info:
            s += " " + additional_info
        print(s)

    # -----------------------------------------------------------------------

    def format_n(self, n):
        K = 1000
        M = K*K
        if n < K:
            return str(n)
        elif n < 10 * K:
            return f"{n / K:.1f}K"
        elif n < M:
            return f"{int(n / K)}K"
        elif n < 10 * M:
            return f"{n / M:.1f}M"

        return f"{int(n / M)}M"

    # -----------------------------------------------------------------------

    def tick(self, trace_fn=None):

        self.i_tick += 1

        if (self.i_tick % self.check_every_ticks) == 0:
            time_1 = time.time()
            time_delta_ms = (time_1 - self.time_0) * 1000
            if time_delta_ms >= self.trace_threshold_ms:
                self.trace_tick(additional_info=trace_fn() if trace_fn else None)
                self.time_0 = time_1

    # -----------------------------------------------------------------------

    def done(self, trace_fn=None):
        self.trace_final_tick(additional_info=trace_fn() if trace_fn else None)
