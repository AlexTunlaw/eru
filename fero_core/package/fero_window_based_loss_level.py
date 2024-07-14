# ---------------------------------------------------------------------------

class FeroWindowBasedLossLevel:

    # -----------------------------------------------------------------------

    def __init__(self, window_size=5):

        self.window_size = window_size
        self.losses_window = []

    # -----------------------------------------------------------------------

    def reset(self):
        self.losses_window = []

    # -----------------------------------------------------------------------

    def observe(self, train_loss: float):
        self.losses_window.append(train_loss)
        if len(self.losses_window) > self.window_size:
            self.losses_window.pop(0)

    # -----------------------------------------------------------------------

    @property
    def current_level(self):
        return sum(self.losses_window) / len(self.losses_window)
