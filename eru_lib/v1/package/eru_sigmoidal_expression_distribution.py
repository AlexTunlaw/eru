import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))

# ---------------------------------------------------------------------------
# In this distribution first expressions have equal distribution, and towards
# the half the distribution drops begins drop to zero, sigmoidally. We also
# quadratically stretch the range so that we have short head and long tail.

class EruSigmoidalExpressionDistribution:

    # -----------------------------------------------------------------------

    def __init__(self, c_expressions, make_item_fn=None):

        base_range = 7
        probabilities_raw = [
            sigmoid(2 * base_range * pow(c_expressions - k, 2) / pow(c_expressions, 2) - base_range)
            for k in range(c_expressions)
        ]
        self.probabilities = (probabilities_raw / sum(probabilities_raw)).tolist()
        self.items = [None] * len(self.probabilities)

        self.make_item_fn = make_item_fn if make_item_fn is not None else lambda i: i
        return

    # -----------------------------------------------------------------------

    @property
    def size(self): return len(self.probabilities)

    # -----------------------------------------------------------------------

    @property
    def item_count(self): return sum(1 for item in self.items if item is not None)

    # -----------------------------------------------------------------------

    def make_next(self):
        k = k_initial = np.random.randint(0, len(self.items))
        while self.items[k] is not None:
            k = (k + 1) % len(self.items)
            assert k != k_initial, "the distribution is exhausted"
        self.items[k] = self.make_item_fn(k)
        return self.items[k], self.probabilities[k]

    # -----------------------------------------------------------------------

    def get_made_items(self):
        return [
            (item, self.probabilities[k])
            for k, item in enumerate(self.items)
            if item is not None
        ]

    # -----------------------------------------------------------------------

    def get_item_p(self, item):
        for item_cur, p in zip(self.items, self.probabilities):
            if item_cur == item:
                return p
        assert f"item not found: {item.id}"

    # -----------------------------------------------------------------------

    def get_sample(self):

        return np.random.choice(
            len(self.probabilities),
            p=self.probabilities
        )

    # -----------------------------------------------------------------------

    def get_samples(self, k, max_attempts_per_expression=100):
        
        expressions = set()
        for _k in range(k):

            for _i_attempt in range(max_attempts_per_expression):
                expression = self.get_sample()
                if expression not in expressions:
                    expressions.add(expression)
                    break

        return list(expressions)
