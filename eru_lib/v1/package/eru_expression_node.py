# ---------------------------------------------------------------------------

class EruExpressionNode:

    # -----------------------------------------------------------------------

    def __init__(self, expression, signature, children):
        self.expression = expression
        self.signature = signature
        self.children = children
        return