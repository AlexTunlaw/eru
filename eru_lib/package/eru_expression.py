import numpy as np

# ---------------------------------------------------------------------------

class EruExpression:

    # -----------------------------------------------------------------------

    @classmethod
    def make_with_one_vocab_token(cls, id, vocab_distribution):
        expression = EruExpression(id=id)
        expression.add_new_vocab_token(vocab_distribution=vocab_distribution)
        return expression

    # -----------------------------------------------------------------------

    def __init__(self, id: int, depth: int=0):

        self.id = id
        self.depth = depth

        self.vocab_tokens = []
        self.vocab_tokens_distribution = []
        self.vocab_tokens_distribution_normalized = None
        self.modifier_expressions = []
        self.modifier_expressions_distribution = []
        self.modifier_expressions_distribution_normalized = None

        return

    # -----------------------------------------------------------------------

    def get_max_depth(self, visited=None):
        if not self.modifier_expressions:
            return 0
        if visited is None:
            visited = set()
        visited.add(self)
        max_depth = 1 + max([0] + [
            e.get_max_depth(visited=visited)
            for e in self.modifier_expressions
            if visited is None or e not in visited
        ])
        visited.remove(self)
        return max_depth

    # -----------------------------------------------------------------------

    def get_all_modifier_expressions(self, visited=None):
        if visited is None:
            visited = set()
        visited.add(self)
        result = self.modifier_expressions.copy()
        for e in self.modifier_expressions:
            if visited is None or e not in visited:
                result.extend(e.get_all_modifier_expressions(visited=visited))
        visited.remove(self)
        return result

    # -----------------------------------------------------------------------

    def add_new_vocab_token(self, vocab_distribution):

        vocab_token, vocab_token_p = vocab_distribution.make_next()
        self.vocab_tokens.append(vocab_token)
        self.vocab_tokens_distribution.append(vocab_token_p)
        self.vocab_tokens_distribution_normalized = None
        return vocab_token

    # -----------------------------------------------------------------------

    def add_new_modifier_expression(self, expression_distribution, new_modifier_expression=None):

        if new_modifier_expression is None:
            modifier_expression, modifier_expression_p = expression_distribution.make_next()
        else:
            modifier_expression = new_modifier_expression
            modifier_expression_p = expression_distribution.get_item_p(new_modifier_expression)

        self.modifier_expressions.append(modifier_expression)
        self.modifier_expressions_distribution.append(modifier_expression_p)
        self.modifier_expressions_distribution_normalized = None
        return modifier_expression

    # -----------------------------------------------------------------------

    def ensure_vocab_tokens_distribution_normalized(self):
        if self.vocab_tokens_distribution_normalized is None:
            s = sum(self.vocab_tokens_distribution)
            self.vocab_tokens_distribution_normalized = [p/s for p in self.vocab_tokens_distribution]
        return

    # -----------------------------------------------------------------------

    def ensure_modifier_expressions_distribution_normalized(self):
        if self.modifier_expressions_distribution_normalized is None:
            s = sum(self.modifier_expressions_distribution)
            self.modifier_expressions_distribution_normalized = [p/s for p in self.modifier_expressions_distribution]
        return

    # -----------------------------------------------------------------------

    def make_utterance(self):

        self.ensure_vocab_tokens_distribution_normalized()
        self.ensure_modifier_expressions_distribution_normalized()

        utterance = [np.random.choice(self.vocab_tokens, p=self.vocab_tokens_distribution_normalized)]
        signature = [self.id]
        for modifier_expression, modifier_expression_p in \
            zip(self.modifier_expressions, self.modifier_expressions_distribution_normalized):
            if np.random.rand() < modifier_expression_p:
                modifier_utterance, modifier_signature = modifier_expression.make_utterance()
                utterance.extend(modifier_utterance)
                signature.extend(modifier_signature)

        np.random.shuffle(utterance)
        return utterance, signature