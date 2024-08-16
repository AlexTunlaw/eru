import numpy as np

from ml3 import SimpleObject

from .eru_expression_node import EruExpressionNode

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
        modifier_trees = []
        for modifier_expression, modifier_expression_p in \
            zip(self.modifier_expressions, self.modifier_expressions_distribution_normalized):
            if np.random.rand() < modifier_expression_p:
                modifier_utterance, modifier_signature, modifier_tree = modifier_expression.make_utterance()
                utterance.extend(modifier_utterance)
                signature.extend(modifier_signature)
                modifier_trees.append(modifier_tree)

        np.random.shuffle(utterance)
        tree = EruExpressionNode(expression=self, signature=signature, children=modifier_trees)
        return utterance, signature, tree

    # -----------------------------------------------------------------------

    def make_similar_utterance(self, tree_1, requested_similarity, expression_vocab_loss, modifiers_weight):

        assert tree_1.expression == self

        self_expression_token = tree_1.signature[0]
        self_expression_similarity = 1.0
        if len(self.vocab_tokens) > 1 and requested_similarity < np.random.rand():
            self_expression_similarity -= expression_vocab_loss
            while True:
                new_self_expression_token = [np.random.choice(self.vocab_tokens, p=self.vocab_tokens_distribution_normalized)]
                if self_expression_token != new_self_expression_token:
                    break
                continue

        utterance = [self_expression_token]
        signature = [self.id]
        modifier_trees = []

        modifiers_similarity_sum = 0.0

        for modifier_expression in self.modifier_expressions:
            matching_child = None
            for child in tree_1.children:
                if modifier_expression == child.expression:
                    matching_child = child
                    break
            if matching_child:
                if np.random.rand() < requested_similarity:
                    modifier_utterance, modifier_signature, modifier_tree, modifier_actual_similarity = \
                        matching_child.expression.make_similar_utterance(
                            tree_1=matching_child,
                            requested_similarity=requested_similarity,
                            expression_vocab_loss=expression_vocab_loss,
                            modifiers_weight=modifiers_weight
                        )
                    utterance.extend(modifier_utterance)
                    signature.extend(modifier_signature)
                    modifier_trees.append(modifier_tree)
                    modifiers_similarity_sum += modifier_actual_similarity
            else:
                if requested_similarity < np.random.rand():
                    modifier_utterance, modifier_signature, modifier_tree = \
                        modifier_expression.make_utterance()
                    utterance.extend(modifier_utterance)
                    signature.extend(modifier_signature)
                    modifier_trees.append(modifier_tree)
                    modifiers_similarity_sum += 0.0 # this modifier is completely new, so zero similarity

        np.random.shuffle(utterance)
        tree = EruExpressionNode(expression=self, signature=signature, children=modifier_trees)
        actual_similarity = (
            (1.0 - modifiers_weight) * self_expression_similarity +
            (
                modifiers_weight * modifiers_similarity_sum / len(self.modifier_expressions)
                if modifier_trees
                else 0.0
            )
        )
        assert 0.0 <= actual_similarity <= 1.0
        return utterance, signature, tree, actual_similarity
