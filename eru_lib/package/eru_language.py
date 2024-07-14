import collections

import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))

from .eru_sigmoidal_expression_distribution import EruSigmoidalExpressionDistribution
from .eru_expression import EruExpression

# ---------------------------------------------------------------------------

class EruLanguage:

    # -----------------------------------------------------------------------

    def __init__ (self, expressions, probabilities, vocab_distribution):
        
        self.expressions = expressions
        self.probabilities = probabilities
        self.vocab_distribution = vocab_distribution
        return

    # -----------------------------------------------------------------------

    @classmethod
    def make_from_config(cls, config, print_stats=True):

        return cls.make(
            vocab_distribution=EruSigmoidalExpressionDistribution(c_expressions=config["vocab-size"]),
            p_vocab_expansion=config["p-vocab-expansion"],
            p_modifier_expansion=config["p-modifier-expansion"],
            p_modifier_reuse=config["p-modifier-reuse"],
            c_build_steps=config["build-steps"],
            max_expression_depth=config["max-expression-depth"],
            print_stats=print_stats
        )

    # -----------------------------------------------------------------------

    @classmethod
    def make(cls,
        vocab_distribution,
        p_vocab_expansion: float,
        p_modifier_expansion: float,
        p_modifier_reuse: float,
        c_build_steps: int,
        max_expression_depth: int,
        print_stats=False
    ):

        expression_distribution = EruSigmoidalExpressionDistribution(
            c_expressions=vocab_distribution.size,
            make_item_fn=lambda i: EruExpression.make_with_one_vocab_token(
                id=i,
                vocab_distribution=vocab_distribution
            )
        )

        expressions = []
        probabilities = []

        for i_build_step in range(c_build_steps):

            if (i_build_step == 0 or
                np.random.rand() < 1.0 - sigmoid(5.0 * i_build_step / c_build_steps) # p~=0.5 in the beginning, p~=1.0 in the end, sigmoidally
            ):
                new_expression, new_expression_p = expression_distribution.make_next()
                new_expression.depth = 0
                expressions.append(new_expression)
                probabilities.append(new_expression_p)

            else:
                expression_to_extend = np.random.choice(expressions) # uniform distribution; we don't necessarily want frequent expressions to be extended more than less frequent
                cls.expand_existing_expression(
                    expression=expression_to_extend,
                    vocab_distribution=vocab_distribution,
                    p_vocab_expansion=p_vocab_expansion,
                    expression_distribution=expression_distribution,
                    p_modifier_expansion=p_modifier_expansion,
                    p_modifier_reuse=p_modifier_reuse,
                    max_expression_depth=max_expression_depth
                )

        def histo_display(histo):
            for key in range(max(histo.keys()) + 1):
                print(f"  {key}: {histo.get(key, 0)}")

        # stats
        if print_stats:

            print(
                f"build steps:          {c_build_steps}\n" +
                f"vocab:                {vocab_distribution.size} (used: {vocab_distribution.item_count})\n" +
                f"expressions:          {expression_distribution.item_count}\n" +
                f"p vocab expansion:    {p_vocab_expansion:.2f}\n" +
                f"p modifier expansion: {p_modifier_expansion:.2f}\n" +
                f"p modifier reuse:     {p_modifier_reuse:.2f}\n" +
                f"max expression depth: {max_expression_depth}\n"
            )

            print("vocab widths; width: count")
            histo_display(collections.Counter(len(e.vocab_tokens) for e in expressions))

            print("modifier widths; width: count")
            histo_display(collections.Counter(len(e.modifier_expressions) for e in expressions))

            # stats - depth
            depths_histo = collections.Counter(e.get_max_depth() for e in expressions)
            print("depths; depth: count")
            histo_display(collections.Counter(e.get_max_depth() for e in expressions))

            # stats - reuse
            reuse_histo = collections.Counter()
            for e in expressions:
                reuse_histo.update(ee.id for ee in e.get_all_modifier_expressions())
            reuse_histo_value_histo = collections.Counter(reuse_histo.values())
            print("reuse; reuse count: count of reuse count")
            histo_display(reuse_histo_value_histo)

        return EruLanguage(
            expressions=expressions,
            probabilities=probabilities,
            vocab_distribution=vocab_distribution
        )

    # -----------------------------------------------------------------------

    @classmethod
    def expand_existing_expression(cls,
        expression,
        vocab_distribution,
        p_vocab_expansion,
        expression_distribution,
        p_modifier_expansion,
        p_modifier_reuse,
        max_expression_depth
    ):
        
        if (
            expression.depth == max_expression_depth or
            np.random.rand() < p_vocab_expansion
        ):
            # expand core vocab
            expression.add_new_vocab_token(vocab_distribution=vocab_distribution)
            
        elif len(expression.modifier_expressions) == 0 or np.random.rand() < p_modifier_expansion:
            # expand modifier expressions, by adding a new one, or reusing one

            if np.random.rand() < p_modifier_reuse:
                # reuse
                items_and_probabilities = expression_distribution.get_made_items()
                p_sum = sum(p for _item, p in items_and_probabilities) 
                new_modifier_expression = np.random.choice(
                    [item for item, _p in items_and_probabilities],
                    p=[p / p_sum for _item, p in items_and_probabilities]
                )
                expression.add_new_modifier_expression(
                    expression_distribution=expression_distribution,
                    new_modifier_expression=new_modifier_expression
                )
            else:
                # new
                new_modifier_expression = expression.add_new_modifier_expression(
                    expression_distribution=expression_distribution
                )

            new_modifier_expression.depth = max(expression.depth + 1, new_modifier_expression.depth)

        else: 
            # expand modifier expressions, recursively
            assert len(expression.modifier_expressions) >= 1
            modifier_expresson_to_expand = np.random.choice(expression.modifier_expressions)

            cls.expand_existing_expression(
                expression=modifier_expresson_to_expand,
                vocab_distribution=vocab_distribution,
                p_vocab_expansion=p_vocab_expansion,
                expression_distribution=expression_distribution,
                p_modifier_expansion=p_modifier_expansion,
                p_modifier_reuse=p_modifier_reuse,
                max_expression_depth=max_expression_depth
            )

        return

    # -----------------------------------------------------------------------

    def make_utterance(self, utterance_min_len, utterance_max_len):
        max_iterations = 1000
        c_iterations = 0
        while True:
            expression = np.random.choice(self.expressions)
            utterance, signature = expression.make_utterance()
            if utterance_min_len <= len(utterance) <= utterance_max_len:
                break
            c_iterations += 1
            assert c_iterations < max_iterations, f"Failed to make utterance in {max_iterations} iterations"
        return utterance, signature
