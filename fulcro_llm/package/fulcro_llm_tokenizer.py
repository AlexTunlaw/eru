import tiktoken

import tiktoken

# ---------------------------------------------------------------------------

class FeroLlmTokenizer:

    # -----------------------------------------------------------------------

    def __init__(self):

        self.tiktoken_encoding = tiktoken.get_encoding("cl100k_base")

    # -----------------------------------------------------------------------

    def get_tokens(self, text):

        return self.tiktoken_encoding.encode(text)

    # -----------------------------------------------------------------------

    def get_text_roughly_up_to_max_tokens(self, text, max_tokens):
        tokens = self.tiktoken_encoding.encode(text)
        n_tokens = len(tokens)
        fraction_to_take = min(max_tokens, n_tokens) / n_tokens
        lim = int(len(text) * fraction_to_take)
        return text[:lim]

