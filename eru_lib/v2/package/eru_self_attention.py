import math

import torch

# ---------------------------------------------------------------------------
# based on:
# https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
# https://github.com/sooftware/attentions/blob/master/attentions.py
#

# ---------------------------------------------------------------------------

class SelfAttention:

    # -----------------------------------------------------------------------

    def __init__(self, input_dim, attention_dim, output_dim, c_heads):
        self.layer_norm = torch.nn.LayerNorm((input_dim,))
        self.W_key = torch.nn.Parameter(torch.rand(c_heads, attention_dim, input_dim))
        self.W_query = torch.nn.Parameter(torch.rand(c_heads, attention_dim, input_dim))
        self.W_value = torch.nn.Parameter(torch.rand(c_heads, output_dim, input_dim))
        self.scale = math.sqrt(input_dim)
        return

    # -----------------------------------------------------------------------

    def forward(self, input):

        # batch, num-heads, seq, input-d
        input_normalized = self.layer_norm(input)

        # W: num-heads, attention-d, input-d
        # query, keys, values:
        # batch, num-heads, seq, attention-d
        queries = torch.matmul(self.W_query, input_normalized.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        keys = torch.matmul(self.W_key, input_normalized.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        values = torch.matmul(self.W_value, input_normalized.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        # batch, num-heads, seq, seq
        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / self.scale

        # batch, num-heads, seq, seq
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # batch, num-heads, seq, output-d
        output_valued = torch.matmul(attention_weights, values)

        return output_valued
