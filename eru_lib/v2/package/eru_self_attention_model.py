import math
import torch

from ml3 import SimpleObject

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

# ---------------------------------------------------------------------------
# multi-head self-attention

class EruSelfAttentionModel(torch.nn.Module): 
    
    # -----------------------------------------------------------------------

    def __init__(self,
        vocab_size,
        embedding_dim,
        attention_dim,
        c_heads,
        c_layers=1,
        dropout=None
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.c_heads = c_heads
        attention_v_dim = embedding_dim # attention value dimensionality

        self.layers = [
            SelfAttention(
                input_dim=embedding_dim,
                attention_dim=attention_dim,
                output_dim=attention_v_dim,
                c_heads=c_heads,
            )
        ] + [
            SelfAttention(
                input_dim=attention_v_dim,
                attention_dim=attention_dim,
                output_dim=attention_v_dim,
                c_heads=c_heads,
            )
            for _i_layer in range(c_layers - 1)
        ]

        return

    # -----------------------------------------------------------------------

    def forward(self, x):

        # batch, seq, embedding-d
        embedded = self.embedding(x)

        # batch, seq, embedding-d -> embedding-d, seq, batch ->
        #   num-heads, embedding-d, seq, batch ->
        #     batch, num-heads, seq, embedding-d
        embedded_repeated = embedded.permute(2, 1, 0).repeat(self.c_heads, 1, 1, 1).permute(3, 0, 2, 1)

        layer_output_cur = embedded_repeated
        for layer in self.layers:
            layer_output_cur = layer.forward(layer_output_cur)
            # batch, num-heads, seq, attention-v-d

        output_valued = layer_output_cur
        return output_valued
