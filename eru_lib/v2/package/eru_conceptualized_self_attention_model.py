import torch

from .eru_self_attention import SelfAttention
from .eru_conceptualized_self_attention import ConceptualizedSelfAttention

# ---------------------------------------------------------------------------
# multi-head self-attention

class EruConceptualizedSelfAttentionModel(torch.nn.Module): 
    
    # -----------------------------------------------------------------------

    def __init__(self,
        vocab_size,
        embedding_dim,
        c_conceptualizations,
        c_heads=1,
        c_layers=1,
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.c_heads = c_heads

        attention_v_dim = embedding_dim # attention value dimensionality

        self.layers = torch.nn.ModuleList([
            ConceptualizedSelfAttention(
                input_dim=embedding_dim,
                c_conceptualizations=c_conceptualizations,
                output_dim=attention_v_dim,
                c_heads=self.c_heads,
                desc=str(0),
            )
        ] + [
            ConceptualizedSelfAttention(
                input_dim=attention_v_dim,
                c_conceptualizations=c_conceptualizations,
                output_dim=attention_v_dim,
                c_heads=self.c_heads,
                desc=str(i_layer),
            )
            for i_layer in range(1, c_layers)
        ])

        return

    # -----------------------------------------------------------------------

    def forward(self, x, ctx=None, observe_fn=None):

        # batch, seq, embedding-d
        embedded = self.embedding(x)

        # batch, seq, embedding-d -> embedding-d, seq, batch ->
        #   num-heads, embedding-d, seq, batch ->
        #     batch, num-heads, seq, embedding-d
        embedded_repeated = embedded.permute(2, 1, 0).repeat(self.c_heads, 1, 1, 1).permute(3, 0, 2, 1)

        layer_output_cur = embedded_repeated
        for layer in self.layers:
            # batch, num-heads, seq, attention-v-d
            layer_output_cur = layer.forward(layer_output_cur, ctx, observe_fn)

        output_valued = layer_output_cur
        return output_valued
