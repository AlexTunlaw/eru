import math
import torch

from ml3 import SimpleObject

from .eru_self_attention_model import EruSelfAttentionModel

# ---------------------------------------------------------------------------
# multi-head self-attention

class EruSelfAttentionBinaryClassificationModel(EruSelfAttentionModel): 
    
    # -----------------------------------------------------------------------
    # based on:
    # https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
    # https://github.com/sooftware/attentions/blob/master/attentions.py
    #
    # novel: attention_weight_mode for faster convergence
    #

    def __init__(self,
        vocab_size,
        embedding_dim,
        attention_dim,
        c_heads,
        attention_weights_mode: str="overtaking-sigmoid",
        dropout=None
    ):

        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            c_heads=c_heads,
            attention_weights_mode=attention_weights_mode,
            dropout=dropout
        )

        output_dim = 1
        self.fc = torch.nn.Linear(c_heads * embedding_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        return

    # -----------------------------------------------------------------------

    def forward(self, x, observe_fn=None, i_batch=None):

        batch_size, seq_len = x.shape

        # -> batch_size, c_heads, seq_len, embedding_dim
        r_all_heads = super().forward(
            x=x,
            observe_fn=observe_fn,
            i_batch=i_batch
        )

        # We use BOS (token 0) for seq meaning
        # -> batch_size, c_heads, embedding_dim
        r_all_heads_bos = r_all_heads[:, :, 0, :]

        # -> batch_size
        output = self.sigmoid(
            self.fc(r_all_heads_bos.reshape(batch_size, -1)
            ).reshape(batch_size)
        )
        assert output.shape == (batch_size, )
        return output
