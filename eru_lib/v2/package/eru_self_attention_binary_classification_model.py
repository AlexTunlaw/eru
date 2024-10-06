import math
import torch

from ml3 import SimpleObject

from .eru_self_attention_model import EruSelfAttentionModel

# ---------------------------------------------------------------------------
# multi-head self-attention

class EruSelfAttentionBinaryClassificationModel(EruSelfAttentionModel): 
    
    # -----------------------------------------------------------------------

    def __init__(self,
        vocab_size,
        embedding_dim,
        attention_dim,
        c_heads,
        c_layers=1,
        dropout=None
    ):

        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            c_heads=c_heads,
            c_layers=c_layers,
            dropout=dropout
        )

        output_dim = 1
        self.fc = torch.nn.Linear(c_heads * embedding_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        return

    # -----------------------------------------------------------------------

    def forward(self, x):

        batch_size, seq_len = x.shape

        # -> batch_size, c_heads, seq_len, embedding_dim
        r_all_heads = super().forward(x=x)

        # We use EOU (last token) for seq meaning
        # -> batch_size, c_heads, embedding_dim
        r_all_heads_eou = r_all_heads[:, :, -1, :]

        # -> batch_size
        output = self.sigmoid(
            self.fc(r_all_heads_eou.reshape(batch_size, -1)
            ).reshape(batch_size)
        )
        assert output.shape == (batch_size, )
        return output
