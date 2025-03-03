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
        dropout=None,
        attention_v_dim=None,
        sharpening_mode="softmax",
        alignment_mode="dot-product",
    ):

        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            c_heads=c_heads,
            c_layers=c_layers,
            dropout=dropout,
            sharpening_mode=sharpening_mode,
            alignment_mode=alignment_mode,
        )

        if attention_v_dim is None:
            attention_v_dim = embedding_dim

        output_dim = 1
        self.fc = torch.nn.Linear(c_heads * attention_v_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        return

    # -----------------------------------------------------------------------

    def forward(self, x, ctx=None, observe_fn=None):

        batch_size, seq_len = x.shape

        # -> batch_size, c_heads, seq_len, embedding_dim
        r_all_heads = super().forward(x=x, ctx=ctx, observe_fn=observe_fn)

        # We use EOU (last token) for seq meaning
        # -> batch_size, c_heads, embedding_dim
        r_all_heads_eou = r_all_heads[:, :, -1, :]

        # -> batch_size
        logits = self.fc(
            r_all_heads_eou.reshape(batch_size, -1)
        ).reshape(batch_size)

        # -> batch_size
        output = self.sigmoid(logits)
        assert output.shape == (batch_size, )

        if observe_fn is not None:
            observe_fn(ctx={
                "kind": "classification-head",
                "logits": logits,
                "output": output,
            })

        return output
