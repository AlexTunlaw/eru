import torch

from .eru_self_attention_model import EruSelfAttentionModel

# ---------------------------------------------------------------------------
# a siamese model

class EruSelfAttentionSimilarityModel(EruSelfAttentionModel):

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

        # we let this layer pick (parts of) representations from all the heads
        self.fc = torch.nn.Linear(c_heads * embedding_dim, embedding_dim)

        self.cosine_similarity_fn = torch.nn.CosineSimilarity(dim=1)

        return

    # -----------------------------------------------------------------------

    def forward(self, x_1, x_2, observe_fn=None, i_batch=None):

        batch_size, seq_len = x_1.shape
        batch_size_2, seq_len_2 = x_2.shape
        assert batch_size == batch_size_2 and seq_len == seq_len_2

        # -> batch_size, c_heads, seq_len, embedding_dim
        r_1_all_heads = super().forward(x_1) # siamese architecture
        r_2_all_heads = super().forward(x_2)

        embedding_dim = r_1_all_heads.shape[-1]

        # -> batch_size, c_heads, embedding_dim
        r_1_all_heads_bos = r_1_all_heads[:, :, 0, :]
        r_2_all_heads_bos = r_2_all_heads[:, :, 0, :]

        # -> batch_size, embedding_dim
        r_1 = self.fc(
            r_1_all_heads_bos.reshape(batch_size, -1)
        ).reshape(batch_size, embedding_dim)
        r_2 = self.fc(
            r_2_all_heads_bos.reshape(batch_size, -1)
        ).reshape(batch_size, embedding_dim)

        # -> batch_size
        sim = self.cosine_similarity_fn(r_1, r_2)
        return sim
    