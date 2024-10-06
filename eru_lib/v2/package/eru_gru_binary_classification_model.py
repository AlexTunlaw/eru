import torch

# ---------------------------------------------------------------------------

class EruGruBinaryClassificationModel(torch.nn.Module):
    
    # -----------------------------------------------------------------------

    def __init__(self, vocab_size, embedding_dim, gru_hidden_dim, dropout=None):

        super().__init__()
        bidirectional = False
        output_dim = 1

        self.embedding = torch.nn.Embedding(
            vocab_size,
            embedding_dim
        )
        self.gru = torch.nn.GRU(
            embedding_dim,
            gru_hidden_dim,
            num_layers=1,
            bidirectional=bidirectional,
            dropout=float(dropout) if dropout is not None else 0.0,
            batch_first=True
        )
        self.fc = torch.nn.Linear(
            gru_hidden_dim * 2 if bidirectional else gru_hidden_dim,
            output_dim
        )
        self.sigmoid = torch.nn.Sigmoid()
        return

    # -----------------------------------------------------------------------

    def forward(self, x):

        embedded = self.embedding(x)
        
        gru_out, _ = self.gru(embedded)
        if self.gru.bidirectional:
            hidden = torch.cat(
                (
                    gru_out[:, -1, :self.gru.hidden_size],
                    gru_out[:, 0, self.gru.hidden_size:]
                ),
                dim=1
            )
        else:
            hidden = gru_out[:, -1, :]
        
        dense_outputs = self.fc(hidden)
        
        outputs = self.sigmoid(dense_outputs)
        return outputs.squeeze(1)