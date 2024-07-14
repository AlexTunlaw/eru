import math
import torch

from ml3 import SimpleObject

# ---------------------------------------------------------------------------

class EruSelfAttentionModel(torch.nn.Module):
    
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

        super().__init__()

        assert attention_weights_mode in ["softmax", "overtaking-sigmoid"]
        self.attention_weights_mode = attention_weights_mode

        self.embedding = torch.nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.layer_norm = torch.nn.LayerNorm((embedding_dim,))

        attention_v_dim = embedding_dim
        self.W_key = torch.nn.Parameter(torch.rand(c_heads, attention_dim, embedding_dim))
        self.W_query = torch.nn.Parameter(torch.rand(c_heads, attention_dim, embedding_dim))
        self.W_value = torch.nn.Parameter(torch.rand(c_heads, attention_v_dim, embedding_dim))

        self.c_heads = c_heads
        self.scale = math.sqrt(embedding_dim)

        output_dim = 1
        self.fc = torch.nn.Linear(c_heads * attention_v_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        return

    # -----------------------------------------------------------------------

    def forward(self, x, observe_fn=None, i_batch=None, normalize_attention_weights=False):

        batch_size, seq_len = x.shape

        ctx = SimpleObject(
            i_batch=i_batch,
            x=x,
            batch_size=batch_size, seq_len=seq_len, c_heads=self.c_heads,
            embedded=None, embedded_repeated=None,
            queries=None, keys=None, values=None,
            attention_scores=None, attention_weights=None,
            output_valued=None, output_final=None
        )

        # batch, seq, embedding-d
        ctx.embedded = self.layer_norm(self.embedding(x))
        # batch, seq, embedding-d -> embedding-d, seq, batch ->
        #   num-heads, embedding-d, seq, batch ->
        #     batch, num-heads, embedding-d, seq 
        ctx.embedded_repeated = ctx.embedded.permute(2, 1, 0).repeat(self.c_heads, 1, 1, 1).permute(3, 0, 1, 2)

        # W: num-heads, attention-d, embedding-d
        # query, keys, values:
        # batch, num-heads, seq, attention-d
        ctx.queries = torch.matmul(self.W_query, ctx.embedded_repeated).permute(0, 1, 3, 2)
        ctx.keys = torch.matmul(self.W_key, ctx.embedded_repeated).permute(0, 1, 3, 2)
        ctx.values = torch.matmul(self.W_value, ctx.embedded_repeated).permute(0, 1, 3, 2)

        # batch, num-heads, seq, seq
        ctx.attention_scores = torch.matmul(ctx.queries, ctx.keys.permute(0, 1, 3, 2)) / self.scale

        # batch, num-heads, seq, seq
        match self.attention_weights_mode:
            case "overtaking-sigmoid": # novel
                sigmoid_domain = 10
                attention_scores_max = torch.max(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_min = torch.min(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_normalized = (ctx.attention_scores - attention_scores_min) / (attention_scores_max - attention_scores_min)
                sigmoid_selector = torch.max(attention_scores_normalized, dim=-1).values.unsqueeze(-1)
                ctx.attention_weights = (
                    (1 - sigmoid_selector) * torch.softmax(attention_scores_normalized, dim=-1) +
                    sigmoid_selector * self.sigmoid(attention_scores_normalized * 2 * sigmoid_domain - sigmoid_domain)
                )

            case "softmax":
                ctx.attention_weights = torch.softmax(ctx.attention_scores, dim=-1) # canonical implementation

        # batch, num-heads, seq, attention-v-d
        ctx.output_valued = torch.matmul(ctx.attention_weights, ctx.values)

        # batch, seq, 1
        ctx.output_final = self.sigmoid(
            self.fc(
                ctx.output_valued
                    .permute(0, 2, 1, 3).reshape(batch_size * seq_len, -1)
            ).reshape(batch_size, seq_len)
        )
        assert ctx.output_final.shape == (batch_size, seq_len)
        
        if observe_fn is not None:
            observe_fn(ctx)

        # last or first ones (BOS or EOS) are good for utterance meaning
        return ctx.output_final[:, -1] 
