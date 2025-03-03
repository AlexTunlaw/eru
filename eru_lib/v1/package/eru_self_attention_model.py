import math
import torch

from ml3 import SimpleObject

# ---------------------------------------------------------------------------
# multi-head self-attention

class EruSelfAttentionModel(torch.nn.Module): 
    
    # -----------------------------------------------------------------------
    # based on:
    # https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
    # https://github.com/sooftware/attentions/blob/master/attentions.py
    #
    # novel: attention_weights_mode for faster convergence on the synthetic language
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

        assert attention_weights_mode in ["softmax", "sigmoid", "overtaking-sigmoid", "overtaking-sigmoid-tuned"]
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

        self.overtaking_sigmoid = torch.nn.Sigmoid()
        return

    # -----------------------------------------------------------------------

    def forward(self, x, observe_fn=None, i_batch=None):

        batch_size, seq_len = x.shape

        ctx = SimpleObject(
            i_batch=i_batch,
            x=x,
            batch_size=batch_size, seq_len=seq_len, c_heads=self.c_heads,
            embedded=None, embedded_repeated=None,
            queries=None, keys=None, values=None,
            attention_scores=None, attention_weights=None,
            output_valued=None
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
                sigmoid_domain = 5
                attention_scores_max = torch.max(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_min = torch.min(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_normalized = (ctx.attention_scores - attention_scores_min) / (attention_scores_max - attention_scores_min)
                softmax_selector = 1.0 - torch.max(attention_scores_normalized, dim=-1).values.unsqueeze(-1)
                sigmoid_selector = 1.0 - softmax_selector
                ctx.attention_weights = \
                    softmax_selector * torch.softmax(attention_scores_normalized, dim=-1) + \
                    sigmoid_selector * self.overtaking_sigmoid(attention_scores_normalized * 2 * sigmoid_domain - sigmoid_domain)

            case "overtaking-sigmoid-tuned": # novel
                sigmoid_domain = 5
                attention_scores_max = torch.max(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_min = torch.min(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_normalized = (ctx.attention_scores - attention_scores_min) / (attention_scores_max - attention_scores_min)
                softmax_selector = torch.sigmoid(
                    (1.0 - torch.max(attention_scores_normalized, dim=-1).values.unsqueeze(-1)) * \
                        2.0 * sigmoid_domain + sigmoid_domain
                )
                sigmoid_selector = 1.0 - softmax_selector
                ctx.attention_weights = \
                    softmax_selector * torch.softmax(attention_scores_normalized, dim=-1) + \
                    sigmoid_selector * self.overtaking_sigmoid(attention_scores_normalized * 2 * sigmoid_domain - sigmoid_domain)

            case "sigmoid": # novel
                sigmoid_domain = 5
                attention_scores_max = torch.max(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_min = torch.min(ctx.attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_normalized = (ctx.attention_scores - attention_scores_min) / (attention_scores_max - attention_scores_min)
                ctx.attention_weights = \
                    self.overtaking_sigmoid(attention_scores_normalized * 2 * sigmoid_domain - sigmoid_domain)
                # attention_weights_sum = torch.sum(ctx.attention_scores, dim=-1).unsqueeze(-1)
                # ctx.attention_weights = ctx.attention_weights / attention_weights_sum

            case "softmax":
                ctx.attention_weights = torch.softmax(ctx.attention_scores, dim=-1) # canonical implementation

        # batch, num-heads, seq, attention-v-d
        ctx.output_valued = torch.matmul(ctx.attention_weights, ctx.values)

        if observe_fn is not None:
            observe_fn(ctx)

        return ctx.output_valued
