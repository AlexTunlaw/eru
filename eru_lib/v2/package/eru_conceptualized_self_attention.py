import math

import torch

# ---------------------------------------------------------------------------

def custom_base_softmax(x, base=2, dim=-1):
    x_exp = torch.pow(base, x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

# ---------------------------------------------------------------------------

class ConceptualizedSelfAttention(torch.nn.Module):

    # -----------------------------------------------------------------------

    def __init__(self, input_dim, c_conceptualizations, output_dim, c_heads, desc:str=None):

        super().__init__()

        self.layer_norm = torch.nn.LayerNorm((input_dim,))
        self.W_value = torch.nn.Parameter(torch.rand(c_heads, c_conceptualizations, output_dim, input_dim))
        self.W_key_fc = torch.nn.Linear(input_dim, c_conceptualizations)
        self.scale = math.sqrt(input_dim)
        self.desc = desc

        self.sigmoid = torch.nn.Sigmoid()
        return

    # -----------------------------------------------------------------------

    def forward(self, input, ctx=None, observe_fn=None):

        # batch, num-heads, seq, input-d
        input_normalized = self.layer_norm(input)

        # W: num-heads, c-conceptualizations, input-d
        # query, keys, values:
        # -> batch, num-heads, seq, c-conceptualizations
        conceptualizations_seq = self.sigmoid(self.W_key_fc(input))

        # -> batch, num-heads, c-conceptualizations
        conceptualizations = torch.softmax(
            torch.sum(conceptualizations_seq, dim=2),
            dim=-1
        ).squeeze(1)

        # -> batch, seq, c-conceptualizations, output-d
        assert self.W_value.shape[0] == 1 # note the below line doesn't deal correctly with num-heads, hence only accepting 1
        values = torch.matmul(input_normalized, self.W_value).permute(0, 2, 1, 3)

        # -> batch, num-heads, seq, output-d
        output_representation = torch.einsum('ai,abij->abj', conceptualizations, values).unsqueeze(1)

        return output_representation
