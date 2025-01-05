import math

import torch

from fulcro_core import custom_base_softmax

# ---------------------------------------------------------------------------

class ConceptualizedSelfAttention(torch.nn.Module):

    # -----------------------------------------------------------------------

    def __init__(self, input_dim, c_conceptualizations, output_dim, c_heads, desc:str=None):

        super().__init__()

        self.layer_norm = torch.nn.LayerNorm((input_dim,))
        self.W_value = torch.nn.Parameter(torch.rand(c_heads, c_conceptualizations, output_dim, input_dim))
        self.W_key_fc = torch.nn.Linear(input_dim, c_conceptualizations)
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
        custom_softmax_base = None # 2 * torch.e (TODO maybe make this a param)
        softmax = (
            (lambda t: custom_base_softmax(t, base=custom_softmax_base, dim=-1))
            if custom_softmax_base is not None
            else (lambda t: torch.softmax(t, dim=-1))
        )
        conceptualizations = softmax(
            torch.sum(conceptualizations_seq, dim=2)
        ).squeeze(1)

        # -> batch, num-heads, c-conceptualizations, seq, output-d
        values = torch.matmul(
            input_normalized.unsqueeze(1),
            self.W_value.permute(1, 0, 2, 3).unsqueeze(0),
        ).permute(0, 2, 1, 3, 4)

        # -> batch, num-heads, seq, output-d
        output_representation = torch.einsum('ahc,ahcij->ahij', conceptualizations.unsqueeze(1), values)

        return output_representation
