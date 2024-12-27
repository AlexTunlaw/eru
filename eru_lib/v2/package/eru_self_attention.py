import math

import torch

# ---------------------------------------------------------------------------

def custom_base_softmax(x, base=2, dim=-1):
    x_exp = torch.pow(base, x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

# ---------------------------------------------------------------------------
# based on:
# https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
# https://github.com/sooftware/attentions/blob/master/attentions.py
#

# ---------------------------------------------------------------------------

class SelfAttention(torch.nn.Module):

    # -----------------------------------------------------------------------

    def __init__(self, input_dim, attention_dim, output_dim, c_heads, desc:str=None, sharpening_mode="softmax", alignment_mode="dot-product"):

        super().__init__()

        self.layer_norm = torch.nn.LayerNorm((input_dim,))
        self.W_key = torch.nn.Parameter(torch.rand(c_heads, attention_dim, input_dim))
        self.W_query = torch.nn.Parameter(torch.rand(c_heads, attention_dim, input_dim))
        self.W_value = torch.nn.Parameter(torch.rand(c_heads, output_dim, input_dim))
        self.scale = math.sqrt(input_dim)
        self.desc = desc

        self.sharpening_mode = sharpening_mode
        self.alignment_mode = alignment_mode
        self.overtaking_sigmoid = torch.nn.Sigmoid() if self.sharpening_mode != "softmax" else None
        return

    # -----------------------------------------------------------------------

    def forward(self, input, ctx=None, observe_fn=None):

        # batch, num-heads, seq, input-d
        input_normalized = self.layer_norm(input)

        # W: num-heads, attention-d, input-d
        # query, keys, values:
        # -> batch, num-heads, seq, attention-d
        queries = torch.matmul(self.W_query, input_normalized.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        keys = torch.matmul(self.W_key, input_normalized.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        values = torch.matmul(self.W_value, input_normalized.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        # -> batch, num-heads, seq, seq
        match self.alignment_mode:
            case "dot-product": # classic/textbook
                attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / self.scale
            case "mse": # novel
                attention_scores = torch.cdist(keys, queries) / self.scale

        # -> batch, num-heads, seq, seq
        match self.sharpening_mode:

            case "softmax": # classic/textbook
                attention_weights = torch.softmax(attention_scores, dim=-1)

            case "softmax-temperature-loss-guided": # novel
                previous_loss = ctx["previous-loss"]
                min_temperature, max_temperature = 1.1, torch.e
                previous_loss_effective = previous_loss if previous_loss is not None else 1.0
                soft_temperature_selector = previous_loss_effective
                temperature_effective = (
                    min_temperature * soft_temperature_selector +         # start with close to min temperature (when loss is high)
                    max_temperature * (1.0 - soft_temperature_selector)   # finish with max temperature (when loss is low)
                )
                attention_weights = custom_base_softmax(attention_scores, base=temperature_effective, dim=-1)

            case "sigmoid": # not robust for simple eru languages (e.g. bigram-based binary classification)
                sigmoid_domain = 5
                attention_scores_max = torch.max(attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_min = torch.min(attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_normalized = (attention_scores - attention_scores_min) / (attention_scores_max - attention_scores_min)
                attention_weights = \
                    self.overtaking_sigmoid(attention_scores_normalized * 2 * sigmoid_domain - sigmoid_domain)

            case "overtaking-sigmoid-tuned": # novel
                sigmoid_domain = 5
                attention_scores_max = torch.max(attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_min = torch.min(attention_scores, dim=-1).values.unsqueeze(-1)
                attention_scores_normalized = (attention_scores - attention_scores_min) / (attention_scores_max - attention_scores_min)
                softmax_selector = 1.0 - torch.sigmoid(
                    (1.0 - torch.max(attention_scores_normalized, dim=-1).values.unsqueeze(-1)) * \
                        2.0 * sigmoid_domain + sigmoid_domain
                )
                sigmoid_selector = 1.0 - softmax_selector
                attention_weights = \
                    softmax_selector * torch.softmax(attention_scores_normalized, dim=-1) + \
                    sigmoid_selector * self.overtaking_sigmoid(attention_scores_normalized * 2 * sigmoid_domain - sigmoid_domain)

        # -> batch, num-heads, seq, output-d
        output_valued = torch.matmul(attention_weights, values)

        if observe_fn is not None:
            observe_fn(ctx={
                "kind": f"self-attention {self.desc}",
                "queries": queries,
                "keys": keys,
                "values": values,
                "attention-scores": attention_scores,
                "attention-weights": attention_weights,
                "output-valued": output_valued
            })

        return output_valued
