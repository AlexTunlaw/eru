import numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.colors import _colors_full_map as colors_full_map

from .observer_base import ObserverBase

# ---------------------------------------------------------------------------

class ProjectionsObserver(ObserverBase):

    # -----------------------------------------------------------------------

    def __init__(self, observed_layers):

        super().__init__()
        self.observed_layers = observed_layers
        return

    # -----------------------------------------------------------------------
    # Note: in this experiment, tokens 1 and 2 carry signal; all others are random.
    # We use token 3 as a representative of noise.

    def observe(self, ctx):

        i_sample = 0 # we examine a single sample in every batch
        x = ctx["batch"]["x"][0]
        is_all_satisfactory_observation_targets = lambda iss: \
            all(len(is_) >= 1 for is_ in iss) #TODO what happens with attention when miltiple i_0 and i_1 are present? Does it split?

        while i_sample < x.shape[0]:
            x_values = x[i_sample].tolist()
            i1s, i2s = [], []
            for i, v in enumerate(x_values):
                match v: # in this setup, tokens 1 and 2 carry main signal - we track them. Token 3 is irrelevant and is a baseline
                    case 1: i1s.append(i)
                    case 2: i2s.append(i)
            if is_all_satisfactory_observation_targets([i1s, i2s]):
                break
            i_sample += 1

        if is_all_satisfactory_observation_targets([i1s, i2s]):

            i1, i2, iEOS = i1s[0], i2s[0], -1
            observations_cur = []
            c_heads = ctx["self-attention 0"]["keys"][i_sample].shape[0]
            observation_plan = (
                (
                    [
                        ((1, i1), "self-attention 0", "keys"),
                        ((2, i2), "self-attention 0", "keys"),
                        (("EOS", iEOS), "self-attention 0", "keys"),
                        ((1, i1), "self-attention 0", "queries"),
                        ((2, i2), "self-attention 0", "queries"),
                        (("EOS", iEOS), "self-attention 0", "queries"),
                    ]
                    if "self-attention 0" in self.observed_layers
                    else []
                ) + (
                    [
                        ((1, i1), "self-attention 1", "keys"),
                        ((2, i2), "self-attention 1", "keys"),
                        (("EOS", iEOS), "self-attention 1", "keys"),
                        ((1, i1), "self-attention 1", "queries"),
                        ((2, i2), "self-attention 1", "queries"),
                        (("EOS", iEOS), "self-attention 1", "queries"),
                        # ((1, i1), "self-attention 1", "values"),
                        # ((2, i2), "self-attention 1", "values"),
                        # (("EOS", iEOS), "self-attention 1", "values"),
                    ]
                    if "self-attention 1" in self.observed_layers
                    else []
                )
            )
            for (token, token_i), self_attention_layer, kind in observation_plan:
                for i_head in range(c_heads):
                    observations_cur.append((
                        "|".join([self_attention_layer, kind, str(i_head), str(token)]),
                        ctx[self_attention_layer][kind][i_sample][i_head][token_i].tolist(),
                    ))
            self.observations.append({
                "i-batch": ctx["batch"]["i-batch"],
                "i-sample": i_sample,
                "x": x_values,
                "projections": observations_cur
            })

        return

    # -----------------------------------------------------------------------

    def finalize(self, params):

        tsne_perplexity = 30

        tuples_flat = [
            (legend, point) # tuple layout enforcement and flattening
            for observations_cur in self.observations
            for legend, point in observations_cur["projections"]
        ]

        colors_by_legend = {}
        all_colors = list(color for color in colors_full_map.keys() if color.startswith("xkcd"))
        for legend, _point in tuples_flat:
            previous_color = colors_by_legend.get(legend)
            if previous_color is None:
                colors_by_legend[legend] = np.random.choice(all_colors)
        legends = list(colors_by_legend.keys())

        points_flat = np.array([point for _legend, point in tuples_flat])

        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=777)
        points_flat_2d = tsne.fit_transform(points_flat)

        points_flat_2d_by_legend = {
            legend: np.array([
                point_2d
                for point_2d, (legend_cur, _point) in zip(points_flat_2d, tuples_flat)
                if legend_cur == legend
            ])
            for legend in legends
        }

        def noised(xs, noise_scale: float):
            return [np.random.random() * noise_scale + x for x in xs]

        plt.figure(figsize=(8, 6))
        for legend in legends:
            plt.plot(
                noised(points_flat_2d_by_legend[legend][:, 0], noise_scale=0.01),
                noised(points_flat_2d_by_legend[legend][:, 1], noise_scale=0.01),
                marker='o',
                linestyle='--',
                c=colors_by_legend[legend],
                label=legend,
                linewidth=1,
                markersize=2,
            )
            plt.scatter(
                points_flat_2d_by_legend[legend][:, 0][-1],
                points_flat_2d_by_legend[legend][:, 1][-1],
                marker='x',
                zorder=10,
                c=colors_by_legend[legend],
            )
        plt.title('2d t-SNE of kqv')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        return
