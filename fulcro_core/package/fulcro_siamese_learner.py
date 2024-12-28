import torch

# ---------------------------------------------------------------------------

class FulcroSiameseLearner(torch.nn.Module):
    
    # -----------------------------------------------------------------------

    def __init__(self, input_dim, contrastive_head_dim, dtype=torch.float32, dropout=None):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, contrastive_head_dim, dtype=dtype)
        self.dropout1 = torch.nn.Dropout(dropout) if dropout not in [None, "None"] else None
        self.centroids = None
    
    # -----------------------------------------------------------------------

    def forward_1(self, x):
        y = self.dropout1(self.fc1(x)) if self.dropout1 is not None else self.fc1(x)
        return y
    
    # -----------------------------------------------------------------------

    def forward(self, x_1, x_2):
        x_1_out = self.forward_1(x_1)
        x_2_out = self.forward_1(x_2)
        return x_1_out, x_2_out

    # -----------------------------------------------------------------------

    def get_save_data(self):
        return (self.state_dict(), self.centroids)

    # -----------------------------------------------------------------------

    def save(self, file):
        torch.save(self.get_save_data(), file)

    # -----------------------------------------------------------------------

    def load_from_save_data(self, save_data):
        model_params, self.centroids = save_data
        self.load_state_dict(model_params)
        self.eval() # important

    # -----------------------------------------------------------------------

    def load(self, file):
        save_data = torch.load(file)
        self.load_from_save_data(save_data)

    # -----------------------------------------------------------------------

    def get_centroid_scores(self, example):

        assert self.centroids

        sim_fn = torch.nn.CosineSimilarity(dim=0)
        with torch.no_grad():
            projected = self.forward_1(example)
        assert all(projected.shape == centroid.shape for centroid in self.centroids)
        scores = [
            sim_fn(centroid, projected).item()
            for centroid in self.centroids
        ]
        return scores