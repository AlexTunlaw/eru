from scripts._20240708_eru_language.eru_scripts import run_scripts
run_scripts()

# from pathlib import Path

# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# data = np.random.rand(100, 10)

# tsne = TSNE(n_components=2, random_state=42)
# data_2d = tsne.fit_transform(data)

# plt.figure(figsize=(8, 6))
# plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', label='Data Points')
# plt.title('2D Projection using t-SNE')
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.legend()

# plt.savefig(Path("~/Downloads/1.png").expanduser(), format='png', dpi=300)

print("DONE")