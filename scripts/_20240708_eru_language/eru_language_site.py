import os
from pathlib import Path

# ---------------------------------------------------------------------------

class EruLanguageSite:

    # -----------------------------------------------------------------------

    def __init__(self, data_dir):

        self.projects_home = Path(os.environ["PROJECTS_HOME"]).resolve()
        self.data_directory = self.projects_home.joinpath(data_dir)
        self.data_directory.mkdir(exist_ok=True)

        self.oai_completions_cache_dir = self.projects_home.joinpath("_data/oai-completions-cache")
        self.oai_embeddings_cache_dir = self.projects_home.joinpath("_data/oai-embeddings-cache")
        return
