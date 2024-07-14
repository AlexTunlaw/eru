# ---------------------------------------------------------------------------

class ConfigChecker:

    # -----------------------------------------------------------------------
    # paths is a list of path, where each path is like an xpath, in
    # the form of: "x/y/z".
    # config is [ ] (and get-) indexable with the components of paths

    @classmethod
    def check_paths_in_config(cls, cfg, paths):

        for path in paths:
            parts = path.split("/")
            cur_level = cfg
            for part in parts:
                next_level = cur_level.get(part, None)
                if next_level is None:
                    raise Exception(f"Can't resolve path {path} in the given config.")
                cur_level = next_level

        return cfg