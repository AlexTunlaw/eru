from .. import compute_tri_word_hash

# ---------------------------------------------------------------------------

class TriWordHashTests:

    # -----------------------------------------------------------------------

    def __init__(self):
        pass

    # -----------------------------------------------------------------------

    def run(self):
        assert compute_tri_word_hash("what is tri-word hash?") == "book page breakfast"
        assert compute_tri_word_hash(123) == "man slip pay"
        assert compute_tri_word_hash(124) == "fever provide thing"

        print("TriWordHashTests.basics")
