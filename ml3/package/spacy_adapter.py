import sys

# ---------------------------------------------------------------------------

class SpacyAdapter:

    # -----------------------------------------------------------------------

    def __init__(self, model='en_core_web_sm'):

        if "spacy" not in sys.modules:
            import spacy

        self.nlp = spacy.load(model)

    # -----------------------------------------------------------------------

    def process_text(self, text):
        return self.nlp(text)
