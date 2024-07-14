from .tests.csv_helpers_tests import CsvHelpersTests
from .tests.levenstein_tests import LevensteinTests
from .tests.tri_word_hash_tests import TriWordHashTests
from .tests.alpha_text_projection_tests import AlphaTextProjectionTests

def run_all_tests():

    CsvHelpersTests().run()
    LevensteinTests().run()
    TriWordHashTests().run()
    AlphaTextProjectionTests().run()