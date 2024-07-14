from .. import AlphaTextProjection

# ---------------------------------------------------------------------------

class AlphaTextProjectionTests:

    # ------------------------------------------------------------------------

    def __init__(self):
        pass

    # ------------------------------------------------------------------------

    def run(self):

        self.test_basics_1()
        self.test_dsb_1()

    # ------------------------------------------------------------------------

    def test_basics_1(self):

        #                        0123456 7890123
        #                        012      45  78
        p = AlphaTextProjection("abc,  \n de, fg.")

        assert p.projected == "abc de fg"

        assert (p.to_source(0), p.to_source(1), p.to_source(2)) == (0, 1, 2)
        assert (p.to_source(4), p.to_source(5)) == (8, 9)
        assert (p.to_source(7), p.to_source(8)) == (12, 13)

        assert p.to_source((0, 3)) == (0, 3)
        assert p.to_source((4, 6)) == (8, 10)
        assert p.to_source((7, 9)) == (12, 14)

        assert p.to_source((0, 6)) == (0, 10)
        assert p.to_source((0, 9)) == (0, 14)

        print("AlphaTextProjectionTests.test_basics_1")
        return
    
    # ------------------------------------------------------------------------

    def test_dsb_1(self):

        #                        012345678901234567890123
        #                          01  2345678  90  12345
        p = AlphaTextProjection("[[aa]] bb cc [[dd]]. ee")
        assert p.to_projection((0, 6)) == (0, 2)
        assert p.to_projection((13, 19)) == (9, 11)

        p = AlphaTextProjection("[[aa]]")
        assert p.to_projection((0, 6)) == (0, 2)

        print("AlphaTextProjectionTests.test_dsb_1")
        return
