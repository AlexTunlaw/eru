from .. import Levenstein

# ---------------------------------------------------------------------------

class LevensteinTests:

    # -----------------------------------------------------------------------

    def run(self):

        self.test_basics_1()
        self.test_bracket_moves_1()
        self.test_bracket_moves_2()

    # -----------------------------------------------------------------------

    def test_basics_1(self):

        assert Levenstein.compute_edit_ops("ABC", "AC") == (1.0, "-d-")
        assert Levenstein.compute_edit_ops("ABC", "ABBC") == (1.0, "--i-")
        assert Levenstein.compute_edit_ops("ABC", "AXC") == (1.0, "-s-")
        assert Levenstein.compute_edit_ops("ABC", "ACB") == (2.0, "-d-i")

        assert Levenstein.compute_edits("ABC", "AC") == [("d", 1, "B", None)]
        assert Levenstein.compute_edits("ABC", "ABBC") == [("i", 2, "B", None)]
        assert Levenstein.compute_edits("ABC", "AXC") == [("s", 1, "B", "X")]
        assert Levenstein.compute_edits("ABC", "ACB") == [("d", 1, "B", None), ("i", 3, "B", None)]
        assert (
            Levenstein.compute_edits("AC", "ABBBC") ==
            [
                ("i", 1, "B", None), ("i", 2, "B", None), ("i", 3, "B", None)
            ]
        )

        assert Levenstein.compute_combined_edits("AC", "ABBBC") == [("i", 1, "BBB", None)]
        assert (
            Levenstein.compute_combined_edits("ACDEFGHIJKKK", "ABBBCDEFGHIJ") ==
            [
                ("i", 1, "BBB", None),
                ("d", 12, "KKK", None),
            ]
        )

        print("LevensteinTests.test_basics_1")

    # -----------------------------------------------------------------------

    def test_bracket_moves_1(self):

        def bracket_friendly_op_cost_fn(a, b):
            if a in ["[", "]"] or b in ["[", "]"]:
                return 0.001
            return 1.
        cost_fn = bracket_friendly_op_cost_fn

        # these tests ensure that we always move the brackets and not other chars
        # distance 1
        assert (
            Levenstein.compute_edits("A[BCDE", "AB[CDE", cost_fn=cost_fn)
            == [("d", 1, "[", None), ("i", 3, "[", None)]
        )
        assert (
            Levenstein.compute_edits("AB[CDE", "A[BCDE", cost_fn=cost_fn)
            == [("i", 1, "[", None), ("d", 3, "[", None)]
        )

        assert (
            Levenstein.compute_edits("ABC]DE", "ABCD]E", cost_fn=cost_fn)
            == [("d", 3, "]", None), ("i", 5, "]", None)]
        )
        assert (
            Levenstein.compute_edits("ABCD]E", "ABC]DE", cost_fn=cost_fn)
            == [("i", 3, "]", None), ("d", 5, "]", None)]
        )

        # same thing, distance 2
        assert (
            Levenstein.compute_edits("A[BCDE", "ABCD[E", cost_fn=cost_fn)
            == [("d", 1, "[", None), ("i", 5, "[", None)]
        )
        assert (
            Levenstein.compute_edits("ABCD[E", "A[BCDE", cost_fn=cost_fn)
            == [("i", 1, "[", None), ("d", 5, "[", None)]
        )

        assert (
            Levenstein.compute_edits("A]BCDE", "ABCD]E", cost_fn=cost_fn)
            == [("d", 1, "]", None), ("i", 5, "]", None)]
        )
        assert (
            Levenstein.compute_edits("ABCD]E", "A]BCDE", cost_fn=cost_fn)
            == [("i", 1, "]", None), ("d", 5, "]", None)]
        )

        print("LevensteinTests.test_bracket_moves_1")

    # -----------------------------------------------------------------------

    def test_bracket_moves_2(self):

        is_easily_movable_item_fn = lambda a: a in ["[", "]", "(", ")"]

        # combined edits
        assert (
            Levenstein.compute_combined_edits_2("ABCD]E", "A]BCDE", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [("-", 0, "A", None), ("e", 1, "]", "BCD"), ("-", 6, "E", None)]
        )
        assert (
            Levenstein.compute_combined_edits_2("A]BCDE", "ABCD]E", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [("-", 0, "A", None), ("e", 1, "BCD", "]"), ("-", 6, "E", None)]
        )

        assert (
            Levenstein.compute_combined_edits_2("ABC[D]EFG", "A[BCDEF]G", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [("-", 0, "A", None), ("e", 1, "[", "BC"), ("-", 5, "D", None), ("e", 6, "EF", "]"), ("-", 10, "G", None)]
        )
        assert (
            Levenstein.compute_combined_edits_2("A[BCDEF]G", "ABC[D]EFG", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [("-", 0, "A", None), ("e", 1, "BC", "["), ("-", 5, "D", None), ("e", 6, "]", "EF"), ("-", 10, "G", None)]
        )

        assert (
            Levenstein.compute_combined_edits_2("ABC[DEF]GHI", "A[BCDEFGH]I", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [("-", 0, "A", None), ("e", 1, "[", "BC"), ("-", 5, "DEF", None), ("e", 8, "GH", "]"), ("-", 12, "I", None)]
        )

        assert (
            Levenstein.compute_combined_edits_2("AB[CD]EFG", "ABC[DEF]G", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [("-", 0, "AB", None), ("e", 2, "C", "["), ("-", 5, "D", None), ("e", 6, "EF", "]"), ("-", 10, "G", None)]
        )

        assert (
            Levenstein.compute_combined_edits_2("[AB[CD]EF]", "[AB[CDEF]]", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [("-", 0, "[AB[CD", None), ("e", 6, "EF]", "]")]
        )

        # bracket exchanges
        assert (
            Levenstein.compute_combined_edits_2("[AB[CD]EF]", "(AB[CD]EF)", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [('s', 0, '[', '('), ('-', 1, 'AB[CD]EF', None), ('s', 9, ']', ')')]
        )
        assert (
            Levenstein.compute_combined_edits_2("[AB[CD]EF]", "[AB(CD)EF]", is_easily_movable_item_fn=is_easily_movable_item_fn)
            == [('-', 0, '[AB', None), ('s', 3, '[', '('), ('-', 4, 'CD', None), ('s', 6, ']', ')'), ('-', 7, 'EF]', None)]
        )

        print("LevensteinTests.test_bracket_moves_2")
