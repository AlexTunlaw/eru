# ----------------------------------------------------------------------------
# An implementation of Levenstein distance DP algo that make a few operations
# more ergonomic, e.g. cost fn, exchanges, etc.
# Before using this, consider vs the more common and richer levenshtein library:
# `pip install levenstein`

class Levenstein:

    # ------------------------------------------------------------------------

    default_cost_fn = lambda a, b: 1.0
    zero_cost_fn = lambda a: 0.0

    # ------------------------------------------------------------------------

    @classmethod
    def compute_edit_matrix(self, seq_a, seq_b, cost_fn=None):

        ins_cost = del_cost = sub_cost = self.default_cost_fn
        keep_cost = self.zero_cost_fn
        if cost_fn:
            if isinstance(cost_fn, dict):
                ins_cost = cost_fn.get("i", ins_cost)
                del_cost = cost_fn.get("d", del_cost)
                sub_cost = cost_fn.get("s", sub_cost)
                keep_cost = cost_fn.get("-", keep_cost)
            else:
                ins_cost = del_cost = sub_cost = cost_fn

        len_a = len(seq_a)
        len_b = len(seq_b)

        d = [[(0, "")] * (len_b + 1) for __i in range(len_a + 1)]

        for i in range(1, len_a + 1):
            d[i][0] = (
                d[i - 1][0][0] + del_cost(seq_a[i - 1], None),
                d[i - 1][0][1] + "d",
            )
        for j in range(1, len_b + 1):
            d[0][j] = (
                d[0][j - 1][0] + ins_cost(None, seq_b[j - 1]),
                d[0][j - 1][1] + "i",
            )

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                d[i][j] = min(
                    (
                        d[i - 1][j][0] + del_cost(seq_a[i - 1], seq_b[j - 1]),
                        d[i - 1][j][1] + "d",
                    ),
                    (
                        d[i][j - 1][0] + ins_cost(seq_a[i - 1], seq_b[j - 1]),
                        d[i][j - 1][1] + "i",
                    ),
                    (
                        (
                            d[i - 1][j - 1][0] + keep_cost(seq_a[i - 1]),
                            d[i - 1][j - 1][1] + "-",
                        )
                        if seq_a[i - 1] == seq_b[j - 1]
                        else (
                            d[i - 1][j - 1][0] + sub_cost(seq_a[i - 1], seq_b[j - 1]),
                            d[i - 1][j - 1][1] + "s",
                        )
                    ),
                )
        return d

    # ------------------------------------------------------------------------

    @classmethod
    def compute_edit_ops(self, seq_a, seq_b, cost_fn=None):
        return self.compute_edit_matrix(seq_a, seq_b, cost_fn)[-1][-1]

    # ------------------------------------------------------------------------

    @classmethod
    def compute_edits(self, seq_a, seq_b, cost_fn=None, output_no_edit=False):

        d = self.compute_edit_matrix(seq_a, seq_b, cost_fn)

        ops = d[-1][-1][1]
        edits = []
        i = j = 0
        for k in range(0, len(ops)):
            if ops[k] == "-":
                if output_no_edit:
                    edits.append(("-", k, seq_a[i], None))
                i += 1
                j += 1
            elif ops[k] == "s":
                edits.append(("s", k, seq_a[i], seq_b[j]))
                i += 1
                j += 1
            elif ops[k] == "d":
                edits.append(("d", k, seq_a[i], None))
                i += 1
            elif ops[k] == "i":
                assert ops[k] == "i"
                edits.append(("i", k, seq_b[j], None))
                j += 1

        return edits

    # ------------------------------------------------------------------------

    @classmethod
    def compute_combined_edits(self, seq_a, seq_b, cost_fn=None, output_no_edit=False):

        edits = self.compute_edits(seq_a, seq_b, cost_fn, output_no_edit=output_no_edit)
        if len(edits) == 0:
            return edits

        combined_edits = []
        previous_op = None
        previous_op_k = None
        previous_op_b = None
        previous_k = -1
        running_a = None
        for op, k, a, b in edits:
            if op == previous_op and k == previous_k + 1 and b == None:
                running_a += a
            else:
                if previous_k != -1:
                    combined_edits.append(
                        (previous_op, previous_op_k, running_a, previous_op_b)
                    )

                previous_op = op
                previous_op_k = k
                running_a = a
                previous_op_b = b

            previous_k = k

        combined_edits.append((previous_op, previous_op_k, running_a, previous_op_b))

        return combined_edits

    # ------------------------------------------------------------------------
    # this introduces an additional operator:
    # "e", a, b - exchange/swap a and b
    # For example, "A[BCDE" vs "ABCD[E", will have an op ("e", "[", "BCD")
    # to capture the bracket movement

    @classmethod
    def compute_combined_edits_2(
        self, seq_a, seq_b, cost_fn=None, is_easily_movable_item_fn=None
    ):

        if is_easily_movable_item_fn:
            assert not cost_fn

            def easily_movable_aware_op_cost_fn(a, b):
                if a and is_easily_movable_item_fn(a):
                    return 0.000001
                if b and is_easily_movable_item_fn(b):
                    return 0.000001
                return 1.0

            cost_fn = easily_movable_aware_op_cost_fn

        edits = self.compute_combined_edits(
            seq_a, seq_b, cost_fn=cost_fn, output_no_edit=True
        )
        if len(edits) == 0:
            return edits

        edits_2 = []
        for i, edit in enumerate(edits):
            if i >= 2:
                new_edit = None
                if (
                    edits_2[-2][0] == "i"
                    and edits_2[-1][0] == "-"
                    and edit[0] == "d"
                    and edits_2[-2][2] == edit[2]
                ):
                    new_edit = ("e", edits_2[-2][1], edit[2], edits_2[-1][2])
                elif (
                    edits_2[-2][0] == "d"
                    and edits_2[-1][0] == "-"
                    and edit[0] == "i"
                    and edits_2[-2][2] == edit[2]
                ):
                    new_edit = ("e", edits_2[-2][1], edits_2[-1][2], edit[2])
                if new_edit:
                    edits_2 = edits_2[:-2] + [new_edit]
                    continue
            edits_2.append(edit)

        return edits_2
