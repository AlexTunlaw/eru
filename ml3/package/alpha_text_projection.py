# ---------------------------------------------------------------------------
# projects text that may contain punctuation and other non-alpha characters
# onto a string where all these characters are removed. Also replaces multiple 
# spaces with a single one during projection

class AlphaTextProjection:

    # -----------------------------------------------------------------------

    def __init__(self, source: str,
        projection_filter_is_accepted=lambda c: c.isalpha() or c in ['_']
    ):
        self.source = source

        projected_raw = ['_'] * len(source)
        self.offsets_source_by_projection = [None] * (len(source) + 1)
        self.offsets_projected_by_source = [None] * (len(source) + 1)
        k, m = 0, 0
        while k < len(source):
            if projection_filter_is_accepted(source[k]):
                projected_raw[m] = source[k]
                self.offsets_source_by_projection[m] = k
                self.offsets_projected_by_source[k] = m
                m += 1
            elif (
                source[k].isspace() and
                (m == 0 or projected_raw[m - 1] != ' ')
            ):
                projected_raw[m] = ' '
                self.offsets_source_by_projection[m] = k
                self.offsets_projected_by_source[k] = m
                m += 1

            k += 1
        self.offsets_source_by_projection[m] = k
        self.offsets_projected_by_source[k] = m

        j = m
        for i in range(k, -1, -1):
            if self.offsets_projected_by_source[i] is None:
                self.offsets_projected_by_source[i] = j
            else:
                j = self.offsets_projected_by_source[i]

        self.projected = "".join(projected_raw[:m])
        return
    
    # -----------------------------------------------------------------------

    def to_source(self, o: int):
        
        if isinstance(o, int):
            return self.offsets_source_by_projection[o]
        
        assert len(o) == 2 # o assumed to be a span
        s, l = o
        return self.offsets_source_by_projection[s], self.offsets_source_by_projection[l - 1] + 1

    # -----------------------------------------------------------------------

    def to_projection(self, o: int):
        
        if isinstance(o, int):
            return self.offsets_projected_by_source[o]
        
        assert len(o) == 2 # o assumed to be a span
        s, l = o
        return self.offsets_projected_by_source[s], self.offsets_projected_by_source[l]
