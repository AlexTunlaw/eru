from .. import CsvSchema, CsvLine

class CsvHelpersTests:

    # ------------------------------------------------------------------------

    def __init__(self):
        pass

    # ------------------------------------------------------------------------

    def run(self):

        self.test_basic_schema()
        self.test_basic_line_write()
        self.test_basic_dynamic_line()
        self.test_basic_line_read()
        self.test_bulk_line_add()
        self.test_bulk_line_add_2()

    # ------------------------------------------------------------------------

    def test_basic_schema(self):

        schema = CsvSchema(["foo", "bar", "splat"])
        assert schema.get_column_index("foo") == 0
        assert schema.get_column_index("bar") == 1
        assert schema.get_column_index("splat") == 2
        assert schema.get_column_name(0) == "foo"
        assert schema.get_column_name(1) == "bar"
        assert schema.get_column_name(2) == "splat"
        assert schema.get_column_names() == ["foo", "bar", "splat"]

        print("CsvHelpersTests.test_basic_schema")

    # ------------------------------------------------------------------------

    def test_basic_line_write(self):

        schema = CsvSchema(["foo", "bar", "splat"])

        line = CsvLine(schema)
        assert line["foo"] == ""
        assert line["bar"] == ""
        assert line["splat"] == ""
        assert line.get_values() == ["", "", ""]

        line["foo"] = 1
        line["splat"] = 3
        assert line["foo"] == "1"
        assert line["bar"] == ""
        assert line["splat"] == "3"
        assert line.get_values() == ["1", "", "3"]

        print("CsvHelpersTests.test_basic_line_write")

    # ------------------------------------------------------------------------

    def test_basic_dynamic_line(self):

        line = CsvLine()
        assert line.get_values() == []
        assert line.schema.get_column_names() == []

        line["foo"] = 1
        line["bar"] = 2
        assert line["foo"] == "1"
        assert line["bar"] == "2"
        assert line.get_values() == ["1", "2"]
        assert line.schema.get_column_names() == ["foo", "bar"]

        print("CsvHelpersTests.test_basic_dynamic_line")

    # ------------------------------------------------------------------------

    def test_basic_line_read(self):

        line = CsvLine(CsvSchema(["foo", "bar"]), values=[1, 2])
        assert line["foo"] == "1"
        assert line["bar"] == "2"
        assert line.get_values() == ["1", "2"]
        assert line.schema.get_column_names() == ["foo", "bar"]

        print("CsvHelpersTests.test_basic_line_read")

    # ------------------------------------------------------------------------

    def test_bulk_line_add(self):

        line = CsvLine(CsvSchema(["foo", "bar"]))
        line.add_values({"foo": 123, "bar": ["bb", "aa", "rr"]})
        assert line.get_values() == ["123", "bb aa rr"]

        print("CsvHelpersTests.test_bulk_line_add")

    # ------------------------------------------------------------------------

    def test_bulk_line_add_2(self):

        line = CsvLine(CsvSchema(["foo", "bar"]))
        line.add_values((["foo", "bar"], [123, ["bb", "aa", "rr"]]))
        assert line.get_values() == ["123", "bb aa rr"]

        print("CsvHelpersTests.test_bulk_line_add_2")
