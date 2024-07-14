import csv
import numpy as np

# ----------------------------------------------------------------------------
# Helpers (CsvFile, CsvSchema, CsvLine) for lightweight manipulation of csv files.
# For use cases, see csv_helpers_tests.py

class CsvSchema:

    # ------------------------------------------------------------------------

    @classmethod
    def from_lines(cls, lines, additional_columns=None):
        return lines[0].schema.copy(additional_columns=additional_columns)

    # ------------------------------------------------------------------------

    def __init__(self, columns=None):
        self.columns = columns
        if not self.columns:
            self.columns = []
        self.column_indices_by_name = dict((t[1], t[0]) for t in enumerate(self.columns))

    # ------------------------------------------------------------------------

    def copy(self, additional_columns=None):
        new_columns = self.columns.copy()
        if additional_columns is not None:
            assert isinstance(additional_columns, list) or isinstance(additional_columns, tuple)
            assert all(column not in self.columns for column in additional_columns)
            new_columns.extend(additional_columns)
        return CsvSchema(columns=new_columns)

    # ------------------------------------------------------------------------

    def __contains__(self, other):
        return all(column in self.columns for column in other.columns)

    # ------------------------------------------------------------------------

    @classmethod
    def from_file(self, file_name, delimiter=None):
        with open(file_name, newline="") as csv_file:
            csv_reader = (
                csv.reader(csv_file, delimiter=delimiter)
                if delimiter
                else csv.reader(csv_file)
            )
            for line in csv_reader:
                return CsvSchema(line)

    # ------------------------------------------------------------------------

    def get_column_index(self, column_name):
        assert column_name in self.column_indices_by_name, f"Schema mismatch: '{column_name}' not in {', '.join(self.column_indices_by_name.keys())}"
        return self.column_indices_by_name[column_name]

    # ------------------------------------------------------------------------

    def add_column(self, column_name):
        self.ensure_column_index(column_name)

    # ------------------------------------------------------------------------

    def ensure_column_index(self, column_name):
        index = self.column_indices_by_name.get(column_name)
        if index == None:
            index = len(self.columns)
            self.column_indices_by_name[column_name] = index
            self.columns.append(column_name)
        return index

    # ------------------------------------------------------------------------

    def get_column_name(self, column_index):
        return self.columns[column_index]

    # ------------------------------------------------------------------------

    def get_column_names(self):
        return self.columns

    # ------------------------------------------------------------------------

    def get_column_count(self):
        return len(self.columns)

# ----------------------------------------------------------------------------

class CsvLine:

    # ------------------------------------------------------------------------

    @classmethod
    def from_line(cls, schema, source_line, additional_values=None):
        if schema is None:
            schema = source_line.schema
        line = CsvLine(schema=schema, values=source_line.get_values_as_dict())
        if additional_values:
            line.add_values(additional_values)
        return line

    # ------------------------------------------------------------------------

    def __init__(self, schema=None, values=None, assert_lengths=True):

        self.schema = schema
        if not self.schema:
            self.schema = CsvSchema()

        if not values:
            self.values = [""] * self.schema.get_column_count()
        elif isinstance(values, dict):
            self.values = [""] * self.schema.get_column_count()
            for key, value in values.items():
                self[key] = value
        else:
            self.values = [self.format_value(value) for value in values]

        if assert_lengths:
            assert len(self.values) == self.schema.get_column_count()

    # ------------------------------------------------------------------------

    def clear_columns(self, columns):
        for column in columns:
            self[column] = ""

    # ------------------------------------------------------------------------

    def copy(self, additional_values=None):
        new_line = CsvLine(self.schema.copy(), self.values.copy())
        if additional_values:
            for k, v in additional_values.items():
                new_line[k] = v
        return new_line

    # ------------------------------------------------------------------------

    def create_new_line(self, new_line_schema):
        new_line = CsvLine(new_line_schema)
        for column in self.schema.columns:
            new_line[column] = self[column]
        return new_line

    # ------------------------------------------------------------------------

    def format_value(self, value):
        if isinstance(value, list):
            return " ".join(self.format_value(value_item) for value_item in value)
        return str(value)

    # ------------------------------------------------------------------------

    def get_value(self, column_name, default_value=None, remap_columns=None):
        if remap_columns:
            column_name = remap_columns.get(column_name, column_name)
        return self[column_name] if column_name in self.schema.columns else default_value

    # ------------------------------------------------------------------------

    def get_values(self, column_names=None, default_value=None, remap_columns=None):
        
        if (
            column_names is None and
            default_value is None and
            remap_columns is None
        ):
            return self.values
        
        return [
            self.get_value(column_name, default_value=default_value, remap_columns=remap_columns)
            for column_name in column_names
        ]

    # ------------------------------------------------------------------------

    def get_values_as_dict(self):
        return dict((k, v) for k, v in zip(self.schema.columns, self.values))

    # ------------------------------------------------------------------------

    def get_bool_value(self, column_name):
        s = self.get_value(column_name, default_value="FALSE")
        if not s:
            return False
        if s == "TRUE":
            return True
        if s == "FALSE":
            return False
        return bool(eval(s))

    # ------------------------------------------------------------------------

    def write(self, csv_writer):
        csv_writer.writerow(self.get_values())

    # ------------------------------------------------------------------------

    def add_values(self, values):
        if isinstance(values, dict):
            values = list(values.items())

        if (
            isinstance(values, tuple)
            and len(values) == 2
            and isinstance(values[0], list)
            and isinstance(values[1], list)
        ):
            if not len(values[0]) == len(values[1]):
                raise BaseException("Expecting both list have the same length.")
            for column_name, value in zip(values[0], values[1]):
                self[column_name] = value
            return
        elif isinstance(values, list):
            for item in values:
                if not (isinstance(item, tuple) and len(item) == 2):
                    raise BaseException("Can" "t handle: %r" & item)
                self[item[0]] = item[1]
            return
        elif isinstance(values, tuple) and len(values) == 2:
            self[values[0]] = values[1]
            return

        raise BaseException("Can" "t handle: %r" & values)

    # ------------------------------------------------------------------------

    def with_added_values(self, values):
        copy = self.copy()
        copy.add_values(values=values)
        return copy

    # ------------------------------------------------------------------------

    def copy_values(self, source, keys):
        for key in keys:
            self[key] = source[key]

    # ------------------------------------------------------------------------

    def __setitem__(self, column_name, value):
        value_formatted = self.format_value(value)
        column_index = self.schema.ensure_column_index(column_name)
        if column_index < len(self.values):
            self.values[column_index] = value_formatted
        else:
            assert column_index == len(self.values)
            self.values.append(value_formatted)

    # ------------------------------------------------------------------------

    def __getitem__(self, column_name):
        column_index = self.schema.get_column_index(column_name)
        return self.values[column_index]

# ----------------------------------------------------------------------------

class CsvFile:

    # ------------------------------------------------------------------------
    # other useful encodings: "ISO-8859-1"

    @classmethod
    def get_lines_from_csv(
        cls, file_name, csv_schema, assert_schema=False, delimiter=None, header_line=True,
        create_schema_from_header_line=False, encoding="utf-8-sig", csv_file=None
    ):
        return list(cls.get_lines_from_csv_iter(
            file_name, csv_schema,
            assert_schema=assert_schema,
            delimiter=delimiter,
            header_line=header_line,
            create_schema_from_header_line=create_schema_from_header_line,
            encoding=encoding,
            csv_file=csv_file
        ))
        
    # ------------------------------------------------------------------------
    # other useful encodings: "ISO-8859-1"

    @classmethod
    def get_lines_from_csv_iter(
        cls, file_name, csv_schema, assert_schema=False, delimiter=None, header_line=True,
        create_schema_from_header_line=False, encoding="utf-8-sig", csv_file=None
    ):
        if file_name is not None:
            assert csv_file is None
            with open(file_name, newline="", encoding=encoding) as csv_file:
                yield from cls.get_lines_from_csv_iter_core(
                    csv_file, csv_schema, assert_schema=assert_schema, delimiter=delimiter, header_line=header_line,
                    create_schema_from_header_line=create_schema_from_header_line, encoding=encoding
                )
        else:
            assert csv_file is not None
            yield from cls.get_lines_from_csv_iter_core(
                csv_file, csv_schema, assert_schema=assert_schema, delimiter=delimiter, header_line=header_line,
                create_schema_from_header_line=create_schema_from_header_line, encoding=encoding
            )

    # ------------------------------------------------------------------------

    @classmethod
    def get_lines_from_csv_iter_core(
        cls, csv_file, csv_schema, assert_schema, delimiter, header_line,
        create_schema_from_header_line, encoding
    ):

        csv_reader = (
            csv.reader(csv_file, delimiter=delimiter)  # dialect=csv.excel
            if delimiter
            else csv.reader(csv_file)
        )

        schema_line = header_line

        for line in csv_reader:

            if schema_line:
                if assert_schema:
                    assert line == csv_schema.get_column_names()
                if create_schema_from_header_line:
                    csv_schema=CsvSchema(columns=line)
                schema_line = False
                continue

            csv_line = CsvLine(csv_schema, line) if csv_schema else line
            values_count = sum(1 for v in csv_line.values if v)
            if values_count:
                yield csv_line

    # ------------------------------------------------------------------------

    @classmethod
    def save_to_csv(
        cls,
        file_name,
        data_items,
        csv_schema,
        header_row=True,
        get_values_fn=None,
        delimiter=None,
        stream=None
    ):
        if file_name is not None:
            assert stream is None
            with open(file_name, "w") as file:
                cls.save_to_csv_core(file, data_items, csv_schema,
                    header_row=header_row, get_values_fn=get_values_fn, delimiter=delimiter
                )
        else:
            assert stream is not None
            cls.save_to_csv_core(stream, data_items, csv_schema,
                header_row=header_row, get_values_fn=get_values_fn, delimiter=delimiter
            )

    # ------------------------------------------------------------------------

    @classmethod
    def save_to_csv_core(
        cls,
        file_or_stream,
        data_items,
        csv_schema,
        header_row=True,
        get_values_fn=None,
        delimiter=None,
    ):
        csv_writer = (
            csv.writer(file_or_stream, delimiter=delimiter)
            if delimiter
            else csv.writer(file_or_stream)
        )
        if header_row and csv_schema:
            csv_writer.writerow(csv_schema.get_column_names())
        for item in data_items:
            if csv_schema:
                line = (
                    item
                    if isinstance(item, CsvLine)
                    else CsvLine(csv_schema, get_values_fn(item))
                )
                line.write(csv_writer)
            else:
                assert isinstance(item, list)
                csv_writer.writerow(item)

# ----------------------------------------------------------------------------
# Helpers to do train/test split for csv lines, in a stable/reproducible fashion
# ("reproducible": same result run to run as long as numpy random seed is the same)

class CsvSplit:

    # ------------------------------------------------------------------------

    def __init__(self, key_columns):
        self.key_columns = key_columns

    # -----------------------------------------------------------------------

    def get_doc_key_from_line(self, line, i_line):
        key = '|'.join([line[column_name] for column_name in self.key_columns])
        if not key or all(c == '|' for c in key):
            key = f"line_{i_line}"
        return key

    # -----------------------------------------------------------------------

    def define_split_keys(self, lines, test_fraction):

        # Note 1: this needs to be done in a way where no doc shows up simultaneously
        # both training and test set (don't allow "leaks of signal" of this nature)
        # Note 2: it is important to make this process stable (same order) run-to-run,
        # for reproducibility of results and simpler debugging (hence using the lists
        # as well as the `triaged` set)

        train_keys, test_keys = [], []
        triaged = set()
        for i_line, line in enumerate(lines):
            key = self.get_doc_key_from_line(line, i_line=i_line)
            if key in triaged:
                continue
            if np.random.rand() < test_fraction:
                test_keys.append(key)
            else:
                train_keys.append(key)
            triaged.add(key)

        return train_keys, test_keys

    # ------------------------------------------------------------------------

    def split_lines (self, lines, test_fraction, train_keys=None, test_keys=None, do_lines_shuffle=True):

        assert hasattr(lines, "__len__") # (don't accept iterators)

        assert (
            ((train_keys is None) and (test_keys is None)) or
            ((train_keys is not None) and (test_keys is not None))
        )
        if train_keys is None:
            train_keys, test_keys = self.define_split_keys(lines, test_fraction)

        train_keys_set = set(train_keys)
        test_keys_set = set(test_keys)

        lines = lines.copy()

        if do_lines_shuffle:
            np.random.shuffle(lines) # by default, shufflee; important for training process that converges smoother

        train_lines, test_lines = None, None
        for i_line, line in enumerate(lines):

            key = self.get_doc_key_from_line(line, i_line=i_line)

            if train_keys and key in train_keys_set:
                if not train_lines:
                    train_lines = []
                train_lines.append(line)

            if test_keys and key in test_keys_set:
                if not test_lines:
                    test_lines = []
                test_lines.append(line)

        return train_lines, test_lines