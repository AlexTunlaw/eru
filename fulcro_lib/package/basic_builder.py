import numpy as np
from unidecode import unidecode
import itertools
import csv
from pathlib import Path
import re

from ml3 import CsvFile, CsvLine, CsvSchema, CsvSplit, ProgressTrace
 
# ---------------------------------------------------------------------------

class FileSpec:

    def __init__(self, file_name=None, file_type=None):
        self.file_name = file_name
        self.file_type = file_type

# ---------------------------------------------------------------------------

class BasicBuilder:

    # -----------------------------------------------------------------------

    def __init__(self, site, name_key):

        self.name_key = name_key
        self.site = site

    # -----------------------------------------------------------------------

    def run_steps(self, steps):
        self.run_builder_steps(self, steps)

    # -----------------------------------------------------------------------

    @classmethod
    def run_builder_steps(cls, builder, steps):

        for step in steps:

            if step["enabled"]:

                current_step = step.get("key", None)
                key_str = f" - {current_step}" if current_step else ""

                print (f"*** {type(builder).__name__}.{step['step-method'].__name__}{key_str}")
                
                return_value = builder.run_step(step)
                
                if return_value == False:
                    break

    # -----------------------------------------------------------------------

    def run_step(self, step):

        fn = step["step-method"]
        
        if isinstance(fn, str):
            fn = getattr(self, fn)
            
        return fn(self, params=step)

    # -----------------------------------------------------------------------

    def get_file_name_and_type(self, spec_str, expected_file_type):

        if isinstance(spec_str, Path):
            return spec_str, expected_file_type

        i_colon = spec_str.find(":")
        if i_colon != -1:
            file_name = spec_str[:i_colon].strip()
            file_type_2 = spec_str[i_colon + 1:].strip()
            if expected_file_type:
                assert expected_file_type == file_type_2
            file_type = file_type_2
        else:
            file_name = spec_str.strip()
            file_type = None
            assert not expected_file_type

        return file_name, file_type

    # -----------------------------------------------------------------------

    def get_file_spec(self, spec_str, file_type=None):

        if isinstance(spec_str, Path):
            return FileSpec(file_name=spec_str, file_type=file_type)

        file_name, file_type = self.get_file_name_and_type(
            spec_str=spec_str,
            expected_file_type=file_type
        )

        base_directory = Path(self.site.data_directory)
        name_key_parts = [self.name_key]
        if isinstance(file_name, str) and (not file_name.startswith("=")) and "/" in file_name:
            file_name_slashed_parts = file_name.split("/")
            base_directory = base_directory.joinpath("/".join(file_name_slashed_parts[:-1])).resolve()
            file_name = file_name_slashed_parts[-1]
            name_key_parts = []

        def standardized_name(file_name, file_extension):
            
            if file_name.startswith("="):
                return ".".join([
                    part
                    for part in [file_name[1:], file_extension]
                    if part
                ])
            
            name_key_parts_cur = name_key_parts
            if file_name.startswith("#"):
                file_name = file_name[1:]
                name_key_parts_cur = []

            if file_name == "-":
                file_name = None
            
            return ".".join([
                part
                for part in name_key_parts_cur + [file_name, file_extension]
                if part
            ])

        if file_name.startswith("="):
            return FileSpec(file_name=str(Path(file_name[1:]).expanduser()), file_type=file_type)

        file_extension = {
            "model": "model.pt",
            "eval": "eval.csv",
        }.get(file_type, file_type)
        
        return FileSpec(
            file_name=base_directory.joinpath(
                standardized_name(file_name=file_name, file_extension=file_extension
            )),
            file_type=file_type
        )

    # -----------------------------------------------------------------------

    def get_path(self, spec_str, file_type=None):
        return Path(self.get_file_spec(spec_str=spec_str, file_type=file_type).file_name)

    # -----------------------------------------------------------------------

    def get_lines(self, input_spec_str, file_type=None):
        input_spec = self.get_file_spec(input_spec_str, file_type=file_type)
        return list(CsvFile.get_lines_from_csv(input_spec.file_name,
            csv_schema=None,
            create_schema_from_header_line=True
        ))

    # -----------------------------------------------------------------------

    def pause(self, params):
        print(params["message"])
        while True:
            response = input("Continue? [y/n]")
            if response.lower() in ["y", "yes"]:
                return
            if response.lower() in ["n", "no"]:
                return False

    # -----------------------------------------------------------------------

    def manual_step(self, params):
        return

    # -----------------------------------------------------------------------

    def train_test_split_csv(self, params):

        input_spec = self.get_file_spec(params["input"], "csv")
        test_fraction = params["test-fraction"]
        split_key_columns = params["split-key-columns"]
        do_lines_shuffle = params.get("do-lines-shuffle", True)
        assert isinstance(split_key_columns, list)
        train_output, test_output = params["outputs"]
        train_output_spec = self.get_file_spec(train_output, "csv")
        test_output_spec = self.get_file_spec(test_output, "csv")
        assert str(input_spec.file_name) != str(train_output_spec.file_name)
        assert str(train_output_spec.file_name) != str(test_output_spec.file_name)

        input_lines = list(CsvFile.get_lines_from_csv(input_spec.file_name,
            csv_schema=None, create_schema_from_header_line=True
        ))
        if all( # split key columns all empty?
            not "".join(input_line[column] for column in split_key_columns)
            for input_line in input_lines
        ):
            ensured_key = split_key_columns[0]
            for i, input_line in enumerate(input_lines):
                input_line[ensured_key] = i
        
        schema = input_lines[0].schema
        
        split = CsvSplit(key_columns=split_key_columns)
        train_lines, test_lines = split.split_lines(
            input_lines,
            test_fraction=test_fraction,
            do_lines_shuffle=do_lines_shuffle,
        )

        CsvFile.save_to_csv(train_output_spec.file_name, train_lines, schema)
        CsvFile.save_to_csv(test_output_spec.file_name, test_lines, schema)
        return

    # -----------------------------------------------------------------------

    def train_test_split_csvs(self, params):

        test_fraction = params["test-fraction"]
        split_key_column = params["split-key-column"]
        must_match_columns = params["must-match-columns"]

        input_specs = [
            self.get_file_spec(spec_str, "csv")
            for spec_str in params["inputs"]
        ]
        output_specs = [
            (self.get_file_spec(train_spec_str, "csv"), self.get_file_spec(test_spec_str, "csv"))
            for train_spec_str, test_spec_str in params["outputs"]
        ]
        tasks = [
            (input_spec, train_spec, test_spec)
            for input_spec, (train_spec, test_spec) in zip(input_specs, output_specs)
        ]


        input_spec, train_spec, test_spec = tasks[0]
        assert str(input_spec.file_name) != str(train_spec.file_name)
        assert str(train_spec.file_name) != str(test_spec.file_name)

        input_lines = list(CsvFile.get_lines_from_csv(input_spec.file_name,
            csv_schema=None, create_schema_from_header_line=True
        ))
        base_lines_by_key = { line[split_key_column]: line for line in input_lines }
        schema = input_lines[0].schema
        
        split = CsvSplit(key_columns=[split_key_column])
        train_lines, test_lines = split.split_lines(input_lines, test_fraction=test_fraction)
        CsvFile.save_to_csv(train_spec.file_name, train_lines, schema)
        CsvFile.save_to_csv(test_spec.file_name, test_lines, schema)
        train_keys = set([line[split_key_column] for line in train_lines])

        for input_spec, train_spec, test_spec in tasks[1:]:
            input_lines = list(CsvFile.get_lines_from_csv(input_spec.file_name,
                csv_schema=None, create_schema_from_header_line=True
            ))
            assert all(line.schema.columns == schema.columns for line in input_lines)
            for line in input_lines:
                base_line = base_lines_by_key[line[split_key_column]]
                assert all(base_line[column] == line[column] for column in must_match_columns)

            train_lines = [line for line in input_lines if line[split_key_column] in train_keys]
            test_lines = [line for line in input_lines if line[split_key_column] not in train_keys]
            CsvFile.save_to_csv(train_spec.file_name, train_lines, schema)
            CsvFile.save_to_csv(test_spec.file_name, test_lines, schema)

        return

    # -----------------------------------------------------------------------

    def check_csv(self, params):

        input_spec = self.get_file_spec(params["input"])
        assert input_spec.file_name.exists(), f"Expecting file to exist: {input_spec.file_name}"
        input_lines = CsvFile.get_lines_from_csv(input_spec.file_name, csv_schema=None, create_schema_from_header_line=True)
        assert len(input_lines) >= 1, f"Expecting file to be not empty: {input_spec.file_name}"
        input_schema = input_lines[0].schema

        expected_columns = params.get("expected-columns")
        if expected_columns is not None:
            for column in expected_columns:
                assert column in input_schema.columns, \
                    f"Expecting column `{column}` in {input_spec.file_name} (found columns: {input_schema.columns})"

        input_lines_assert_fn = params.get("input-lines-assert-fn")
        if input_lines_assert_fn is not None:
            assert input_lines_assert_fn(input_lines)

        return

    # -----------------------------------------------------------------------

    def normalize_csv(self, params):

        input_spec = self.get_file_spec(params["input"], "csv")
        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(input_spec.file_name) != str(output_spec.file_name)

        input_encoding = params.get("input-encoding", "utf-8")
        output_encoding = params.get("output-encoding", "utf-8")

        with open(input_spec.file_name, encoding=input_encoding, errors='replace') as input_file:
            
            reader = csv.DictReader(input_file)
            
            with open(output_spec.file_name, 'w', encoding=output_encoding) as output_file:

                writer = None

                for row in reader:
                    
                    if writer is None:
                        normalized_header = [unidecode(v) for v in reader.fieldnames]
                        writer = csv.DictWriter(output_file, fieldnames=normalized_header)
                        writer.writeheader()

                    normalized_row = {unidecode(k) : unidecode(v) for k, v in row.items()}
                    writer.writerow(normalized_row)

        return

    # -----------------------------------------------------------------------

    def make_new_columns_in_csv(self, params):

        input_spec = self.get_file_spec(params["input"], "csv")
        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(input_spec.file_name) != str(output_spec.file_name)

        new_columns = params["new-columns"]
        new_columns_fn = params["new-columns-fn"]

        input_lines = list(CsvFile.get_lines_from_csv(input_spec.file_name,
            csv_schema=None, create_schema_from_header_line=True
        ))
        input_schema = input_lines[0].schema
        output_schema = CsvSchema(
            columns=input_schema.columns + [name for name in new_columns if name not in input_schema.columns]
        )

        output_lines = []
        for i, input_line in enumerate(input_lines):
            output_line = CsvLine(schema=output_schema, values=input_line.get_values_as_dict())
            for column, value in new_columns_fn(self, i, input_line, params):
                output_line[column] = value
            output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def merge_columns_from_csvs(self, params):
        input_lines = self.get_lines(params["inputs"]["main"], "csv")
        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(self.get_file_spec(params["inputs"]["main"]).file_name) != str(output_spec.file_name)

        key_columns = params["key-columns"]
        line_key_fn = lambda line: "|".join(line[column] for column in key_columns)

        addeds_specs = params["inputs"]["added"]
        all_additional_columns = []
        addeds = []
        for item in addeds_specs:
            all_additional_columns.extend(item["columns"])
            item_lines_by_key = {
                line_key_fn(line): line
                for line in self.get_lines(item["input"])
            }
            addeds.append((item["columns"], item_lines_by_key))

        output_schema = CsvSchema.from_lines(input_lines, additional_columns=all_additional_columns)
        output_lines = []
        for input_line in input_lines:
            output_line = CsvLine(schema=output_schema, values=input_line.get_values_as_dict())
            for added_columns, added_lines in addeds:
                matching_line = added_lines[line_key_fn(input_line)]
                output_line.add_values({
                    column: matching_line[column]
                    for column in added_columns
                })
            output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def combine_csvs(self, params):

        output_schema = CsvSchema(columns=params["output-columns"])
        output_spec = self.get_file_spec(params["output"], "csv")

        output_lines = []

        for input_spec_string, make_output_line_fn in params["inputs"]:
            
            input_spec = self.get_file_spec(input_spec_string)
            input_lines = list(CsvFile.get_lines_from_csv(input_spec.file_name,
                csv_schema=None, create_schema_from_header_line=True
            ))

            for input_line in input_lines:
                output_line = CsvLine(schema=output_schema)
                if "id" in output_schema.columns:
                    output_line["id"] = str(len(output_lines))
                for column, value in make_output_line_fn(input_line):
                    output_line[column] = value

                output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def concat_csvs(self, params):

        input_specs = params["inputs"]
        assert isinstance(input_specs, list)

        schema = None
        concatenated_lines = []
        for input_spec in input_specs:
            input_lines = self.get_lines(input_spec)
            input_lines_schema = input_lines[0].schema
            if schema is not None:
                assert input_lines_schema.columns == schema.columns
            else:
                schema = input_lines_schema
            concatenated_lines += input_lines

        output_spec = self.get_file_spec(params["output"], "csv")
        CsvFile.save_to_csv(output_spec.file_name, concatenated_lines, schema)
        return

    # -----------------------------------------------------------------------

    def filter_csv(self, params):

        input_lines = self.get_lines(params["input"])
        schema = input_lines[0].schema

        output_spec = self.get_file_spec(params["output"], "csv")

        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        filter_fn = params["filter-fn"]

        output_lines = [
            line
            for line in input_lines
            if filter_fn(line)
        ]

        CsvFile.save_to_csv(output_spec.file_name, output_lines, schema)
        return

    # -----------------------------------------------------------------------

    def shuffle_csv(self, params):

        input_lines = self.get_lines(params["input"])
        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        schema = input_lines[0].schema

        output_lines = [line for line in input_lines]
        np.random.shuffle(output_lines)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, schema)
        return

    # -----------------------------------------------------------------------

    def sample_csv(self, params):

        input_spec = self.get_file_spec(params["input"], "csv")
        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(input_spec.file_name) != str(output_spec.file_name)

        choose_from_non_empty_only_column = params.get("choose-from-non-empty-only-column", None)

        input_lines = list(CsvFile.get_lines_from_csv(input_spec.file_name,
            csv_schema=None, create_schema_from_header_line=True
        ))
        schema = input_lines[0].schema

        sample_count = int(params["sample-count"])

        candidate_lines = [
            line
            for line in input_lines
            if line[choose_from_non_empty_only_column]
        ] if choose_from_non_empty_only_column else input_lines

        indices = np.random.choice(list(range(0, len(candidate_lines))), sample_count)

        output_lines = [candidate_lines[k] for k in indices]

        CsvFile.save_to_csv(output_spec.file_name, output_lines, schema)
        return

    # -----------------------------------------------------------------------

    def copy_range_from_csv(self, params):

        input_lines = self.get_lines(params["input"])
        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        schema = input_lines[0].schema

        s, l = params["range"]

        output_lines = [line for line in input_lines][s:l]

        CsvFile.save_to_csv(output_spec.file_name, output_lines, schema)
        return

    # -----------------------------------------------------------------------

    def add_id_column(self, params):

        input_lines = self.get_lines(params["input"], schema=None, create_schema_from_header_line=True)
        input_schema = input_lines[0].schema
        
        output_spec = self.get_file_spec(params["output"])
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        id_column = params["id-column"]

        output_schema = CsvSchema(columns=[params["id-column"]] + input_schema.columns)
        output_lines = []
        for i, input_line in enumerate(input_lines):
            output_line = CsvLine(schema=output_schema, values=input_line.get_values_as_dict())
            output_line[id_column] = i
            output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def apply_overrides_to_column(self, params):

        input_column_to_override, input_spec_str = params["inputs"]["input"]
        input_lines = self.get_lines(input_spec_str, schema=None, create_schema_from_header_line=True)
        input_schema = input_lines[0].schema

        override_source_column, override_new_column, override_spec_str = params["inputs"]["overrides"]
        override_lines = self.get_lines(override_spec_str, schema=None, create_schema_from_header_line=True)
        overrides = {
            line[override_source_column]: line[override_new_column]
            for line in override_lines
        }

        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(self.get_file_spec(input_spec_str).file_name) != str(output_spec.file_name)

        output_schema = input_schema
        output_lines = []
        for input_line in input_lines:
            output_line = CsvLine(schema=output_schema, values=input_line.get_values_as_dict())
            override = overrides.get(input_line[input_column_to_override])
            if override:
                output_line[input_column_to_override] = override
            output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def apply_lambda(self, params):

        input_lines = self.get_lines(params["input"])

        output_schema = CsvSchema.from_lines(input_lines)
        output_spec = self.get_file_spec(params["output"], "csv")
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        target_columns = sorted(params["target-columns"])
        assert isinstance(target_columns, list)
        assert all(column in output_schema.columns for column in target_columns)
        fn = params["fn"]
        assert callable(fn)

        output_lines = []
        for input_line in input_lines:
            output_line = CsvLine.from_line(source_line=input_line, schema=output_schema)
            lambda_outputs = fn(input_line)
            if lambda_outputs is not None:
                assert isinstance(lambda_outputs, dict)
                assert target_columns == sorted(lambda_outputs.keys())
                for column, value in lambda_outputs.items():
                    output_line[column] = value
                output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    numbering_re = re.compile(r"^\d+\.\s")

    @classmethod
    def split_lines_and_prepare_dsb_examples(cls, examples_raw):
        examples = []
        for line in examples_raw.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith("[[") and line_stripped.endswith("]]"):
                line_sanitized = cls.numbering_re.sub("", line_stripped[2:-2])
                examples.append(line_sanitized.strip())
        return examples

    # -----------------------------------------------------------------------

    def split_column_value_into_multiple_lines(self, params):

        input_lines = self.get_lines(params["input"])
        output_spec = self.get_file_spec(params["output"], file_type="csv")
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        input_column = params["input-column"]
        output_column = params["output-column"]
        split_fn = params["split-fn"]

        input_schema = input_lines[0].schema
        output_schema = CsvSchema(
            columns=
                [
                    column
                    for column in input_schema.columns
                    if column not in [input_column, output_column]
                ] +
                [output_column]
        )

        progress_trace = ProgressTrace(trace_every_seconds=2, check_every_ticks=2) if params.get("progress-trace") else None

        output_lines = []
        for input_line in input_lines:

            output_values = split_fn(input_line[input_column])
            assert isinstance(output_values, list)

            for output_value in output_values:

                output_line = CsvLine(schema=output_schema)
                
                for column in output_schema.columns:
                    if column != output_column:
                        output_line[column] = input_line[column]
                
                output_line[output_column] = output_value
                
                output_lines.append(output_line)

            if progress_trace is not None:
                progress_trace.tick()
                
        if progress_trace is not None:
            progress_trace.done()

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def dedupe_column_value(self, params):

        input_lines = self.get_lines(params["input"])
        output_spec = self.get_file_spec(params["output"], file_type="csv")
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        target_column = params["target-column"]

        output_schema = input_lines[0].schema

        output_lines = []
        seen_values = set()
        for input_line in input_lines:

            target_column_value = input_line[target_column]
            if target_column_value not in seen_values:
                seen_values.add(target_column_value)
                output_lines.append(input_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def lookup_column_value(self, params):

        main_input_lines = self.get_lines(params["inputs"]["main"])
        source_input_lines = self.get_lines(params["inputs"]["lookup-source"])
        output_spec = self.get_file_spec(params["output"], file_type="csv")
        assert str(self.get_file_spec(params["inputs"]["main"]).file_name) != str(output_spec.file_name)
        assert str(self.get_file_spec(params["inputs"]["lookup-source"]).file_name) != str(output_spec.file_name)

        result_column = params["result-column"]
        output_schema = CsvSchema(columns=main_input_lines[0].schema.columns + [result_column])

        output_lines = []
        lookup_column = params["lookup-column"]
        source_key_column = params["source-key-column"]
        source_value_column = params["source-value-column"]
        for input_line in main_input_lines:

            source_key = input_line[lookup_column]
            source_line = next(filter(
                lambda source_line: source_line[source_key_column] == source_key,
                source_input_lines
            ))
            output_line = input_line.copy().with_added_values(values={
                result_column: source_line[source_value_column]
            })
            output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def combine_multiple_lines_into_single_column(self, params):

        input_lines = self.get_lines(params["input"])
        output_spec = self.get_file_spec(params["output"], file_type="csv")
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        input_columns = params["input-columns"]
        assert isinstance(input_columns, list) and all(isinstance(item, str) for item in input_columns)
        output_column = params["output-column"]
        assert isinstance(output_column, str)

        combine_fn = params["combine-fn"]

        output_schema =CsvSchema(columns=
            [column for column in input_lines[0].schema.columns if column not in input_columns] +
            [output_column]
        )

        output_lines = []

        for _key, group_lines_iter in itertools.groupby(input_lines, lambda line: line[params["group-key"]]):

            group_lines = list(group_lines_iter)

            output_line = CsvLine(schema=output_schema, values={
                column: group_lines[0][column]
                for column in output_schema.columns
                if column != output_column
            })
            output_line[output_column] = combine_fn(group_lines)
            output_lines.append(output_line)

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return

    # -----------------------------------------------------------------------

    def add_columns_from_csv(self, params):
        input_lines = self.get_lines(params["input"])
        input_2_lines = self.get_lines(params["input-2"])
        output_spec = self.get_file_spec(params["output"], file_type="csv")
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)
        assert str(self.get_file_spec(params["input-2"]).file_name) != str(output_spec.file_name)

        key_columns = params["key-columns"]
        line_key = lambda line: "|".join(line[column] for column in key_columns)

        add_columns = params["add-columns"]
        input_columns = input_lines[0].schema.columns
        output_schema = CsvSchema(columns=input_columns +
            [name for name in add_columns if name not in input_columns]
        )
        output_lines = [
            CsvLine(schema=output_schema, values=input_line.get_values_as_dict())
            for input_line in input_lines
        ]
        output_lines_by_key = {line_key(output_line): output_line for output_line in output_lines }

        c_merged = 0
        for input_2_line in input_2_lines:
            output_line = output_lines_by_key.get(line_key(input_2_line))
            if output_line:
                merged_something = False
                for column in add_columns:
                    if str(input_2_line[column]):
                        merged_something = True
                    output_line[column] = input_2_line[column]
                c_merged += int(merged_something)
        assert c_merged >= 2, "`input-2` file either doesn't have enough contents to merge on, or the keys don't match"

        CsvFile.save_to_csv(output_spec.file_name, output_lines, output_schema)
        return
