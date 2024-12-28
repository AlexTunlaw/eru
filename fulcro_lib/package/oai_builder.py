from typing import List
import os
import numpy as np
from pathlib import Path
import time
import datetime
import asyncio
from asyncio.exceptions import CancelledError
import aiohttp
import traceback
import shutil

from openai import RateLimitError, APITimeoutError

from ml3 import CsvLine, CsvSchema, CsvFile

from fulcro_llm import (
    FulcroCachingOaiClient,
    FulcroAsyncCachingOaiClient,
    FulcroAsyncCachingLlmClients,
    FulcroOaiFineTuningHelpers
)

from .basic_builder import BasicBuilder

COMPLETION_TO_PROCESS, COMPLETION_IN_PROGRESS, COMPLETION_DONE, COMPLETION_ERROR = 0, 1, 10, -1

# ---------------------------------------------------------------------------

class OaiBuilder(BasicBuilder):

    # -----------------------------------------------------------------------

    def __init__(self, site, name_key):

        super().__init__(site, name_key)
        self.print = print

    # -----------------------------------------------------------------------

    def make_fine_tuning_data(self, params):

        input_lines = self.get_lines(params["input"])
        output_spec = self.get_file_spec(params["output"])
        assert str(self.get_file_spec(params["input"]).file_name) != str(output_spec.file_name)

        system_prompt_fn = params["system-prompt-fn"]
        prompt_template_fn = params["user-prompt-fn"]
        expected_output_fn = params["expected-output-fn"]

        cooked_examples = [
            FulcroOaiFineTuningHelpers.SingleTurnFineTuningExample(
                input=prompt_template_fn(input_line),
                expected_output=expected_output_fn(input_line)
            ).get_serialized_form(system_prompt=system_prompt_fn(input_line))
            for input_line in input_lines
        ]

        FulcroOaiFineTuningHelpers.save_fine_tuning_file(output_spec.file_name, cooked_examples)
        return

    # -----------------------------------------------------------------------

    def upload_fine_tuning_data_to_oai(self, params):

        provider_client = FulcroCachingOaiClient(
            client_type=params["openai"],
            local_cache_dir=self.site.oai_completions_cache_dir
        )

        input_spec = self.get_file_spec(params["input"])
        FulcroOaiFineTuningHelpers.upload_fine_tuning_data_to_oai(input_spec.file_name, provider_client.client)
        return

    # -----------------------------------------------------------------------

    def start_fine_tuning_job(self, params):

        provider_client = FulcroCachingOaiClient(
            client_type=params["openai"],
            local_cache_dir=self.site.oai_completions_cache_dir
        )

        uploaded_train_file_id = params.get("input")
        if uploaded_train_file_id is None:
            inputs_dict = params["inputs"]
            assert isinstance(inputs_dict, dict)
            uploaded_train_file_id = inputs_dict["train"]
            uploaded_validate_file_id = inputs_dict["validate"]
        else:
            uploaded_validate_file_id = None

        batch_size = params.get("batch-size", "auto")

        FulcroOaiFineTuningHelpers.start_fine_tuning_job(
            uploaded_train_file_id=uploaded_train_file_id,
            base_model=params["base-model"],
            n_epochs=params.get("n-epochs"),
            openai_client=provider_client.client,
            batch_size=batch_size,
            uploaded_validate_file_id=uploaded_validate_file_id
        )

        FulcroOaiFineTuningHelpers.print_status_of_fine_tuning_job(
            uploaded_file_id=uploaded_train_file_id,
            openai_client=provider_client.client
        )
        return

    # -----------------------------------------------------------------------

    def print_status_of_fine_tuning_job(self, params):

        FulcroOaiFineTuningHelpers.print_status_of_fine_tuning_job(uploaded_file_id=params["input"])
        return

    # -----------------------------------------------------------------------
    #

    async def oai_completion_fanout_task(self,
        openai_invoke_async_fn,
        time_before_all, startup_timeout,
        input_lines, output_lines, statuses, errors, all_fanout_status,
        output_column, output_file_name, output_schema,
        persist_every_n, post_persist_fn
    ):

        c_tasks = len(input_lines)

        retries = [5, 30, 60, 120]

        def get_remaining_tasks_to_process(except_i_task=None):
            return [
                k
                for k in range(c_tasks)
                if (statuses[k] in [COMPLETION_TO_PROCESS, COMPLETION_IN_PROGRESS]) and k != except_i_task
            ]

        if startup_timeout is not None:
            await asyncio.sleep(startup_timeout)

        while all_fanout_status != COMPLETION_ERROR:

            assert len(input_lines) == len(output_lines)

            # find next unclaimed task
            i_task = 0
            while i_task < c_tasks and statuses[i_task] != COMPLETION_TO_PROCESS:
                i_task += 1

            if c_tasks <= i_task: # if no unclaimed tasks
                break

            # claim the task
            time_before = time.time()
            self.print(
                f"[{datetime.datetime.now()}] " +
                f"invoking: {i_task}"
            )
            statuses[i_task] = COMPLETION_IN_PROGRESS

            # go at it
            input_line = input_lines[i_task]

            try:
                result = await openai_invoke_async_fn(i_task, input_line)

                output_lines[i_task][output_column] = result if result else "__EMPTY__"

                statuses[i_task] = COMPLETION_DONE
                completion_status = "OK"

            except CancelledError:
                output_lines[i_task][output_column] = "__EMPTY__"
                statuses[i_task] = COMPLETION_DONE
                completion_status = "OK"
                return

            except AssertionError as e:
                print(f"Assert")
                print(f"  {type(e)}")
                print(f"  {e}")
                print(traceback.format_exc())

                statuses[i_task] = COMPLETION_ERROR
                all_fanout_status[0] = COMPLETION_ERROR
                break

            except BaseException as e:
                print(f"Exception (low-level):")
                print(f"  {type(e)}")
                print(f"  {e}")
                print(traceback.format_exc())

                errors_list = errors[i_task]
                i_retry = len(errors_list)
                errors_list.append(str(type(e)))

                if i_retry < len(retries):
                    retry_timeout = retries[i_retry]
                    self.print(f"waiting ({retry_timeout} sec) & retrying")
                    await asyncio.sleep(retry_timeout)
                    statuses[i_task] = COMPLETION_TO_PROCESS

                else:
                    self.print(f"Failed task {i_task} - too many retries")
                    statuses[i_task] = COMPLETION_ERROR

                completion_status = str(type(e))

            # logging
            time_after = time.time()

            remaining_tasks_to_process = get_remaining_tasks_to_process(except_i_task=i_task)
            remaining_tasks_message = ""
            if len(remaining_tasks_to_process) == 0:
                remaining_tasks_message = f"(all done)"
            elif len(remaining_tasks_to_process) <= 10:
                remaining_tasks_message = f"(remaining: {remaining_tasks_to_process})"
            elif len(remaining_tasks_to_process) <= 100:
                remaining_tasks_message = f"(remaining: {len(remaining_tasks_to_process)})"

            self.print(
                f"[{datetime.datetime.now()}] " +
                f"                   task {i_task}: {completion_status} " +
                f"({time_after - time_before:.1f} sec | {(time_after - time_before_all)/60:.1f} min) " +
                f"{remaining_tasks_message}"
            )

            # periodic save
            if (
                output_file_name is not None and
                (persist_every_n is None or ((i_task % persist_every_n) == 0))
            ):
                assert len(input_lines) == len(output_lines)
                if persist_every_n is not None:
                    self.print(f"**** saving current output: {output_file_name}", end='')
                tmp_file_name = str(output_file_name) + ".tmp"
                CsvFile.save_to_csv(tmp_file_name, output_lines, output_schema)
                shutil.copyfile(tmp_file_name, output_file_name)
                os.remove(tmp_file_name)
                if persist_every_n is not None:
                    self.print(" - done")
                post_persist_fn()

            continue # main loop

        return

    # -----------------------------------------------------------------------

    async def get_completions(self,
        model,
        messages,
        repeat_k,
        repeat_temperatures: List[float],
        max_tokens,
        force_key,
        timeout,
        provider_client
    ):


        if all(repeat_temperatures[0] == v for v in repeat_temperatures):
            create_api_calls = 1
            create_api_param_n = repeat_k
        else:
            create_api_calls = repeat_k
            create_api_param_n = 1

        c_timeouts = 0
        rate_limit_base_timeout_seconds, rate_limit_timeout_backoff_rate = 10, 1.5
        completions = []
        for k in range(create_api_calls):
            while True:
                try:
                    completions += await provider_client.create_completions(
                        model=model,
                        messages=messages,
                        temperature=repeat_temperatures[k],
                        n=create_api_param_n,
                        max_tokens=max_tokens,
                        force_key=force_key,
                        timeout=timeout
                    )
                    break

                except RateLimitError as e:
                    sleep_seconds = int(rate_limit_base_timeout_seconds) + np.random.randint(5, 15)
                    self.print(f"Rate limit timeout; sleep({sleep_seconds})")
                    await asyncio.sleep(sleep_seconds)
                    rate_limit_base_timeout_seconds *= rate_limit_timeout_backoff_rate

                except APITimeoutError as e:
                    c_timeouts += 1
                    if c_timeouts <= 3:
                        sleep_seconds = np.random.randint(5, 15)
                        self.print(f"API timeout; sleep({sleep_seconds})")
                        await asyncio.sleep(sleep_seconds)
                    else:
                        raise e

                except CancelledError as e:
                    raise e

                except BaseException as e:
                    # print("----")
                    # print(e)
                    # print(messages)
                    raise e

        return completions

    # -----------------------------------------------------------------------

    @staticmethod
    def resolve_template(template, replacements):
        resolved = template
        for placeholder, value in replacements:
            assert placeholder in resolved, f"expecting '{placeholder}' in the template"
            resolved = resolved.replace(placeholder, value)
        return resolved

    # -----------------------------------------------------------------------

    def get_oai_fanout_repeat_k_and_temperatures(self, params):

        repeat_k = params.get("repeat-k", 1)
        repeat_temperatures = params.get("repeat-temperatures")
        temperature = params.get("temperature")
        if repeat_k == 1:
            if temperature is not None:
                assert repeat_temperatures is None
                repeat_temperatures = [temperature]
            else:
                if repeat_temperatures is None:
                    repeat_temperatures = [1.0]
        else:
            if repeat_temperatures is None:
                repeat_temperatures = [
                    1.0 if temperature is None else temperature
                ] * repeat_k
            else:
                assert temperature is None

        assert len(repeat_temperatures) == repeat_k
        return repeat_k, repeat_temperatures

    # -----------------------------------------------------------------------

    def get_input_lines(self, params):

        input_value = params["input"]
        if (
            isinstance(input_value, list) and
            all(isinstance(item, CsvLine) for item in input_value)
        ):
            input_lines = input_value
        elif (
            isinstance(input_value, list) and
            all(isinstance(item, dict) for item in input_value)
        ):
            assert all(input_value[0].keys() == item.keys() for item in input_value)
            schema = CsvSchema(columns=list(input_value[0].keys()))
            input_lines = [
                CsvLine(schema=schema, values=item)
                for item in input_value
            ]
        else:
            input_lines = self.get_lines(input_value, "csv")

        return input_lines

    # -----------------------------------------------------------------------
    # params:
    #   "input": <file spec>
    #   "outputs": {
    #       "csv": <file spec>,
    #       "cache": Optional[str]
    #   }
    #   "output-column": str
    #   "fanout-n": int
    #   "repeat-k": Optional[int, default:1]
    #   "repeat-temperatures": Optional[List[int], default:1.0]
    #   "model": str
    #   "system-prompt": str
    #   "prompt-template": str  | assert "__TASK__" in params["prompt-template"]
    #   "prompt-template-replacements": List[Tuple2[placeholder: str, column_name: str]]
    #   "oai-invoke-async-fn": Optional[async func]
    #

    def run_oai_completion_fanout(self, params):

        if "csv" in params["outputs"]:
            assert params["input"] != params["outputs"]["csv"]

        input_lines = self.get_input_lines(params)
        input_schema = CsvSchema.from_lines(input_lines)

        take_max_lines = params.get("take-max-lines", len(input_lines))
        input_lines = input_lines[:take_max_lines]

        if "csv" in params["outputs"]:
            output_spec = self.get_file_spec(params["outputs"]["csv"], "csv")
            output_file = Path(output_spec.file_name).resolve()
        else:
            output_file = None
            output_lines_receptacle = params["outputs"]["lines"]

        fanout_n = params["fanout-n"]
        repeat_k, repeat_temperatures = self.get_oai_fanout_repeat_k_and_temperatures(params)
        repeat_merge_with = params.get("repeat-merge-with", "\n")

        is_done = params.get("is-done",
            lambda c_completed, output_lines: c_completed == len(output_lines)
        )
        if isinstance(is_done, int) or isinstance(is_done, float):
            is_done_threshold = is_done
            is_done = lambda c_completed, output_lines: c_completed >= is_done_threshold

        completions_timeout = params.get("completions-timeout")
        max_tokens = params.get("max-tokens")
        force_key = params.get("force-key")
        persist_every_n = params.get("persist-every-n", 100)
        post_persist_fn = params.get("post-persist-fn", lambda: None)

        retries_of_main_loop = [5, 30, 60, 120]

        provider_client = (
            FulcroAsyncCachingLlmClients.make_client(
                client_type=params["openai"],
                local_cache_dir=self.site.oai_completions_cache_dir
            )
            if not hasattr(params["openai"], "is_caching_llm_client")
            else params["openai"]
        ) if "openai" in params else None
        make_completion_messages = params.get("make-completion-messages-fn",
            lambda line: [
                { 'role': 'system','content': params["system-prompt"] },
                { 'role': 'user', 'content': self.resolve_template(
                    params["prompt-template"],
                    [
                        (placeholder, line[column])
                        for placeholder, column in params["prompt-template-replacements"]
                    ]
                )},
            ]
        )
        async def invoke_oai(i_line, line):
            messages = make_completion_messages(line)
            if "model" in messages:
                model = messages["model"]
                messages = messages["messages"]
            else:
                model = params["model"]
            completions = await self.get_completions(
                model=model,
                messages=messages,
                repeat_k=repeat_k,
                repeat_temperatures=repeat_temperatures,
                max_tokens=max_tokens,
                force_key=f"{i_line}|{force_key if force_key is not None else '-'}",
                timeout=completions_timeout,
                provider_client=provider_client
            )
            return repeat_merge_with.join([
                (text if text is not None else "__NONE__")
                for text in completions
            ])
        invoke_oai_fn = params.get("oai-invoke-async-fn", invoke_oai)

        output_column = params["output-column"]
        assert output_column not in input_lines[0].schema.columns, f"'{output_column}' is already in: {input_lines[0].schema.columns}"

        output_schema = CsvSchema(columns=input_lines[0].schema.columns + [output_column])

        c_tasks = len(input_lines)
        errors = [[]] * c_tasks

        # note: this is important so if a job fails mid-way despite all the timeouts (e.g., because of service outage),
        # it can be resumed from where it failed
        if output_file is not None and output_file.exists():
            previous_output_lines = list(CsvFile.get_lines_from_csv(
                output_file,
                csv_schema=None, create_schema_from_header_line=True
            ))
            # once the fanout is under way, it will persist output file every N task completions

            if "make-completion-messages-fn" in params:
                assert "cache-key-columns" in params
            if "cache-key-columns" in params:
                assert all(column in input_schema.columns for column in params["cache-key-columns"])
                source_texts_fn = lambda line: \
                    "|".join(str(line[column]) for column in params["cache-key-columns"])
            else:
                source_texts_fn = lambda line: \
                    "|".join(str(line[column]) for _placeholder, column in params["prompt-template-replacements"])

            # we do it this way because a save might have been interrupted
            cache = {
                source_texts_fn(line): line[output_column]
                for line in previous_output_lines
                if line[output_column]
            }
            output_lines = []
            for input_line in input_lines:
                output_line = CsvLine(schema=output_schema, values=input_line.get_values_as_dict())
                output_text = cache.get(source_texts_fn(input_line))
                if output_text:
                    output_line[output_column] = output_text
                output_lines.append(output_line)

        else:
            output_lines = [
                CsvLine(schema=output_schema, values=input_line.get_values_as_dict())
                for input_line in input_lines
            ]
        assert len(input_lines) == len(output_lines)

        statuses = [
            (COMPLETION_DONE if output_line[output_column] else COMPLETION_TO_PROCESS)
            for output_line in output_lines
        ]
        all_fanout_status = [COMPLETION_TO_PROCESS]

        def get_incomplete_task_count():
            return sum(int(statuses[k] in [COMPLETION_TO_PROCESS, COMPLETION_IN_PROGRESS]) for k in range(c_tasks))

        async def oai_fanout(fanout_n, startup_timeout):
            connector = aiohttp.TCPConnector(limit=round(fanout_n * 2 * 1.25)) # 2 connections per server, 1 for post and 1 for get response
            timeout = aiohttp.ClientTimeout(total= 24 * 60 * 60) # note: this timeout is for duration of the session
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                tasks = [
                    self.oai_completion_fanout_task(
                        invoke_oai_fn,
                        time_before_all, startup_timeout,
                        input_lines, output_lines, statuses, errors, all_fanout_status,
                        output_column, output_file, output_schema,
                        persist_every_n, post_persist_fn
                    )
                    for _k in range(min(fanout_n, len(input_lines)))
                ]

                # "soft" await asyncio.gather(*tasks)
                tasks_remaining = [asyncio.ensure_future(task) for task in tasks]
                c_tasks_completed = 0
                while not is_done(c_tasks_completed, output_lines) and tasks_remaining:
                    _tasks_done, tasks_remaining = await asyncio.wait(
                        tasks_remaining,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    c_tasks_completed = len(tasks) - len(tasks_remaining)
                for task in tasks_remaining:
                    task.cancel() # (only hit if "is-done" provided)
                    await task

        i_retry_timeout = -1
        startup_timeout = 0.01
        time_before_all = time.time()
        while True:

            # reset in-progress statuses in case we got a top-level fanout fail
            for k in range(c_tasks):
                if statuses[k] == COMPLETION_IN_PROGRESS:
                    statuses[k] = COMPLETION_TO_PROCESS

            incomplete_task_count = get_incomplete_task_count()
            if is_done(c_completed=len(input_lines) - incomplete_task_count, output_lines=output_lines):
                break

            try:
                previous_incomplete_task_count = incomplete_task_count
                asyncio.run(oai_fanout(fanout_n=fanout_n, startup_timeout=startup_timeout))
                i_retry_timeout = -1

            except CancelledError:
                i_retry_timeout = -1
                break

            except KeyboardInterrupt:
                self.print(f"Keyboard interrupt")
                break

            except BaseException as e:
                incomplete_task_count = get_incomplete_task_count()
                self.print(f"Exception (top-level):")
                self.print(f"  {type(e)}")
                self.print(f"  {e}")
                self.print(traceback.format_exc())
                self.print(f"  remaining batches: {incomplete_task_count}")

                startup_timeout = 1 # sec; always minimal wait

                if previous_incomplete_task_count == incomplete_task_count:

                    self.print(f"  no progress")
                    i_retry_timeout += 1

                    if len(retries_of_main_loop) <= i_retry_timeout:
                        self.print("  DONE with all retries, exiting")
                        break
                    self.print(f"  doing a timeout+retry")
                    startup_timeout += retries_of_main_loop[i_retry_timeout]

                else: # if made some progress
                    i_retry_timeout = -1

                self.print(f"  sleep: {startup_timeout}")

        key_str = params.get("key", "")
        if key_str:
            key_str = f"{key_str}, "
        self.print(f"done ({key_str}in {time.time() - time_before_all:.1f} sec)")

        if output_file is not None:
            assert len(input_lines) == len(output_lines)
            CsvFile.save_to_csv(output_file, output_lines, output_schema)
            self.print(f"**** saved output: {output_file}")
        else:
            assert output_lines_receptacle is not None
            output_lines_receptacle.extend(output_lines)

        return
