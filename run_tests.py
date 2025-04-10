"""
Runs tests for agent.py. Usage:

```bash
python3 run_tests.py \
    --concurrency N \
    --num-samples M \
    --tests TESTS_FILE \
    --log LOG_FILE \
```

N is the number of tests to run concurrently.
M is the number of attempts to make for each test case.
LOG_FILE is the path to a file where all interactions are logged.
TESTS_FILE is the path to a YAML file with the following format:

```yaml
- user_messages:
    - What is on my schedule for June 1?
  expected_result:
```

Thus each test case in the YAML file is a list of user messages to send to
the agent and the list of flight IDs that should be booked.

To run each test, the script starts agent.py as a subprocess, sends it the
user messages as input and then checks that the last line of output from
the agent is the list of flight IDs.

We log all interactions to LOG_FILE
"""

import asyncio
from dotenv import load_dotenv
import yaml
import argparse
from pathlib import Path
import os

from typing import List, Tuple
import tqdm
from utils import connect_model

load_dotenv(override=True)

async def run_test(
    user_messages: List[str],
    expected_result: List[str],
    model_name: str
) -> Tuple[bool, str]:

    model_path = "llama.py"
    if model_name == '4o_mini':
        model_path = "4o_mini.py"

    # Start the agent process
    process = await asyncio.create_subprocess_exec(
        "python",
        model_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await process.communicate(
            "\n".join(user_messages).encode("utf-8")
        )
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")

        model = connect_model(
            api_key=os.environ["API_KEY"],
            base_url=os.environ["API_BASE_URL"],
            model="gpt-4o-mini",
        )

        comparison_string = f"""
I need you to compare two schedules, an expected schedule, and a test schedule. Compare both schedules and determine if they are roughly the same. The most important details are the time stamps, but the descriptions are not as important, as long as they convey the same message. Only return "True" or "False".
Truth Schedule: {expected_result}

Test Schedule: {stdout} """

        response = model.invoke(comparison_string)
        content_str = response.content

        success = "true" in content_str.lower()

        log = f"Expected: {expected_result}\n"
        log += f"Full stdout: {stdout}\n"
        if stderr:
            log += f"Stderr: {stderr}\n"

        return success, log

    except Exception as e:
        return False, f"Error running test: {str(e)}\n"

    finally:
        try:
            process.terminate()
            await process.wait()
        except:
            pass


async def main_with_args(
    tests_file: Path,
    model: str,
    log_file: Path,
    concurrency: int,
    num_samples: int,
):
    # Load test cases from YAML file
    with tests_file.open("r") as f:
        test_cases = yaml.safe_load(f)

    # Create semaphore to limit concurrent tests
    sem = asyncio.Semaphore(concurrency)

    # Run tests concurrently with semaphore
    async def run_test_with_sem(test_case, index):
        async with sem:
            success, log = await run_test(
                test_case["user_messages"], test_case["expected_result"], model
            )
            return success, log, index

    # Create all test tasks
    tasks = [
        run_test_with_sem(test_case, i)
        for i, test_case in enumerate(test_cases * num_samples)
    ]

    # Run all tests with progress bar
    with tqdm.tqdm(total=len(tasks), desc="Running tests") as pbar, log_file.open(
        "w"
    ) as log_f:
        results = []
        for coro in asyncio.as_completed(tasks):
            success, log, index = await coro
            results.append((success, log))
            log_f.write(f"\nTest {index + 1}\n")
            log_f.write(f"Success: {success}\n")
            log_f.write(log)
            log_f.write("-" * 80 + "\n")
            log_f.flush()
            pbar.update(1)

    # Print summary
    num_passed = sum(1 for success, _ in results if success)
    print(f"Passed {num_passed} out of {len(results)} tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests",
        type=Path,
        default="tests.yaml",
        help="YAML file containing test cases",
    )
    parser.add_argument(
        "--model", type=str, default="llama", help="Model to run. Default is llama", choices=["llama", "4o_mini"]
    )
    parser.add_argument(
        "--log", type=Path, default="test_log.txt", help="File to write test results to"
    )
    parser.add_argument(
        "--concurrency", type=int, default=2, help="Number of tests to run concurrently"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of attempts to make for each test case",
    )
    args = parser.parse_args()

    asyncio.run(
        main_with_args(args.tests, args.model, args.log, args.concurrency, args.num_samples)
    )


if __name__ == "__main__":
    main()
