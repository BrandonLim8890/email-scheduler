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
import yaml
import argparse
from pathlib import Path

from typing import List, Tuple
import re
import tqdm
import re

def extract_schedule_section(text):
    """
    Extracts the schedule section from text.
    
    Returns:
        The schedule section or None if no schedule section is found.
        For empty schedules, returns an empty string.
    """
    # Check for empty schedule patterns
    if re.search(r'Schedule:.*\s*Empty\.?', text, re.IGNORECASE):
        return ""
    
    # Extract the schedule section
    schedule_pattern = r'Schedule:([^]*?)(?:\n\n|\n[^\s-]|$)'
    match = re.search(schedule_pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return None

def extract_schedule_events(text):
    """
    Extracts schedule events from text.
    
    Args:
        text: The text to extract schedule events from.
    
    Returns:
        List of dictionaries with keys:
        - date: The date of the event (e.g., "06/01")
        - time: The time of the event (may be None)
        - description: The event description
        
        Returns an empty list if the schedule is empty or no events are found.
    """
    schedule_section = extract_schedule_section(text)
    
    if schedule_section is None or schedule_section == "":
        return []
    
    if re.search(r'No events found', schedule_section, re.IGNORECASE):
        return []
    
    # Pattern to match:
    # 1. Date in MM/DD format (allows single digit)
    # 2. Optional time or time range
    # 3. Event description
    pattern = r'\s*-\s*(\d{1,2}/\d{1,2})(?:\s+(\d{1,2}:\d{2}(?:\s*[APap][Mm])?(?:\s*-\s*\d{1,2}:\d{2}(?:\s*[APap][Mm])?)?))?:?\s*(.*)'
    
    events = []
    for match in re.finditer(pattern, schedule_section):
        date = match.group(1)
        time = match.group(2) if match.group(2) else None
        description = match.group(3)
        
        events.append({
            'date': date,
            'time': time,
            'description': description
        })
    
    return events

async def run_test(
    user_messages: List[str],
    expected_result: List[str],
) -> Tuple[bool, str]:
    # Start the agent process
    process = await asyncio.create_subprocess_exec(
        "python",
        "queryv3.py",
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

        extracted_events = extract_schedule_events(stdout)

        found_events = []
        missing_events = []

        for expected_event in expected_result:
            expected_time = expected_event.get("time")
            expected_keywords = expected_event.get("keywords", [])
            expected_date = expected_event.get("date")

            event_found = False
            
            for event in extracted_events:
                date_match = True
                if expected_date:
                     # Normalize date format (remove leading zeros)
                    norm_expected_date = expected_date.replace('0', '', 1) if expected_date.startswith('0') else expected_date
                    norm_event_date = event['date'].replace('0', '', 1) if event['date'].startswith('0') else event['date']
                    date_match = norm_expected_date == norm_event_date

                time_match = True
                if expected_time and event['time']:
                    # Normalize time format (remove spaces and case-insensitive compare)
                    norm_expected_time = expected_time.replace(' ', '').lower()
                    norm_event_time = event['time'].replace(' ', '').lower() if event['time'] else ''
                    time_match = norm_expected_time == norm_event_time

                keyword_match = False
                for keyword in expected_keywords:
                    if keyword.lower() in event['description'].lower():
                        keyword_match = True
                        break

                if date_match and time_match and keyword_match:
                    event_found = True
                    found_events.append(expected_event)

            if not event_found:
                reason = []
                if expected_date and not date_match:
                    reason.append("date not found")
                if expected_time and not time_match:
                    reason.append("time not found")
                if not keyword_match:
                    reason.append("keywords not found")
                
                missing_events.append({
                    "event": expected_event,
                    "reason": ", ".join(reason)
                })

        success = len(missing_events) == 0

        log = f"Messages: {user_messages}\n"
        log += f"Expected events: {expected_result}\n"
        log += f"Found events: {found_events}\n"
        log += f"Missing events: {missing_events}\n"
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
                test_case["user_messages"], test_case["expected_result"]
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
        "--log", type=Path, default="test_log.txt", help="File to write test results to"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Number of tests to run concurrently"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of attempts to make for each test case",
    )
    args = parser.parse_args()

    asyncio.run(
        main_with_args(args.tests, args.log, args.concurrency, args.num_samples)
    )


if __name__ == "__main__":
    main()
