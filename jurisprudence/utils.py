import datetime
import json
import os
import re
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Unpack

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from jurisprudence import console
from jurisprudence.schema import DecisionFull, Jurisdiction, Query, Response


def validate_export_batch(items: list[dict[str, Any]], path: Path) -> None:
    """
    Export a batch of items to a file in JSON Lines format.

    Args:
        items: List of dictionaries to be exported.
        path: File path where the items will be exported.
    Raises:
        AssertionError: If writing to the file fails.

    """
    with open(path, "w") as f:
        for item in items:
            decision = DecisionFull.model_validate(item)
            assert f.write(decision.model_dump_json(indent=None) + "\n")


def split_date_range(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: datetime.timedelta,
) -> Generator[tuple[datetime.datetime, datetime.datetime], None, None]:
    """
    Split a date range into smaller intervals.

    Args:
        start_date: The start datetime of the range.
        end_date: The end datetime of the range.
        interval: The size of each interval.

    Yields:
        A tuple containing the start and end dates of each interval.
    """
    while start_date < end_date:
        current_end = min(start_date + interval - datetime.timedelta(days=1), end_date)
        yield start_date, current_end
        start_date = current_end + datetime.timedelta(days=1)


def _log_retry(retry_state: RetryCallState) -> None:
    """
    Log retry attempts for debugging purposes.

    Args:
        retry_state: The current state of the retry mechanism.
    """
    exception = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    console.log(f"[bold red]Attempt {attempt} failed. Retrying...[/bold red]")
    if exception:
        console.log("[bold yellow]Exception details:[/bold yellow]")
        console.print(type(exception), exception, exception.__traceback__)


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4),
    before_sleep=_log_retry,
)
def fetch_batch(**params: Unpack[Query]) -> Response:
    """
    Fetch a batch of data from the API with retry logic.

    Args:
        **params: Additional parameters to be passed to the API.

    Returns:
        The API response containing the batch data.

    Raises:
        httpx.HTTPError: If the API request fails after all retry attempts.
    """
    res = httpx.get(
        f"{os.environ['JUDILIBRE_API_URL']}/cassation/judilibre/v1.0/export",
        params=params,
        headers={
            "KeyId": os.environ["JUDILIBRE_API_KEY"],
            "accept": "application/json",
        },
    )
    if "x-rate-limit" in res.headers:
        rate_limits = json.loads(res.headers["x-rate-limit"])
        max_remaining_limit = max(
            rate["remaining"] for rate in rate_limits if rate["type"] == "throttle"
        )
        min_window_limit = max(
            min(rate["window"] for rate in rate_limits if rate["type"] == "throttle"), 1
        )
        if max_remaining_limit < 5:
            console.log(
                f"[red]Rate limit almost reached. Waiting for {min_window_limit} seconds.[/red]"
            )
            time.sleep(min_window_limit)
    return res.raise_for_status().json()


def bump_last_export_date(end_date: datetime.datetime) -> None:
    """
    Update the last export date in the settings.py file.

    Args:
        end_date: The new last export date to be written.
    Raises:
        ValueError: If the JURISPRUDENCE_LAST_EXPORT_DATETIME line is not found in the file.
    """
    settings_file = Path(__file__).resolve().parent / "settings.py"
    content = settings_file.read_text()
    new_line = f'JURISPRUDENCE_LAST_EXPORT_DATETIME = "{end_date.strftime("%Y-%m-%d %H:%M:%S")}"'
    # Use regex to find and replace the line
    pattern = r"^JURISPRUDENCE_LAST_EXPORT_DATETIME\s*=.*$"
    new_content, count = re.subn(pattern, new_line, content, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(
            "JURISPRUDENCE_LAST_EXPORT_DATETIME line not found in settings.py"
        )

    # Write the updated content back to the file
    _ = settings_file.write_text(new_content)


@retry(
    retry=retry_if_exception_type(ValueError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4),
    before_sleep=_log_retry,
)
def process_date_range(
    jurisdiction: Jurisdiction,
    start: datetime.datetime,
    end: datetime.datetime,
    batch_size: int,
    sleep: int,
) -> list[dict[str, Any]]:
    """
    Process a date range to fetch and combine batches of data.

    This function handles pagination and recursively splits the date range
    if the total number of results reaches the maximum batch size.

    Args:
        jurisdiction: The jurisdiction to fetch data for.
        start: The start date of the range.
        end: The end date of the range.
        batch_size: The number of items to fetch per batch.
        sleep: The number of seconds to wait between API requests.

    Returns:
        A list of all fetched items within the date range.

    Raises:
        ValueError: If the total number of items doesn't match the sum of batch sizes.
    """
    current_batch: list[dict[str, Any]] = []
    batch: int = 0
    total: int = 0
    while True:
        assert start <= end
        # BUG: datetime in ISO format lead fo data loss and 500 internal server errors
        _start = str(start.date())
        _end = str(end.date())
        console.log(
            f"Fetching {jurisdiction}, from {_start} to {_end}, {batch=}, {batch_size=}",
            end="...",
        )
        response: Response = fetch_batch(
            order="asc",
            resolve_references="true",
            batch=str(batch),
            batch_size=str(batch_size),
            date_start=_start,
            date_end=_end,
            jurisdiction=jurisdiction.lower(),
        )
        total: int = response["total"]
        if total == 0:
            break
        elif total == 10_000:
            # BUG: I have experienced data loss
            # when `total` reaches the batch max size == 10_000
            # so let's reduce it by splitting the time range equally
            pivot = start + (end - start) / 2
            left = process_date_range(jurisdiction, start, pivot, batch_size, sleep)
            right = process_date_range(
                jurisdiction, pivot + datetime.timedelta(days=1), end, batch_size, sleep
            )
            return left + right
        console.log(
            f".. got {len(response['results'])} {jurisdiction} jurisprudences",
        )
        current_batch.extend(response["results"])
        batch += 1
        if not response["next_batch"]:
            break
        time.sleep(sleep)
    if total != len(current_batch):
        raise ValueError(
            f"Total ({total}) != current batch's length ({len(current_batch)})"
        )
    return current_batch


def human_readable_size(size: float, decimal_places: int = 2) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"
