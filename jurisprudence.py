import datetime
import enum
import json
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, Unpack

import click
import httpx
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

console = Console()


class Jurisdiction(str, enum.Enum):
    CA = "CA"
    TJ = "TJ"
    CC = "CC"


class Query(TypedDict):
    batch: int
    type: NotRequired[list[str]]
    theme: NotRequired[list[str]]
    chamber: NotRequired[list[str]]
    formation: NotRequired[list[str]]
    jurisdiction: NotRequired[list[str]]
    location: NotRequired[list[str]]
    publication: NotRequired[list[str]]
    solution: NotRequired[list[str]]
    date_start: NotRequired[str]
    date_end: NotRequired[str]
    abridged: NotRequired[bool]
    date_type: NotRequired[str]
    order: NotRequired[str]
    batch_size: NotRequired[int]
    resolve_references: NotRequired[bool]
    withFileOfType: NotRequired[list[str]]
    particularInterest: NotRequired[bool]


class ResponseDict(TypedDict):
    # incomplete
    total: int
    results: list[dict[str, Any]]
    next_batch: str


def export_batch(items: list[dict[str, Any]], path: Path) -> None:
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
            assert f.write(json.dumps(item, sort_keys=True, indent=None) + "\n")


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


def _log_retry(retry_state):
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
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    before_sleep=_log_retry,
)
def fetch_batch(**params: Unpack[Query]) -> ResponseDict:
    """
    Fetch a batch of data from the API with retry logic.

    Args:
        **params: Additional parameters to be passed to the API.

    Returns:
        The API response containing the batch data.

    Raises:
        httpx.HTTPError: If the API request fails after all retry attempts.
    """
    return (
        httpx.get(
            f"{os.environ['JUDILIBRE_API_URL']}/cassation/judilibre/v1.0/export",
            params=params,
            headers={
                "KeyId": os.environ["JUDILIBRE_API_KEY"],
                "accept": "application/json",
            },
        )
        .raise_for_status()
        .json()
    )


def bump_last_export_date(end_date: datetime.datetime) -> None:
    """
    Update the last export date in the .env file.

    Args:
        end_date: The new last export date to be written.
    Raises:
        AssertionError: If writing to the file fails.
    """
    init_file = Path(__file__).resolve().parent / ".env"
    line_to_write = f'export JURISPRUDENCE_LAST_EXPORT_DATETIME="{end_date.strftime("%Y-%m-%d %H:%M:%S")}"\n'
    assert init_file.write_text(line_to_write)


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
        _start = start.isoformat().replace("+00:00", "Z")
        _end = end.isoformat().replace("+00:00", "Z")
        console.log(
            f"Fetching {jurisdiction}, from {_start} to {_end}, {batch=}, {batch_size=}",
            end="...",
        )
        response: ResponseDict = fetch_batch(
            order="asc",
            date_type="creation",
            resolve_references="true",
            batch=str(batch),
            batch_size=str(batch_size),
            date_start=_start,
            date_end=_end,
            jurisdiction=jurisdiction,
        )
        total: int = response["total"]
        if total == 0:
            break
        elif total == 10_000:
            # I have experienced data loss
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


@click.command()
@click.argument(
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
@click.option(
    "--start-date",
    type=click.DateTime(),
    default=os.getenv(
        "JURISPRUDENCE_LAST_EXPORT_DATETIME", str(datetime.date(1800, 1, 1))
    ),
    help="Start date for data retrieval, UTC",
)
@click.option(
    "--end-date",
    type=click.DateTime(),
    default=datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    help="End date for data, UTC",
)
@click.option(
    "-i",
    "--weeks-interval",
    type=int,
    default=26,
    help="Interval for pagination",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for exports, must be lower or equal than 1000",
)
@click.option(
    "-j",
    "--jurisdictions",
    type=click.Choice(Jurisdiction._member_names_ + ["all"]),
    default=["all"],
    multiple=True,
    help="Jurisdictions to export, default: all",
)
@click.option(
    "-s",
    "--sleep",
    type=int,
    default=1,
    help="Sleep time between two consecutive batch API requests",
)
def main(
    output_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    weeks_interval: int,
    batch_size: int,
    jurisdictions: list[Jurisdiction | Literal["all"]],
    sleep: int,
) -> None:
    """
    Export jurisprudence data for specified jurisdictions and date ranges
    from JUDILIBRE and the PISTE API.

    Args:
        output_path: The directory where exported data will be saved.
        start_date: The start date for data retrieval.
        end_date: The end date for data retrieval (excluded).
        weeks_interval: The number of weeks to use as an interval for pagination.
        batch_size: The number of items to fetch per batch.
        jurisdictions: List of jurisdictions to export data for.
        sleep: Sleeping time between two consecutive batch API requests.

    Raises:
        AssertionError: If input parameters are invalid or required environment variables are missing.
    """
    start_date = start_date.replace(tzinfo=datetime.timezone.utc)
    end_date = end_date.replace(tzinfo=datetime.timezone.utc)
    assert start_date < end_date, "Start date must be strictly lower than end date"
    assert weeks_interval > 0, "Weeks interval cannot be less than or equal to 0"
    assert (
        "JUDILIBRE_API_URL" in os.environ
    ), "JUDILIBRE_API_URL must be set, e.g.: https://api.piste.gouv.fr"
    assert batch_size <= 1000, "Batch size must be lower than or equal to 1000"
    assert (
        sleep >= 0
    ), f"You cannot sleep for {sleep} seconds, time travel to the past is not yet supported."
    interval: datetime.timedelta = datetime.timedelta(weeks=weeks_interval)
    if "all" in jurisdictions:
        jurisdictions: list[Jurisdiction] = list(Jurisdiction._member_names_)  # type: ignore

    for jurisdiction in jurisdictions:
        output_path_jurisdiction = output_path / jurisdiction
        os.makedirs(output_path_jurisdiction, exist_ok=True)
        date_ranges = split_date_range(start_date, end_date, interval)
        for start, end in date_ranges:
            time_range_batch = process_date_range(
                jurisdiction, start, end, batch_size, sleep
            )
            if len(time_range_batch) > 0:
                export_batch(
                    time_range_batch,
                    output_path_jurisdiction
                    / f"{start.isoformat()}-{end.isoformat()}.jsonl",
                )
        bump_last_export_date(end_date)


if __name__ == "__main__":
    main()
