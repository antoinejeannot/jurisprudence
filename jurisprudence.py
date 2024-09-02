import datetime
import enum
import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal, TypedDict

import click
import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

console = Console()


class Jurisdiction(str, enum.Enum):
    CA = "ca"
    TJ = "tj"
    CC = "cc"


class ResponseDict(TypedDict):
    total: int
    results: list[dict[str, Any]]
    next_batch: bool


def export_batch(items: list[dict[str, Any]], path: Path) -> None:
    """
    Export a batch of items to a file in JSON Lines format.

    Args:
        items (list[dict[str, Any]]): List of dictionaries to be exported.
        path (Path): File path where the items will be exported.
    """
    with open(path, "w") as f:
        for item in items:
            assert f.write(json.dumps(item, sort_keys=True, indent=None) + "\n")


def split_date_range(
    start_date: datetime.date, end_date: datetime.date, interval: datetime.timedelta
) -> Generator[tuple[str, str], None, None]:
    """
    Split a date range into smaller intervals.

    Args:
        start_date (datetime.date): The start date of the range.
        end_date (datetime.date): The end date of the range.
        interval (datetime.timedelta): The size of each interval.

    Yields:
        tuple[str, str]: A tuple containing the start and end dates of each interval as strings.
    """
    current_start = start_date
    while current_start < end_date:
        current_end = min(
            current_start + interval - datetime.timedelta(days=1), end_date
        )
        yield str(current_start), str(current_end)
        current_start = current_end + datetime.timedelta(days=1)


def find_middle_date(start_date: str, end_date: str) -> datetime.datetime:
    """
    Find the middle date between two given dates.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        datetime.datetime: The middle date between start_date and end_date.

    Raises:
        AssertionError: If start_date is greater or equal to end_date.
    """
    start_date_time = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date_time = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    assert start_date < end_date
    difference = end_date_time - start_date_time
    midpoint = difference / 2
    middle_date = start_date_time + midpoint
    return middle_date


@retry(
    retry=retry_if_exception_type(httpx.RequestError),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    before_sleep=lambda retry_state: console.print(
        f"Attempt {retry_state.attempt_number} failed. Retrying..."
    ),
)
def fetch_batch(url: str, **params: dict[str, Any]) -> ResponseDict:
    """
    Fetch a batch of data from the API with retry logic.

    Args:
        url (str): The API endpoint URL.
        **params: Additional parameters to be passed to the API.

    Returns:
        ResponseDict: The API response containing the batch data.

    Raises:
        httpx.RequestError: If the request fails after all retry attempts.
    """
    return (
        httpx.get(
            url,
            params=params,
            headers={
                "KeyId": os.environ["JUDILIBRE_API_KEY"],
                "accept": "application/json",
            },
        )
        .raise_for_status()
        .json()
    )


def bump_last_export_date(end_date: datetime.date) -> None:
    """
    Update the last export date in the .env file.

    Args:
        end_date (datetime.date): The new last export date to be written.
    """
    init_file = Path(__file__).resolve().parent / ".env"
    line_to_write = f'JURISPRUDENCE_LAST_EXPORT_DATE = "{str(end_date)}"'
    assert init_file.write_text(line_to_write)


def process_date_range(
    url: str,
    jurisdiction: Jurisdiction,
    start: str,
    end: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """
    Process a date range to fetch and combine batches of data.

    This function handles pagination and recursively splits the date range
    if the total number of results reaches the maximum batch size.

    Args:
        url (str): The API endpoint URL.
        jurisdiction (Jurisdiction): The jurisdiction to fetch data for.
        start (str): The start date of the range in 'YYYY-MM-DD' format.
        end (str): The end date of the range in 'YYYY-MM-DD' format.
        batch_size (int): The number of items to fetch per batch.

    Returns:
        list[dict[str, Any]]: A list of all fetched items within the date range.

    Raises:
        ValueError: If the total number of items doesn't match the sum of batch sizes.
        Exception: If batch processing fails after all retry attempts.
    """
    current_batch: list[dict[str, Any]] = []
    batch: int = 0
    total: int = 0
    while True:
        try:
            response: ResponseDict = fetch_batch(
                url,
                order="asc",
                resolve_references="true",
                batch=str(batch),
                batch_size=str(batch_size),
                date_start=start,
                date_end=end,
                jurisdiction=jurisdiction,
            )
        except Exception as e:
            print(f"Failed to process batch {batch} after all retries: {e}")
            raise
        total: int = response["total"]
        if total == 0:
            break
        elif total == 10_000:
            # I have experienced data loss
            # when `total` reaches the batch max size == 10_000
            # so let's reduce it by splitting the time range equally
            pivot = find_middle_date(start, end)
            left = process_date_range(
                url, jurisdiction, start, str(pivot.date()), batch_size
            )
            right = process_date_range(
                url,
                jurisdiction,
                str((pivot + datetime.timedelta(days=1)).date()),
                end,
                batch_size,
            )
            return left + right
        current_batch.extend(response["results"])
        batch += 1
        if not response["next_batch"]:
            break
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
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(datetime.date(1800, 1, 1)),
    help="Start date for data retrieval",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(datetime.datetime.now().date()),
    help="End date for data retrieval, excluded",
)
@click.option(
    "--weeks-interval",
    type=int,
    default=26,
    help="Interval for pagination",
)
@click.option(
    "--batch-size",
    type=int,
    default=1000,
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
def main(
    output_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    weeks_interval: int,
    batch_size: int,
    jurisdictions: list[Jurisdiction | Literal["all"]],
) -> None:
    """
    Export jurisprudence data for specified jurisdictions and date ranges
    from JUDILIBRE and the PISTE API.

    Args:
        output_path (Path): The directory where exported data will be saved.
        start_date (datetime.datetime): The start date for data retrieval.
        end_date (datetime.datetime): The end date for data retrieval (excluded).
        weeks_interval (int): The number of weeks to use as an interval for pagination.
        batch_size (int): The number of items to fetch per batch
        jurisdictions (list[str]): List of jurisdictions to export data for.
    """
    assert start_date < end_date, "Start date must be stricly lower than end date"
    assert weeks_interval > 0, "Weeks interval cannot be less or equal to 0"
    url = f"{os.environ['JUDILIBRE_API_URL']}/cassation/judilibre/v1.0/export"
    interval: datetime.timedelta = datetime.timedelta(weeks=weeks_interval)

    def create_progress_bar(jurisdiction: str):
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Exporting {jurisdiction}", justify="right"),
            "•",
            TextColumn("[bold green]{task.fields[date_range]}", justify="right"),
            "•",
            BarColumn(bar_width=None),
            "•",
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            "•",
            expand=True,
        )

    if "all" in jurisdictions:
        jurisdictions: list[Jurisdiction] = list(Jurisdiction._member_names_)  # type: ignore

    progress_bars = {j: create_progress_bar(j) for j in jurisdictions}
    group = Group(*progress_bars.values())
    live = Live(group)
    with live:
        tasks = {
            j: progress_bars[j].add_task(f"Exporting {j}", start=True, date_range="")
            for j in jurisdictions
        }

        for jurisdiction in jurisdictions:
            output_path_jurisdiction = output_path / jurisdiction
            os.makedirs(output_path_jurisdiction, exist_ok=True)
            date_ranges = list(
                split_date_range(start_date.date(), end_date.date(), interval)
            )
            progress_bars[jurisdiction].update(
                tasks[jurisdiction], total=len(date_ranges)
            )
            for start, end in date_ranges:
                progress_bars[jurisdiction].update(
                    tasks[jurisdiction], date_range=f"{start} - {end}"
                )
                time_range_batch = process_date_range(
                    url, jurisdiction, start, end, batch_size
                )
                if time_range_batch:
                    export_batch(
                        time_range_batch,
                        output_path_jurisdiction / f"{start}-{end}.jsonl",
                    )
                progress_bars[jurisdiction].advance(tasks[jurisdiction])
        bump_last_export_date(end_date.date())


if __name__ == "__main__":
    main()
