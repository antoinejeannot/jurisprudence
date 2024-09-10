import datetime
import enum
import json
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, Unpack

import click
import duckdb
import httpx
import pyarrow as pa
import pyarrow.json as pj
import pyarrow.parquet as pq
import tiktoken
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, field_serializer, field_validator
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

console = Console()


@click.group()
def cli(): ...


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


class ZoneSegment(BaseModel):
    start: int = Field(description="Indice de début du segment.")
    end: int = Field(description="Indice de fin du segment.")

    @field_validator("end", mode="before")
    @classmethod
    def validate_chamber(cls, v: Any) -> int:
        if v is None:
            print(f"Issue, ZoneSegment end is integer: {v}")
            return -1
        if isinstance(v, int):
            return v
        raise ValueError(f"Chamber must be int or None, not {type(v)}")


class Zone(BaseModel):
    motivations: list[ZoneSegment] | None = Field(
        None, description="Segments de la zone 'motivations'."
    )
    moyens: list[ZoneSegment] | None = Field(
        None, description="Segments de la zone 'moyens'."
    )
    dispositif: list[ZoneSegment] | None = Field(
        None, description="Segments de la zone 'dispositifs'."
    )
    annexes: list[ZoneSegment] | None = Field(
        None, description="Segments de la zone 'moyens annexés'."
    )
    expose: list[ZoneSegment] | None = Field(
        None, description="Segments de la zone 'exposé du litige'."
    )
    introduction: list[ZoneSegment] | None = Field(
        None, description="Segments de la zone 'introduction'."
    )


class TextLink(BaseModel):
    id: int | None = Field(None, description="Identifiant du texte.")
    title: str = Field(description="Intitulé du texte.")
    url: str | None = Field(None, description="URL du texte.")


class DecisionLink(BaseModel):
    date: datetime.date | None = Field(None, description="Date de la décision.")
    jurisdiction: str | None = Field(
        None, description="Juridiction ayant rendu la décision."
    )
    description: str | None = Field(None, description="Description de la décision.")
    source: str | None = Field(None, description="Source de la décision.")
    title: str = Field(description="Intitulé de la décision.")
    content: str | None = Field(None, description="Contenu de la décision.")
    url: str | None = Field(None, description="URL de la décision.")
    number: str | None = Field(None, description="Numéro de la décision.")
    ongoing: bool | None = Field(
        None, description="Indique si la décision n'a pas encore été rendue."
    )
    solution: str | None = Field(None, description="Solution de la décision.")
    chamber: str | None = Field(None, description="Chambre ayant rendu la décision.")
    theme: list[str] | None = Field(
        None, description="Liste des thèmes associés à la décision."
    )
    location: str | None = Field(None, description="Siège ayant rendu la décision.")
    id: str | None = Field(None, description="Identifiant de la décision.")
    partial: bool | None = Field(
        None, description="Indique si le contenu de la décision est partiel."
    )

    @field_validator("chamber", mode="before")
    @classmethod
    def validate_chamber(cls, v: Any) -> str | None:
        if v is None or isinstance(v, str):
            return None
        if isinstance(v, int):
            print(f"Issue, chamber is integer: {v}")
            return str(v)
        raise ValueError(f"Chamber must be str, int, or None, not {type(v)}")


class FileLink(BaseModel):
    date: datetime.date = Field(description="Date d'ajout du document.")
    size: str | None = Field(None, description="Taille du fichier.")
    isCommunication: bool = Field(
        description="Indique si le document est un document de communication."
    )
    name: str = Field(description="Intitulé du document associé.")
    id: str = Field(description="Identifiant du document associé.")
    type: str = Field(description="Code correspondant au type de document associé.")
    url: str = Field(description="URL du document associé.")


class DecisionFull(BaseModel):
    summary: str | None = Field(None, description="Sommaire (texte brut).")
    jurisdiction: str = Field(description="Clé de la juridiction.")
    numbers: list[str] = Field(
        description="Tous les numéros de pourvoi de la décision."
    )
    formation: str | None = Field(None, description="Clé de la formation.")
    type: str | None = Field(None, description="Clé du type de décision.")
    decision_date: datetime.date = Field(description="Date de création de la décision.")
    themes: list[str] | None = Field(
        None, description="Liste des matières par ordre de maillons."
    )
    number: str = Field(description="Numéro de pourvoi principal de la décision.")
    solution: str = Field(description="Clé de la solution.")
    ecli: str | None = Field(None, description="Code ECLI de la décision.")
    chamber: str | None = Field(None, description="Clé de la chambre.")
    solution_alt: str | None = Field(
        None, description="Intitulé complet de la solution."
    )
    publication: list[str] = Field(description="Clés du niveau de publication.")
    files: list[FileLink] | None = Field(
        None, description="Liste des fichiers associés à la décision."
    )
    id: str = Field(description="Identifiant de la décision.")
    bulletin: str | None = Field(None, description="Numéro de publication au bulletin.")

    update_datetime: datetime.datetime | None = Field(
        None, description="Date de dernière mise à jour de la décision."
    )
    decision_datetime: datetime.datetime | None = Field(
        None, description="Date de création de la décision."
    )
    zones: Zone | None = Field(
        None,
        description="Zones détectées dans le texte intégral de la décision.",
    )
    forward: DecisionLink | None = Field(None, description="Lien vers une décision.")
    contested: DecisionLink | None = Field(
        None, description="Lien vers une décision contestée."
    )
    update_date: datetime.date | None = Field(
        None, description="Date de dernière mise à jour de la décision."
    )
    nac: str | None = Field(None, description="Code NAC de la décision.")
    rapprochements: list[DecisionLink] | None = Field(
        None, description="Liste des rapprochements de jurisprudence."
    )
    visa: list[TextLink] | None = Field(
        None, description="Liste des textes appliqués par la décision."
    )
    particularInterest: bool | None = Field(
        None,
        description="Indique si la décision présente un intérêt particulier.",
    )
    timeline: list[DecisionLink] | None = Field(
        None, description="Liste des dates clés relatives à la décision."
    )
    to_be_deleted: bool | None = Field(
        None, description="Indique si la décision doit être supprimée."
    )
    text: str | None = Field(
        None, description="Texte intégral et pseudonymisé de la décision."
    )
    partial: bool | None = Field(
        None, description="Indique si le contenu de la décision est partiel."
    )
    text_highlight: str | None = Field(
        None,
        description="Texte intégral avec correspondances de recherche en surbrillance.",
    )
    titlesAndSummaries: str | None = Field(
        None, description="Titres et sommaires définis pour la décision."
    )
    legacy: str | None = Field(
        None,
        description="Propriétés historiques propres à la source de données.",
    )

    @field_validator("titlesAndSummaries", "legacy", mode="before")
    @classmethod
    def validate_titles_and_summaries(cls, v: Any):
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        elif isinstance(v, str):
            return "{}"
        raise ValueError()

    @field_serializer("update_datetime", "decision_datetime", when_used="json")
    def serialize_timestamp(self, timestamp: datetime.datetime | None):
        if timestamp is None:
            return None
        return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


PARQUET_SCHEMA = pa.schema(
    [
        # ZoneSegment fields
        pa.field(
            "zones",
            pa.struct(
                [
                    pa.field(
                        "motivations",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("start", pa.int64()),
                                    pa.field("end", pa.int64()),
                                ]
                            )
                        ),
                    ),
                    pa.field(
                        "moyens",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("start", pa.int64()),
                                    pa.field("end", pa.int64()),
                                ]
                            )
                        ),
                    ),
                    pa.field(
                        "dispositif",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("start", pa.int64()),
                                    pa.field("end", pa.int64()),
                                ]
                            )
                        ),
                    ),
                    pa.field(
                        "annexes",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("start", pa.int64()),
                                    pa.field("end", pa.int64()),
                                ]
                            )
                        ),
                    ),
                    pa.field(
                        "expose",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("start", pa.int64()),
                                    pa.field("end", pa.int64()),
                                ]
                            )
                        ),
                    ),
                    pa.field(
                        "introduction",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("start", pa.int64()),
                                    pa.field("end", pa.int64()),
                                ]
                            )
                        ),
                    ),
                ]
            ),
        ),
        # TextLink fields
        pa.field(
            "visa",
            pa.list_(
                pa.struct(
                    [
                        pa.field("id", pa.int64()),
                        pa.field("title", pa.string()),
                        pa.field("url", pa.string()),
                    ]
                )
            ),
        ),
        # DecisionLink fields
        pa.field(
            "forward",
            pa.struct(
                [
                    pa.field("date", pa.string()),
                    pa.field("jurisdiction", pa.string()),
                    pa.field("description", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("content", pa.string()),
                    pa.field("url", pa.string()),
                    pa.field("number", pa.string()),
                    pa.field("ongoing", pa.bool_()),
                    pa.field("solution", pa.string()),
                    pa.field("chamber", pa.string()),
                    pa.field("theme", pa.list_(pa.string())),
                    pa.field("location", pa.string()),
                    pa.field("id", pa.string()),
                    pa.field("partial", pa.bool_()),
                ]
            ),
        ),
        # FileLink fields
        pa.field(
            "files",
            pa.list_(
                pa.struct(
                    [
                        pa.field("date", pa.string()),
                        pa.field("size", pa.string()),
                        pa.field("isCommunication", pa.bool_()),
                        pa.field("name", pa.string()),
                        pa.field("id", pa.string()),
                        pa.field("type", pa.string()),
                        pa.field("url", pa.string()),
                    ]
                )
            ),
        ),
        # DecisionFull fields
        pa.field("summary", pa.string()),
        pa.field("jurisdiction", pa.string()),
        pa.field("numbers", pa.list_(pa.string())),
        pa.field("formation", pa.string()),
        pa.field("type", pa.string()),
        pa.field("decision_date", pa.string()),
        pa.field("themes", pa.list_(pa.string())),
        pa.field("number", pa.string()),
        pa.field("solution", pa.string()),
        pa.field("ecli", pa.string()),
        pa.field("chamber", pa.string()),
        pa.field("solution_alt", pa.string()),
        pa.field("publication", pa.list_(pa.string())),
        pa.field("id", pa.string()),
        pa.field("bulletin", pa.string()),
        pa.field("update_datetime", pa.timestamp("ms")),
        pa.field("decision_datetime", pa.timestamp("ms")),
        pa.field(
            "contested",
            pa.struct(
                [
                    pa.field("date", pa.string()),
                    pa.field("jurisdiction", pa.string()),
                    pa.field("description", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("content", pa.string()),
                    pa.field("url", pa.string()),
                    pa.field("number", pa.string()),
                    pa.field("ongoing", pa.bool_()),
                    pa.field("solution", pa.string()),
                    pa.field("chamber", pa.string()),
                    pa.field("theme", pa.list_(pa.string())),
                    pa.field("location", pa.string()),
                    pa.field("id", pa.string()),
                    pa.field("partial", pa.bool_()),
                ]
            ),
        ),
        pa.field("update_date", pa.string()),
        pa.field("nac", pa.string()),
        pa.field(
            "rapprochements",
            pa.list_(
                pa.struct(
                    [
                        pa.field("date", pa.string()),
                        pa.field("jurisdiction", pa.string()),
                        pa.field("description", pa.string()),
                        pa.field("source", pa.string()),
                        pa.field("title", pa.string()),
                        pa.field("content", pa.string()),
                        pa.field("url", pa.string()),
                        pa.field("number", pa.string()),
                        pa.field("ongoing", pa.bool_()),
                        pa.field("solution", pa.string()),
                        pa.field("chamber", pa.string()),
                        pa.field("theme", pa.list_(pa.string())),
                        pa.field("location", pa.string()),
                        pa.field("id", pa.string()),
                        pa.field("partial", pa.bool_()),
                    ]
                )
            ),
        ),
        pa.field("particularInterest", pa.bool_()),
        pa.field(
            "timeline",
            pa.list_(
                pa.struct(
                    [
                        pa.field("date", pa.string()),
                        pa.field("jurisdiction", pa.string()),
                        pa.field("description", pa.string()),
                        pa.field("source", pa.string()),
                        pa.field("title", pa.string()),
                        pa.field("content", pa.string()),
                        pa.field("url", pa.string()),
                        pa.field("number", pa.string()),
                        pa.field("ongoing", pa.bool_()),
                        pa.field("solution", pa.string()),
                        pa.field("chamber", pa.string()),
                        pa.field("theme", pa.list_(pa.string())),
                        pa.field("location", pa.string()),
                        pa.field("id", pa.string()),
                        pa.field("partial", pa.bool_()),
                    ]
                )
            ),
        ),
        pa.field("to_be_deleted", pa.bool_()),
        pa.field("text", pa.string()),
        pa.field("partial", pa.bool_()),
        pa.field("text_highlight", pa.string()),
        pa.field("titlesAndSummaries", pa.string()),  # Changed to string
        pa.field("legacy", pa.string()),  # Changed to string
    ]
)


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
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4),
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
    line_to_write = f'export JURISPRUDENCE_LAST_EXPORT_DATETIME={end_date.strftime("%Y-%m-%d %H:%M:%S")}\n'
    assert init_file.write_text(line_to_write)


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
        response: ResponseDict = fetch_batch(
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


def _human_readable_size(size, decimal_places=2):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


@cli.command()
@click.argument(
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=".",
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
@click.option(
    "-s",
    "--sleep",
    type=int,
    default=1,
    help="Sleep time between two consecutive batch API requests",
)
def export(
    output_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    weeks_interval: int,
    batch_size: int,
    jurisdictions: list[Jurisdiction | Literal["all"]],
    sleep: int,
):
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
                validate_export_batch(
                    time_range_batch,
                    output_path_jurisdiction
                    / f"{start.isoformat()}-{end.isoformat()}.jsonl",
                )
        bump_last_export_date(end_date)


@cli.command()
@click.argument(
    "input_path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--version",
    type=str,
    default=f"v{datetime.datetime.now().strftime('%Y.%m.%d')}",
    help="Version number for the release",
)
def release_note(input_path: Path, output_path: Path, version: str):
    """
    Generate a release note based on the exported data.

    Args:
        input_path: The directory where the exported data is located.
        output_path: The directory where the release-note will be written.
        version: The version number for the release. If not provided, uses the current date.
    """
    encoding = tiktoken.encoding_for_model("gpt-4")
    output_path = output_path / f"{version}.md"

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(os.path.join("release_notes", "template.jinja2"))

    jurisdictions = ["cour_d_appel", "cour_de_cassation", "tribunal_judiciaire"]
    jurisdiction_data = []
    total_jurisprudences = 0
    total_tokens = 0
    overall_oldest_date = datetime.date.max
    overall_latest_date = datetime.date.min
    download_base_url = (
        "https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main/"
    )
    total_jsonl_gz_size = 0
    total_parquet_size = 0

    for jurisdiction in jurisdictions:
        jsonl_gz_path = input_path / (jurisdiction + ".jsonl.gz")
        parquet_path = input_path / (jurisdiction + ".parquet")

        if not (jsonl_gz_path.exists() and parquet_path.exists()):
            console.log(f"[green]{jurisdiction} files do not exist, continue")
            continue

        jsonl_gz_size = jsonl_gz_path.stat().st_size
        parquet_size = parquet_path.stat().st_size
        total_jsonl_gz_size += jsonl_gz_size
        total_parquet_size += parquet_size

        jurisdiction_name = {
            "cour_d_appel": "Cour d'Appel",
            "tribunal_judiciaire": "Tribunal Judiciaire",
            "cour_de_cassation": "Cour de Cassation",
        }.get(jurisdiction, jurisdiction)

        conn = duckdb.connect(database=":memory:")
        conn.execute(
            f"CREATE TABLE data AS SELECT * FROM parquet_scan('{parquet_path}')"
        )

        oldest_date, latest_date = conn.execute(
            "SELECT MIN(decision_date) as oldest, MAX(decision_date) as latest FROM data"
        ).fetchone()
        oldest_date = datetime.datetime.strptime(oldest_date, "%Y-%m-%d").date()
        latest_date = datetime.datetime.strptime(latest_date, "%Y-%m-%d").date()
        overall_oldest_date = min(overall_oldest_date, oldest_date)
        overall_latest_date = max(overall_latest_date, latest_date)

        jurisprudence_count = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]

        conn.create_function(
            "count_tokens", lambda text: len(encoding.encode(text)), return_type=int
        )
        tokens = conn.execute("SELECT SUM(count_tokens(text)) FROM data").fetchone()[0]

        total_jurisprudences += jurisprudence_count
        total_tokens += tokens

        jsonl_gz_link = f"[Download ({_human_readable_size(jsonl_gz_size)})]({download_base_url}{jurisdiction}.jsonl.gz?download=true)"
        parquet_link = f"[Download ({_human_readable_size(parquet_size)})]({download_base_url}{jurisdiction}.parquet?download=true)"

        jurisdiction_data.append(
            {
                "name": jurisdiction_name,
                "count": jurisprudence_count,
                "oldest_date": oldest_date,
                "latest_date": latest_date,
                "tokens": tokens,
                "jsonl_gz_link": jsonl_gz_link,
                "parquet_link": parquet_link,
            }
        )

    context = {
        "version": version,
        "jurisdiction_data": jurisdiction_data,
        "total_jurisprudences": total_jurisprudences,
        "total_tokens": total_tokens,
        "overall_oldest_date": overall_oldest_date,
        "overall_latest_date": overall_latest_date,
        "update_date": version.lstrip("v").replace(".", "-"),
        "total_jsonl_gz_size": _human_readable_size(total_jsonl_gz_size),
        "total_parquet_size": _human_readable_size(total_parquet_size),
    }

    release_note = template.render(context)

    assert output_path.write_text(release_note)
    console.print(f"[green]Release note generated at:[/green] {output_path}")


@cli.command()
@click.argument(
    "input_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
)
def to_parquet(input_path: Path):
    """
    Convert a Jurisprudence JSONL file to Parquet format.

    Args:
        input_path (Path): The path to the input JSONL file.

    The function uses a predefined PARQUET_SCHEMA for writing the Parquet file.
    It reads the JSON file in blocks of 10 MB to handle large files efficiently.
    """
    with pq.ParquetWriter(input_path.with_suffix(".parquet"), PARQUET_SCHEMA) as writer:
        table = pj.read_json(
            input_path,
            parse_options=pj.ParseOptions(
                explicit_schema=PARQUET_SCHEMA, newlines_in_values=False
            ),
            read_options=pj.ReadOptions(block_size=10_000_000),
        )
        writer.write_table(table)


if __name__ == "__main__":
    cli()
