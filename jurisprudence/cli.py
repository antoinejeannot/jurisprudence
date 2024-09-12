import datetime
import os
from importlib import resources
from pathlib import Path
from typing import Literal

import click
import duckdb
import pyarrow.json as pj
import pyarrow.parquet as pq
import tiktoken
from jinja2 import Environment, FileSystemLoader

from jurisprudence import console
from jurisprudence.schema import PARQUET_SCHEMA, Jurisdiction
from jurisprudence.settings import JURISPRUDENCE_LAST_EXPORT_DATETIME
from jurisprudence.utils import (
    bump_last_export_date,
    human_readable_size,
    process_date_range,
    split_date_range,
    validate_export_batch,
)


@click.group()
def cli(): ...


@cli.command()
@click.argument(
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=".",
)
@click.option(
    "--start-date",
    type=click.DateTime(),
    default=JURISPRUDENCE_LAST_EXPORT_DATETIME or str(datetime.date(1800, 1, 1)),
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
    env = Environment(loader=FileSystemLoader("/"))
    template_path = str(
        resources.files("jurisprudence.templates").joinpath("release_note.jinja2")
    )
    template = env.get_template(template_path)
    jurisdictions = {
        "cour_d_appel": "Cour d'Appel",
        "tribunal_judiciaire": "Tribunal Judiciaire",
        "cour_de_cassation": "Cour de Cassation",
    }
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

    for jurisdiction, jurisdiction_name in jurisdictions.items():
        jsonl_gz_path = input_path / (jurisdiction + ".jsonl.gz")
        parquet_path = input_path / (jurisdiction + ".parquet")

        if not (jsonl_gz_path.exists() and parquet_path.exists()):
            console.log(f"[green]{jurisdiction} files do not exist, continue")
            continue

        jsonl_gz_size = jsonl_gz_path.stat().st_size
        parquet_size = parquet_path.stat().st_size
        total_jsonl_gz_size += jsonl_gz_size
        total_parquet_size += parquet_size

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

        jsonl_gz_link = f"[Download ({human_readable_size(jsonl_gz_size)})]({download_base_url}{jurisdiction}.jsonl.gz?download=true)"
        parquet_link = f"[Download ({human_readable_size(parquet_size)})]({download_base_url}{jurisdiction}.parquet?download=true)"

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
        "total_jsonl_gz_size": human_readable_size(total_jsonl_gz_size),
        "total_parquet_size": human_readable_size(total_parquet_size),
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
