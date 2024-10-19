import datetime
import enum
import json
from typing import Any, NotRequired, TypedDict

import pyarrow as pa
from pydantic import BaseModel, Field, field_serializer, field_validator


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


class Response(TypedDict):
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
    location: str | None = Field(None, description="Siège ayant rendu la décision.")
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
        pa.field("location", pa.string()),
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
        pa.field(
            "titlesAndSummaries", pa.string()
        ),  # Changed to string, #TODO: fix its structure
        pa.field("legacy", pa.string()),  # Changed to string, #TODO: fix its structure
    ]
)
