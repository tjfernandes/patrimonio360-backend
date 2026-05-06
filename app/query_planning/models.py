from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PrimitiveValue = str | int | float | bool
QueryMode = Literal["rag", "structured"]
PlanOperation = Literal["count", "list", "group_by", "exists"]
SortOrder = Literal["asc", "desc"]

_ALLOWED_FIELD_TYPES = {
    "keyword",
    "text",
    "integer",
    "long",
    "float",
    "double",
    "boolean",
    "date",
}
_NUMERIC_FIELD_TYPES = {"integer", "long", "float", "double"}


class QuerySchemaField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    facetable: bool = False
    text: bool = False
    semantic: bool = False

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_FIELD_TYPES:
            raise ValueError(f"Unsupported field type: {value}")
        return normalized


class RangeBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gt: int | float | None = None
    gte: int | float | None = None
    lt: int | float | None = None
    lte: int | float | None = None

    @model_validator(mode="after")
    def validate_not_empty(self) -> "RangeBounds":
        if all(value is None for value in (self.gt, self.gte, self.lt, self.lte)):
            raise ValueError("Range bounds cannot be empty.")
        return self


class TermFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["term"]
    field: str
    value: PrimitiveValue


class TermsFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["terms"]
    field: str
    values: list[PrimitiveValue]

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[PrimitiveValue]) -> list[PrimitiveValue]:
        if not value:
            raise ValueError("terms filter requires at least one value.")
        return value


class RangeFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["range"]
    field: str
    range: RangeBounds


class ExistsFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["exists"]
    field: str


FilterClause = Annotated[
    TermFilter | TermsFilter | RangeFilter | ExistsFilter,
    Field(discriminator="kind"),
]


class SortClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    order: SortOrder = "asc"


class ListSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=10, ge=1, le=100)
    fields: list[str] = Field(default_factory=list)
    sort: list[SortClause] = Field(default_factory=list)


class GroupBySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    size: int = Field(default=10, ge=1, le=100)
    order: Literal["count_desc", "count_asc", "term_asc", "term_desc"] = "count_desc"


class QuerySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: str
    fields: dict[str, QuerySchemaField]
    facetable_fields: list[str] = Field(default_factory=list)
    text_fields: list[str] = Field(default_factory=list)
    semantic_fields: list[str] = Field(default_factory=list)
    default_filters: list[FilterClause] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_field_groups(self) -> "QuerySchema":
        existing_fields = set(self.fields.keys())

        if not self.facetable_fields:
            self.facetable_fields = [
                name for name, spec in self.fields.items() if spec.facetable
            ]
        else:
            self.facetable_fields = [
                field for field in self.facetable_fields if field in existing_fields
            ]

        if not self.text_fields:
            self.text_fields = [name for name, spec in self.fields.items() if spec.text]
        else:
            self.text_fields = [field for field in self.text_fields if field in existing_fields]

        if not self.semantic_fields:
            self.semantic_fields = [name for name, spec in self.fields.items() if spec.semantic]
        else:
            self.semantic_fields = [
                field for field in self.semantic_fields if field in existing_fields
            ]

        return self

    def field_type(self, field_name: str) -> str:
        if field_name not in self.fields:
            raise ValueError(f"Unknown field in schema: {field_name}")
        return self.fields[field_name].type

    def is_numeric(self, field_name: str) -> bool:
        return self.field_type(field_name) in _NUMERIC_FIELD_TYPES

    def is_facetable(self, field_name: str) -> bool:
        return field_name in self.facetable_fields


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["structured"] = "structured"
    operation: PlanOperation
    confidence: float = Field(ge=0.0, le=1.0)
    query_text: str | None = None
    semantic_query: str | None = None
    filters: list[FilterClause] = Field(default_factory=list)
    list_spec: ListSpec | None = None
    group_by: GroupBySpec | None = None

    @model_validator(mode="after")
    def validate_operation_payload(self) -> "QueryPlan":
        if self.operation == "list":
            if self.list_spec is None:
                self.list_spec = ListSpec()
            self.group_by = None
        elif self.operation == "group_by":
            if self.group_by is None:
                raise ValueError("group_by operation requires group_by spec.")
            self.list_spec = None
        else:
            self.list_spec = None
            self.group_by = None
        return self

    @field_validator("query_text", "semantic_query")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class CompiledOpenSearchDSL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: Literal["_count", "_search"]
    index: str
    body: dict[str, Any]


class GroupBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    doc_count: int


class QueryExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: PlanOperation
    count: int | None = None
    exists: bool | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    groups: list[GroupBucket] = Field(default_factory=list)
    total: int | None = None

