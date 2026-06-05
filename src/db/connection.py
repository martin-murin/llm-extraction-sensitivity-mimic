from __future__ import annotations

from collections import defaultdict

from sqlalchemy import Engine, create_engine, text

from src import config

_SCHEMA_CACHE: dict[int, dict[str, str | None]] = {}


def get_engine() -> Engine:
    uri = config.SETTINGS.mimic_pg_uri
    if not uri:
        raise RuntimeError(
            "MIMIC_PG_URI is not set. Set it in your environment or in a local .env file."
        )

    return create_engine(
        uri,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
    )


def discover_schemas(engine: Engine) -> dict[str, str | None]:
    cache_key = id(engine)
    cached = _SCHEMA_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    query = text(
        """
        SELECT t.table_schema, t.table_name, c.column_name
        FROM information_schema.tables AS t
        JOIN information_schema.columns AS c
            ON t.table_schema = c.table_schema
           AND t.table_name = c.table_name
        WHERE t.table_type = 'BASE TABLE'
          AND t.table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY t.table_schema, t.table_name, c.ordinal_position
        """
    )

    columns_by_table: dict[tuple[str, str], set[str]] = defaultdict(set)
    with engine.connect() as conn:
        for row in conn.execute(query):
            columns_by_table[(row.table_schema, row.table_name)].add(row.column_name)

    candidates_by_name: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for (schema, table_name), cols in columns_by_table.items():
        candidates_by_name[table_name].append((schema, cols))

    def pick_table(
        table_names: list[str],
        *,
        required_columns: set[str] | None = None,
        optional: bool = False,
    ) -> str | None:
        candidates = []
        for table_name in table_names:
            for schema, cols in candidates_by_name.get(table_name, []):
                if required_columns and not required_columns.issubset(cols):
                    continue
                candidates.append((schema, table_name))

        if not candidates:
            if optional:
                return None
            required = f" with columns {sorted(required_columns)}" if required_columns else ""
            raise RuntimeError(
                f"Required table in {table_names}{required} was not found in any schema."
            )

        def sort_key(item: tuple[str, str]) -> tuple[int, int, str]:
            schema_name, table_name = item
            table_priority = table_names.index(table_name)
            schema_priority = 0 if schema_name.startswith("mimic") else 1
            return (table_priority, schema_priority, schema_name)

        schema_name, table_name = sorted(candidates, key=sort_key)[0]
        return f"{schema_name}.{table_name}"

    resolved: dict[str, str | None] = {
        "discharge_notes": pick_table(
            ["discharge", "discharge_note"],
            required_columns={"hadm_id", "text"},
        ),
        "radiology_notes": pick_table(
            ["radiology", "radiology_note"],
            required_columns={"hadm_id", "text"},
            optional=True,
        ),
        "admissions": pick_table(["admissions"]),
        "diagnoses_icd": pick_table(["diagnoses_icd"]),
        "d_icd_diagnoses": pick_table(["d_icd_diagnoses"]),
        "patients": pick_table(["patients"]),
    }

    _SCHEMA_CACHE[cache_key] = dict(resolved)
    return resolved
