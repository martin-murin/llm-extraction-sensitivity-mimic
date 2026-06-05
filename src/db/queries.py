from __future__ import annotations

import logging

import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import Engine, bindparam, text

from src.db.connection import discover_schemas

logger = logging.getLogger(__name__)


def _quote_qualified_name(qualified_name: str) -> str:
    schema, table = qualified_name.split(".", maxsplit=1)
    schema = schema.replace('"', '""')
    table = table.replace('"', '""')
    return f'"{schema}"."{table}"'


def _table_names(engine: Engine) -> dict[str, str | None]:
    return discover_schemas(engine)


def count_notes(engine: Engine) -> int:
    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))

    query = text(f"SELECT COUNT(*) AS n FROM {discharge}")
    with engine.connect() as conn:
        return int(conn.execute(query).scalar_one())


def note_length_stats(engine: Engine, sample_size: int = 10000) -> pd.DataFrame:
    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))

    query = text(
        f"""
        SELECT
            hadm_id,
            COALESCE(char_length(text), 0)::bigint AS char_len,
            CASE
                WHEN text IS NULL OR btrim(text) = '' THEN 0
                ELSE array_length(regexp_split_to_array(btrim(text), E'\\s+'), 1)
            END::bigint AS word_len_approx
        FROM {discharge}
        ORDER BY random()
        LIMIT :sample_size
        """
    )
    return pd.read_sql_query(query, engine, params={"sample_size": sample_size})


def top_primary_icds(engine: Engine, n: int = 20) -> pd.DataFrame:
    tables = _table_names(engine)
    diagnoses = _quote_qualified_name(str(tables["diagnoses_icd"]))
    dictionary = _quote_qualified_name(str(tables["d_icd_diagnoses"]))

    query = text(
        f"""
        SELECT
            d.icd_code,
            d.icd_version,
            COUNT(*)::bigint AS count,
            COALESCE(MAX(dict.long_title), '') AS description
        FROM {diagnoses} AS d
        LEFT JOIN {dictionary} AS dict
          ON d.icd_code = dict.icd_code
         AND CAST(d.icd_version AS text) = CAST(dict.icd_version AS text)
        WHERE CAST(d.seq_num AS text) = '1'
        GROUP BY d.icd_code, d.icd_version
        ORDER BY count DESC, d.icd_code
        LIMIT :n
        """
    )
    return pd.read_sql_query(query, engine, params={"n": n})


def top_any_position_icds(engine: Engine, n: int = 20) -> pd.DataFrame:
    tables = _table_names(engine)
    diagnoses = _quote_qualified_name(str(tables["diagnoses_icd"]))
    dictionary = _quote_qualified_name(str(tables["d_icd_diagnoses"]))

    query = text(
        f"""
        SELECT
            d.icd_code,
            d.icd_version,
            COUNT(*)::bigint AS count,
            COALESCE(MAX(dict.long_title), '') AS description
        FROM {diagnoses} AS d
        LEFT JOIN {dictionary} AS dict
          ON d.icd_code = dict.icd_code
         AND CAST(d.icd_version AS text) = CAST(dict.icd_version AS text)
        GROUP BY d.icd_code, d.icd_version
        ORDER BY count DESC, d.icd_code
        LIMIT :n
        """
    )
    return pd.read_sql_query(query, engine, params={"n": n})


def join_cardinality(engine: Engine) -> dict[str, int]:
    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))
    admissions = _quote_qualified_name(str(tables["admissions"]))

    query = text(
        f"""
        WITH notes AS (
            SELECT hadm_id FROM {discharge}
        ), admissions AS (
            SELECT hadm_id FROM {admissions}
        )
        SELECT
            (SELECT COUNT(*) FROM notes) AS n_discharge_notes,
            (SELECT COUNT(DISTINCT hadm_id) FROM notes WHERE hadm_id IS NOT NULL)
                AS n_unique_hadm_ids_in_notes,
            (SELECT COUNT(*) FROM admissions) AS n_admissions,
            (
                SELECT COUNT(DISTINCT a.hadm_id)
                FROM admissions AS a
                JOIN notes AS n ON a.hadm_id = n.hadm_id
            ) AS n_admissions_with_discharge_note,
            (
                SELECT COUNT(*)
                FROM notes AS n
                LEFT JOIN admissions AS a ON n.hadm_id = a.hadm_id
                WHERE a.hadm_id IS NULL
            ) AS n_discharge_notes_orphan
        """
    )

    with engine.connect() as conn:
        row = conn.execute(query).mappings().one()

    return {key: int(value) for key, value in row.items()}


def sample_redaction_excerpts(
    engine: Engine,
    n: int = 3,
    max_chars: int = 400,
) -> list[dict[str, int | str | None]]:
    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))

    query = text(
        f"""
        SELECT
            hadm_id,
            LEFT(COALESCE(text, ''), :max_chars) AS excerpt
        FROM {discharge}
        ORDER BY random()
        LIMIT :n
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"n": n, "max_chars": max_chars}).mappings().all()

    return [{"hadm_id": row["hadm_id"], "excerpt": row["excerpt"]} for row in rows]


def fetch_notes_by_hadm_ids(engine: Engine, hadm_ids: list[int]) -> dict[int, str]:
    if not hadm_ids:
        return {}

    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))

    query = text(
        f"""
        SELECT hadm_id, COALESCE(text, '') AS text
        FROM {discharge}
        WHERE hadm_id IN :hadm_ids
        ORDER BY hadm_id
        """
    ).bindparams(bindparam("hadm_ids", expanding=True))

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"hadm_ids": [int(hadm_id) for hadm_id in hadm_ids]},
        ).mappings().all()

    return {int(row["hadm_id"]): str(row["text"]) for row in rows}


def fetch_all_discharge_hadm_ids(engine: Engine) -> list[int]:
    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))

    query = text(
        f"""
        SELECT DISTINCT hadm_id::bigint AS hadm_id
        FROM {discharge}
        WHERE hadm_id IS NOT NULL
        ORDER BY hadm_id::bigint
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [int(row["hadm_id"]) for row in rows]


def fetch_all_discharge_notes(engine: Engine) -> dict[int, str]:
    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))

    query = text(
        f"""
        SELECT hadm_id::bigint AS hadm_id, COALESCE(text, '') AS text
        FROM {discharge}
        WHERE hadm_id IS NOT NULL
        ORDER BY hadm_id::bigint
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    # If duplicate hadm_id rows exist, keep first deterministic row by ORDER BY.
    out: dict[int, str] = {}
    for row in rows:
        hadm_id = int(row["hadm_id"])
        if hadm_id not in out:
            out[hadm_id] = str(row["text"])
    return out


def fetch_icd_codes_by_hadm_ids(
    engine: Engine,
    hadm_ids: list[int],
) -> dict[int, list[tuple[str, int]]]:
    if not hadm_ids:
        return {}

    tables = _table_names(engine)
    diagnoses = _quote_qualified_name(str(tables["diagnoses_icd"]))

    query = text(
        f"""
        SELECT
            d.hadm_id::bigint AS hadm_id,
            d.icd_code::text AS icd_code,
            CASE
                WHEN CAST(d.icd_version AS text) ~ '^[0-9]+$'
                    THEN CAST(d.icd_version AS integer)
                ELSE NULL
            END AS icd_version,
            CASE
                WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                    THEN CAST(d.seq_num AS integer)
                ELSE NULL
            END AS seq_num_int
        FROM {diagnoses} AS d
        WHERE d.hadm_id IN :hadm_ids
        ORDER BY
            d.hadm_id,
            CASE
                WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                    THEN 0
                ELSE 1
            END,
            COALESCE(
                CASE
                    WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                        THEN CAST(d.seq_num AS integer)
                    ELSE NULL
                END,
                2147483647
            ),
            d.icd_code,
            CAST(d.icd_version AS text)
        """
    ).bindparams(bindparam("hadm_ids", expanding=True))

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"hadm_ids": [int(hadm_id) for hadm_id in hadm_ids]},
        ).mappings().all()

    result: dict[int, list[tuple[str, int]]] = {int(hadm_id): [] for hadm_id in hadm_ids}
    for row in rows:
        hadm_id = int(row["hadm_id"])
        icd_code = str(row["icd_code"])
        icd_version_raw = row["icd_version"]
        if icd_version_raw is None:
            continue
        result.setdefault(hadm_id, []).append((icd_code, int(icd_version_raw)))

    logger.info(
        "fetch_icd_codes_by_hadm_ids returned %s ICD rows for %s admissions",
        len(rows),
        len(hadm_ids),
    )
    return result


def fetch_primary_icd_by_hadm_ids(
    engine: Engine,
    hadm_ids: list[int],
) -> dict[int, tuple[str, int]]:
    if not hadm_ids:
        return {}

    tables = _table_names(engine)
    diagnoses = _quote_qualified_name(str(tables["diagnoses_icd"]))

    query = text(
        f"""
        WITH ranked AS (
            SELECT
                d.hadm_id::bigint AS hadm_id,
                d.icd_code::text AS icd_code,
                CASE
                    WHEN CAST(d.icd_version AS text) ~ '^[0-9]+$'
                        THEN CAST(d.icd_version AS integer)
                    ELSE NULL
                END AS icd_version,
                ROW_NUMBER() OVER (
                    PARTITION BY d.hadm_id
                    ORDER BY
                        CASE
                            WHEN (
                                CASE
                                    WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                                        THEN CAST(d.seq_num AS integer)
                                    ELSE NULL
                                END
                            ) = 1
                                THEN 0
                            ELSE 1
                        END,
                        COALESCE(
                            CASE
                                WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                                    THEN CAST(d.seq_num AS integer)
                                ELSE NULL
                            END,
                            2147483647
                        ),
                        COALESCE(d.icd_code, '') ASC,
                        CAST(d.icd_version AS text) ASC
                ) AS rn
            FROM {diagnoses} AS d
            WHERE d.hadm_id IN :hadm_ids
        )
        SELECT hadm_id, icd_code, icd_version
        FROM ranked
        WHERE rn = 1
        """
    ).bindparams(bindparam("hadm_ids", expanding=True))

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"hadm_ids": [int(hadm_id) for hadm_id in hadm_ids]},
        ).mappings().all()

    result: dict[int, tuple[str, int]] = {}
    for row in rows:
        icd_version = row["icd_version"]
        if icd_version is None:
            continue
        result[int(row["hadm_id"])] = (str(row["icd_code"]), int(icd_version))
    return result


def pull_split_candidates(engine: Engine) -> pd.DataFrame:
    tables = _table_names(engine)
    discharge = _quote_qualified_name(str(tables["discharge_notes"]))
    admissions = _quote_qualified_name(str(tables["admissions"]))
    diagnoses = _quote_qualified_name(str(tables["diagnoses_icd"]))

    query = text(
        f"""
        WITH notes_ranked AS (
            SELECT
                dn.hadm_id,
                dn.subject_id,
                COALESCE(char_length(dn.text), 0)::bigint AS note_char_len,
                ROW_NUMBER() OVER (
                    PARTITION BY dn.hadm_id
                    ORDER BY dn.ctid
                ) AS rn_note
            FROM {discharge} AS dn
            WHERE dn.hadm_id IS NOT NULL
        ),
        notes AS (
            SELECT hadm_id, subject_id, note_char_len
            FROM notes_ranked
            WHERE rn_note = 1
        ),
        admissible_notes AS (
            SELECT
                n.hadm_id,
                n.subject_id,
                n.note_char_len
            FROM notes AS n
            INNER JOIN {admissions} AS a
                ON a.hadm_id = n.hadm_id
        ),
        diagnoses_ranked AS (
            SELECT
                d.hadm_id,
                d.icd_code,
                d.icd_version,
                CASE
                    WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                        THEN CAST(d.seq_num AS integer)
                    ELSE NULL
                END AS seq_num_int,
                ROW_NUMBER() OVER (
                    PARTITION BY d.hadm_id
                    ORDER BY
                        CASE
                            WHEN (
                                CASE
                                    WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                                        THEN CAST(d.seq_num AS integer)
                                    ELSE NULL
                                END
                            ) = 1
                                THEN 0
                            ELSE 1
                        END,
                        COALESCE(
                            CASE
                                WHEN CAST(d.seq_num AS text) ~ '^[0-9]+$'
                                    THEN CAST(d.seq_num AS integer)
                                ELSE NULL
                            END,
                            2147483647
                        ),
                        COALESCE(d.icd_code, '') ASC,
                        CAST(d.icd_version AS text) ASC
                ) AS rn_primary,
                COUNT(*) OVER (PARTITION BY d.hadm_id) AS n_diagnoses
            FROM {diagnoses} AS d
            WHERE d.hadm_id IS NOT NULL
        ),
        diagnoses_primary AS (
            SELECT
                dr.hadm_id,
                dr.icd_code::text AS primary_icd_code,
                CASE
                    WHEN CAST(dr.icd_version AS text) ~ '^[0-9]+$'
                        THEN CAST(dr.icd_version AS integer)
                    ELSE NULL
                END AS primary_icd_version,
                dr.n_diagnoses::bigint AS n_diagnoses
            FROM diagnoses_ranked AS dr
            WHERE dr.rn_primary = 1
        )
        SELECT
            an.hadm_id::bigint AS hadm_id,
            an.subject_id::bigint AS subject_id,
            dp.primary_icd_code,
            dp.primary_icd_version,
            an.note_char_len::bigint AS note_char_len,
            dp.n_diagnoses::bigint AS n_diagnoses
        FROM admissible_notes AS an
        INNER JOIN diagnoses_primary AS dp
            ON dp.hadm_id = an.hadm_id
        ORDER BY an.hadm_id
        """
    )

    frame = pd.read_sql_query(query, engine)
    frame["hadm_id"] = pd.to_numeric(frame["hadm_id"], errors="coerce").astype("int64")
    frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce").astype("int64")
    frame["primary_icd_code"] = frame["primary_icd_code"].astype(str)
    frame["primary_icd_version"] = pd.to_numeric(
        frame["primary_icd_version"], errors="coerce"
    ).astype("int64")
    frame["note_char_len"] = pd.to_numeric(frame["note_char_len"], errors="coerce").astype("int64")
    frame["n_diagnoses"] = pd.to_numeric(frame["n_diagnoses"], errors="coerce").astype("int64")

    frame.attrs["source_table"] = str(tables["discharge_notes"])
    logger.info("pull_split_candidates returned %s rows", len(frame))
    return frame
