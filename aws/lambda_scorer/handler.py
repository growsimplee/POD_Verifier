"""
POD scoring Lambda — event-driven per trip, with a batch mode kept for backfills.

PRIMARY (event-driven, one trip):
    Sarathy invokes this function asynchronously when a rider raises a POD
    verification request:

        {"trip_id": 12345, "pod_links": ["https://..."], "awb": "..."}   # links optional

    The function resolves that trip's POD images (TRIP_QUERY against Postgres,
    falling back to the links carried in the event), scores them, and upserts
    into pod_scores. Small, fast, no windowing/continuation needed.

SECONDARY (batch, on demand from Sarathy's admin API or a manual invoke):
    An event with no trip_id scores a whole selection of rows, chosen by:

        {"start_date": "2026-08-01", "end_date": "2026-08-20"}   -> RANGE_QUERY
        {"query": "SELECT awb, trip_id, pod FROM ..."}           -> validated ad-hoc SQL
        {}                                                       -> SOURCE_QUERY (default)

    read POD rows from Postgres  ->  expand links (kept bound to awb/trip)  ->
    skip already-scored (resume)  ->  for each window:
        download images concurrently into memory  ->  EfficientNet score  ->
        idempotent upsert to Postgres  ->  free memory
    ->  if near the time limit, hand off one checkpointed continuation.


BOOTSTRAP:
    {"migrate": true} applies infra/schema.sql from inside the VPC, so CI can
    create the database objects without reaching a private RDS itself.

WARMUP:
    {"warmup": true} loads the checkpoint and returns, keeping a small pool of
    containers hot so rider-triggered scoring does not pay a cold start.

Design guarantees:
  * warm reuse      — model, DB connection and HTTP pool live for the container.
  * no rework       — a re-request only scores POD links that are new or changed.
  * read-only SQL   — ad-hoc queries are validated before they reach Postgres.
  * bounded memory  — only one WINDOW_SIZE of images is held at a time.
  * full coverage   — every input ends as 'scored' or 'download_failed'.
  * idempotent      — upsert on (awb, pod_link, run_date); safe to re-run/resume.
  * time-safe       — a clock check hands off before the 15-minute wall.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
import uuid
from datetime import date
from typing import Any, Optional

import boto3
import cv2
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import torch

from src.model import ATTRIBUTE_NAMES, ATTRIBUTE_WEIGHTS, MultiHeadEfficientNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration (all from environment; set by CloudFormation)
# --------------------------------------------------------------------------- #

# Results database (where pod_scores lives).
PG_HOST = os.environ.get("PG_HOST", "")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DATABASE = os.environ.get("PG_DATABASE", "pod_classifier")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")

# Source of the POD rows: SQL run against Postgres, returning an AWB column, a
# trip-id column, and a column of POD image link(s) named one of POD / pod /
# pod_link (comma-separated links are expanded).
#
# The source lives in a different DATABASE from the results: POD rows come from
# `kaptaan` in the application database, while pod_scores is written wherever
# PG_DATABASE points. Postgres cannot query across databases on one connection,
# so fetch_pod_data() dials its own using SOURCE_PG_* — each of which inherits
# the PG_* value when left empty, so pointing at a sibling database on the same
# cluster means setting SOURCE_PG_DATABASE alone.
SOURCE_QUERY = os.environ.get("SOURCE_QUERY", "")

# Single-trip mode: SQL run with a named %(trip_id)s parameter to resolve one
# trip's POD rows. Same column contract as SOURCE_QUERY.
TRIP_QUERY = os.environ.get(
    "TRIP_QUERY",
    "SELECT awb, trip_id, pod FROM kaptaan WHERE trip_id = %(trip_id)s",
)

# Batch/backfill over an explicit date range: bound as named parameters.
RANGE_QUERY = os.environ.get(
    "RANGE_QUERY",
    "SELECT awb, trip_id, pod FROM kaptaan "
    "WHERE tour_date BETWEEN %(start_date)s AND %(end_date)s",
)

# Ad-hoc SQL carried in the trigger event (admin API). Validated by
# _validate_adhoc_query before it is ever sent to Postgres.
ALLOW_ADHOC_QUERY = os.environ.get("ALLOW_ADHOC_QUERY", "true").lower() == "true"

# `or` rather than a get() default on purpose: CloudFormation always sets these
# env vars, empty when unset, and an empty string must inherit the results-DB
# value instead of blanking it.
SOURCE_PG_HOST = os.environ.get("SOURCE_PG_HOST") or PG_HOST
SOURCE_PG_PORT = os.environ.get("SOURCE_PG_PORT") or PG_PORT
SOURCE_PG_DATABASE = os.environ.get("SOURCE_PG_DATABASE") or PG_DATABASE
SOURCE_PG_USER = os.environ.get("SOURCE_PG_USER") or PG_USER
SOURCE_PG_PASSWORD = os.environ.get("SOURCE_PG_PASSWORD") or PG_PASSWORD

# Model / scoring.
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/best.pt")
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "224"))
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "64"))
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.7"))
# Must match training preprocessing — do not disable in production.
IMAGENET_NORMALIZE = os.environ.get("IMAGENET_NORMALIZE", "true").lower() == "true"

# Download / memory tuning.
MAX_DOWNLOAD_WORKERS = int(os.environ.get("MAX_DOWNLOAD_WORKERS", "64"))
# A trip has a handful of images; 64 threads per invocation would only multiply
# NAT/source-host pressure once many trips are scored concurrently.
TRIP_MAX_DOWNLOAD_WORKERS = int(os.environ.get("TRIP_MAX_DOWNLOAD_WORKERS", "8"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "800"))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "15"))
MIN_CONTENT_BYTES = int(os.environ.get("MIN_CONTENT_BYTES", "500"))

# Resilience.
# Re-request de-dup: a pod_link already 'scored' within this many days is not
# re-downloaded. Trip POD links can change between requests, so this is keyed on
# the link itself — new/changed links are always scored. 0 = look back forever.
RESCORE_LOOKBACK_DAYS = int(os.environ.get("RESCORE_LOOKBACK_DAYS", "30"))

# Warm-pool: how many containers a warmup ping keeps alive (1 = just this one).
WARM_FANOUT = int(os.environ.get("WARM_FANOUT", "3"))

# Schema, baked into the image by deploy.sh. Applied by the {"migrate": true}
# event: the function sits inside the VPC, so it is the only thing that can
# reach a private RDS without a bastion — which is what lets CI bootstrap the
# database on its own.
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", "/opt/schema/schema.sql")

# Objects the destructive teardown drops. Kept explicit so the drop can never
# widen to "whatever is in the database".
TEARDOWN_OBJECTS = ("DROP VIEW IF EXISTS pod_scores_flagged",
                    "DROP TABLE IF EXISTS pod_scores")

CONTINUATION_SAFETY_MS = int(os.environ.get("CONTINUATION_SAFETY_MS", "90000"))
MAX_CONTINUATIONS = int(os.environ.get("MAX_CONTINUATIONS", "5"))

LAMBDA_FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
CW_NAMESPACE = os.environ.get("CW_NAMESPACE", "PODPipeline")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Warm-container state
#
# Everything expensive is built once per container and reused by every later
# invocation that lands on it: the EfficientNet checkpoint (seconds), the
# Postgres connection (TLS + auth round-trips) and the HTTP connection pool.
# Under load — many trips scored concurrently — Lambda runs several containers
# side by side, and each pays this cost exactly once. A warmup ping (see
# handle_warmup) keeps a small pool of them alive between rider requests.
# --------------------------------------------------------------------------- #

_model: Optional[MultiHeadEfficientNet] = None
_device: Optional[torch.device] = None
_session: Optional[requests.Session] = None
_conn = None


def get_model() -> tuple[MultiHeadEfficientNet, torch.device]:
    global _model, _device
    if _model is None:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _device = torch.device("cpu")
        _model = MultiHeadEfficientNet(num_attributes=4, pretrained=False)
        ckpt = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        _model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        _model.to(_device).eval()
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model, _device


# --------------------------------------------------------------------------- #
# Postgres helpers
# --------------------------------------------------------------------------- #

def _connect(host, port, database, user, password):
    return psycopg2.connect(host=host, port=port, database=database,
                            user=user, password=password, connect_timeout=10)


def get_db_connection():
    """Connection to the results database (pod_scores), reused while warm.

    Reconnects transparently if the cached handle was closed or the server hung
    up between invocations (idle timeout, failover).
    """
    global _conn
    if _conn is not None and not _conn.closed:
        try:
            with _conn.cursor() as cur:
                cur.execute("SELECT 1")
            return _conn
        except Exception:  # noqa: BLE001 — stale handle; drop it and redial
            try:
                _conn.close()
            except Exception:  # noqa: BLE001
                pass
            _conn = None
    _conn = _connect(PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD)
    return _conn


def release_db_connection(conn) -> None:
    """Close only connections that are NOT the warm cached handle.

    The batch path opens/closes around a long run; the per-trip path keeps the
    cached connection open so the next trip on this container skips the dial.
    """
    if conn is not None and conn is not _conn:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def fetch_pod_data(query: Optional[str] = None,
                   params: Optional[dict] = None) -> pd.DataFrame:
    """Run a POD-source query against the source Postgres DB and return the rows."""
    sql = query if query is not None else SOURCE_QUERY
    conn = _connect(SOURCE_PG_HOST, SOURCE_PG_PORT, SOURCE_PG_DATABASE,
                    SOURCE_PG_USER, SOURCE_PG_PASSWORD)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params) if params is not None else cur.execute(sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


def fetch_trip_pod_data(trip_id: str) -> pd.DataFrame:
    """Resolve one trip's POD rows via TRIP_QUERY (parameterised — no interpolation)."""
    return fetch_pod_data(TRIP_QUERY, {"trip_id": trip_id})


_FORBIDDEN_SQL = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "copy", "vacuum", "call", "do", "merge",
)


def _validate_adhoc_query(sql: str) -> str:
    """Accept only a single read-only SELECT/WITH statement.

    The trigger API lets an operator pass raw SQL, so this is the boundary that
    keeps a reporting feature from becoming a write primitive. Rejections raise
    ValueError and surface as a 400 — never as a silently-different query.
    """
    if not ALLOW_ADHOC_QUERY:
        raise ValueError("Ad-hoc queries are disabled (ALLOW_ADHOC_QUERY=false)")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("query must be a non-empty string")

    cleaned = sql.strip().rstrip(";").strip()
    if ";" in cleaned:
        raise ValueError("query must be a single statement (no ';')")
    if "--" in cleaned or "/*" in cleaned:
        raise ValueError("query must not contain SQL comments")

    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("query must start with SELECT or WITH")
    for word in _FORBIDDEN_SQL:
        if re.search(rf"\b{word}\b", lowered):
            raise ValueError(f"query must be read-only (found '{word}')")
    return cleaned


def resolve_batch_source(event: dict) -> tuple[str, Optional[dict], str]:
    """Pick the SQL for a batch run: ad-hoc query > date range > SOURCE_QUERY.

    Returns (sql, params, source_label). Raises ValueError on a bad request.
    """
    if event.get("query"):
        return _validate_adhoc_query(event["query"]), None, "adhoc_query"

    start, end = event.get("start_date"), event.get("end_date")
    if start or end:
        start = start or end
        end = end or start
        for label, value in (("start_date", start), ("end_date", end)):
            try:
                date.fromisoformat(str(value))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{label} must be ISO YYYY-MM-DD") from exc
        if str(start) > str(end):
            raise ValueError("start_date must not be after end_date")
        if not RANGE_QUERY:
            raise ValueError("RANGE_QUERY is not configured")
        return RANGE_QUERY, {"start_date": str(start), "end_date": str(end)}, "date_range"

    if not SOURCE_QUERY:
        raise ValueError("Missing SOURCE_QUERY")
    return SOURCE_QUERY, None, "source_query"


def rows_from_event(event: dict, trip_id: str) -> list[dict]:
    """Fallback source: POD links carried in the trigger event itself.

    Used when TRIP_QUERY returns nothing — e.g. Sarathy fires the event the
    moment the rider raises the request, before the source table has the row.
    """
    links = event.get("pod_links") or event.get("pod") or []
    if isinstance(links, str):
        links = [l.strip() for l in links.split(",")]
    awb = str(event.get("awb") or "").strip() or f"TRIP-{trip_id}"
    seen, rows = set(), []
    for link in (str(l).strip() for l in links):
        if link.startswith("http") and link not in seen:
            seen.add(link)
            rows.append({"awb": awb, "trip_id": str(trip_id), "pod_link": link})
    return rows


def expand_pod_links(df: pd.DataFrame) -> pd.DataFrame:
    """Explode comma-separated POD links into one row each, keeping awb + trip_id."""
    pod_col = next((c for c in ("POD", "pod", "pod_link") if c in df.columns), None)
    if pod_col is None:
        raise ValueError("No POD link column (POD / pod / pod_link) in source rows")

    awb_col = "AWB" if "AWB" in df.columns else "awb"
    trip_col = "Trip Id" if "Trip Id" in df.columns else "trip_id"

    rows = []
    for _, row in df.iterrows():
        raw = row.get(pod_col, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        for link in (l.strip() for l in raw.split(",")):
            if link.startswith("http"):
                rows.append({"awb": str(row.get(awb_col, "")),
                             "trip_id": str(row.get(trip_col, "")),
                             "pod_link": link})
    return (pd.DataFrame(rows)
            .drop_duplicates(subset=["awb", "pod_link"])
            .reset_index(drop=True))


def load_scored_links(conn, trip_id: str, pod_links: list[str]) -> set:
    """Which of these pod_links are already 'scored' for this trip?

    Keyed on the LINK, not the trip: a rider can re-raise a request after
    replacing the POD photos, so the trip is not a useful unit of "done" — the
    individual image is. Anything new or changed therefore always gets scored,
    and anything unchanged is not re-downloaded.

    RESCORE_LOOKBACK_DAYS bounds how far back a link counts as already done
    (0 = forever), so a genuinely old score can age out and be refreshed.
    """
    if not pod_links:
        return set()
    sql = ("SELECT pod_link FROM pod_scores "
           "WHERE trip_id = %(trip_id)s AND pod_link = ANY(%(links)s) "
           "AND status = 'scored' AND pod_score IS NOT NULL")
    params = {"trip_id": str(trip_id), "links": list(pod_links)}
    if RESCORE_LOOKBACK_DAYS > 0:
        sql += " AND run_date >= CURRENT_DATE - %(lookback)s::int"
        params["lookback"] = RESCORE_LOOKBACK_DAYS
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return {r[0] for r in cur.fetchall()}


def load_done_keys(conn, run_date: str) -> set:
    """Resume checkpoint: (awb, pod_link) already scored today."""
    with conn.cursor() as cur:
        cur.execute("SELECT awb, pod_link FROM pod_scores "
                    "WHERE run_date = %s AND status = 'scored'", (run_date,))
        return {(awb, link) for awb, link in cur.fetchall()}


def upsert_results(conn, rows: list[dict], run_date: str) -> int:
    """Idempotent bulk upsert on (awb, pod_link, run_date)."""
    if not rows:
        return 0
    sql = """
        INSERT INTO pod_scores
            (awb, trip_id, pod_link, run_date, status, failure_reason,
             pod_score, context_valid_prob, package_visible_prob,
             label_readable_prob, image_clarity_prob)
        VALUES %s
        ON CONFLICT (awb, pod_link, run_date) DO UPDATE SET
            status = EXCLUDED.status,
            failure_reason = EXCLUDED.failure_reason,
            pod_score = EXCLUDED.pod_score,
            context_valid_prob = EXCLUDED.context_valid_prob,
            package_visible_prob = EXCLUDED.package_visible_prob,
            label_readable_prob = EXCLUDED.label_readable_prob,
            image_clarity_prob = EXCLUDED.image_clarity_prob,
            scored_at = NOW()
    """
    tuples = [(r["awb"], r.get("trip_id"), r["pod_link"], run_date, r["status"],
               r.get("failure_reason"), r.get("pod_score"),
               r.get("context_valid_prob"), r.get("package_visible_prob"),
               r.get("label_readable_prob"), r.get("image_clarity_prob"))
              for r in rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, tuples, page_size=200)
    conn.commit()
    return len(tuples)


# --------------------------------------------------------------------------- #
# Download + preprocess + score
# --------------------------------------------------------------------------- #

def build_session(max_workers: int = MAX_DOWNLOAD_WORKERS) -> requests.Session:
    """HTTP session with a keep-alive pool, cached for the life of the container."""
    global _session
    if _session is not None:
        return _session
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers, pool_maxsize=max_workers,
        max_retries=requests.adapters.Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _session = session
    return session


def preprocess_image(img_rgb: np.ndarray, size: int = INPUT_SIZE,
                     normalize: bool = IMAGENET_NORMALIZE) -> np.ndarray:
    """Resize -> float[0,1] -> (optional) ImageNet normalize -> CHW float32."""
    img = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0
    if normalize:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(img, (2, 0, 1)).astype(np.float32)


def download_and_prepare(session: requests.Session, row: dict) -> dict:
    """Download + decode + resize one image. Always returns an outcome dict."""
    base = {"awb": row["awb"], "trip_id": row["trip_id"], "pod_link": row["pod_link"]}
    try:
        resp = session.get(row["pod_link"], timeout=DOWNLOAD_TIMEOUT)
        if resp.status_code != 200:
            return {**base, "status": "download_failed", "failure_reason": f"http_{resp.status_code}"}
        if len(resp.content) < MIN_CONTENT_BYTES:
            return {**base, "status": "download_failed", "failure_reason": "too_small"}
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {**base, "status": "download_failed", "failure_reason": "decode_failed"}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {**base, "chw": preprocess_image(img_rgb)}   # prepared tensor = success
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "download_failed", "failure_reason": type(e).__name__}


def download_window(session: requests.Session, rows: list[dict],
                    max_workers: int = MAX_DOWNLOAD_WORKERS) -> tuple[list[dict], list[dict]]:
    """Concurrently download a window. Returns (prepared, failures)."""
    prepared, failures = [], []
    workers = min(max_workers, max(1, len(rows)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(lambda r: download_and_prepare(session, r), rows):
            (prepared if "chw" in out else failures).append(out)
    return prepared, failures


def score_prepared(model, device, successes: list[dict]) -> list[dict]:
    """Run inference; attach per-attribute probs + weighted composite pod_score."""
    results = []
    for start in range(0, len(successes), INFERENCE_BATCH_SIZE):
        chunk = successes[start:start + INFERENCE_BATCH_SIZE]
        batch = torch.from_numpy(np.stack([s["chw"] for s in chunk])).to(device)
        with torch.no_grad():
            logits = model(batch)
        probs = {name: torch.sigmoid(logits[name]).cpu() for name in ATTRIBUTE_NAMES}
        composite = sum(probs[ATTRIBUTE_NAMES[i]] * ATTRIBUTE_WEIGHTS[i] for i in range(4))
        for j, s in enumerate(chunk):
            results.append({
                "awb": s["awb"], "trip_id": s["trip_id"], "pod_link": s["pod_link"],
                "status": "scored", "failure_reason": None,
                "pod_score": round(float(composite[j]), 6),
                "context_valid_prob": round(float(probs["context_valid"][j]), 6),
                "package_visible_prob": round(float(probs["package_visible"][j]), 6),
                "label_readable_prob": round(float(probs["label_readable"][j]), 6),
                "image_clarity_prob": round(float(probs["image_clarity"][j]), 6),
            })
    return results


# --------------------------------------------------------------------------- #
# Observability + self-continuation
# --------------------------------------------------------------------------- #

def emit_coverage(total: int, scored: int, failed: int) -> None:
    logger.info("COVERAGE total=%d scored=%d failed=%d covered=%s",
                total, scored, failed, scored + failed >= total)
    try:
        boto3.client("cloudwatch").put_metric_data(
            Namespace=CW_NAMESPACE,
            MetricData=[
                {"MetricName": "ImagesTotal", "Value": total},
                {"MetricName": "ImagesScored", "Value": scored},
                {"MetricName": "ImagesFailed", "Value": failed},
                {"MetricName": "ImagesUncovered", "Value": max(0, total - scored - failed)},
            ],
        )
    except Exception as e:  # noqa: BLE001 — metrics are best-effort
        logger.warning("put_metric_data failed: %s", e)


def invoke_continuation(run_id: str, run_date: str, continuation: int,
                        selection: Optional[dict] = None) -> None:
    """Queue exactly one checkpointed continuation before the time limit.

    The row selection (ad-hoc query / date range) rides along, so the
    continuation resumes the same set of rows rather than falling back to the
    default SOURCE_QUERY.
    """
    payload = {"run_id": run_id, "run_date": run_date, "continuation": continuation}
    payload.update(selection or {})
    boto3.client("lambda").invoke(
        FunctionName=LAMBDA_FUNCTION_NAME, InvocationType="Event",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    logger.warning("Queued continuation #%d for run %s", continuation, run_id)


def _remaining_ms(context: Any) -> int:
    try:
        return int(context.get_remaining_time_in_millis())
    except Exception:  # noqa: BLE001 — local/tests have no real context
        return 10 ** 9


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #

def handler(event: Any, context: Any) -> dict:
    """Entry point. Routes per-trip events to the single-trip path, everything
    else (EventBridge/manual/backfill) to the whole-dataset batch path."""
    event = _normalise_event(event)
    if event.get("migrate"):
        return handle_migrate(event)
    if event.get("warmup"):
        return handle_warmup(event)
    if event.get("trip_id") not in (None, ""):
        return handle_single_trip(event, context)
    return handle_batch(event, context)


# --------------------------------------------------------------------------- #
# Schema migration / teardown
# --------------------------------------------------------------------------- #

def handle_migrate(event: dict) -> dict:
    """Apply (or, on explicit request, drop) the pod_scores schema.

    CI runs this right after the stack is created: the database usually sits in
    a private subnet that CircleCI cannot reach, but this function is already
    inside the VPC, so it is the natural place to bootstrap the schema.

        {"migrate": true}                            -> apply schema.sql (idempotent)
        {"migrate": "drop", "confirm": "<fn name>"}  -> DROP the table and view

    schema.sql is written with IF NOT EXISTS / CREATE OR REPLACE throughout, so
    applying it repeatedly is a no-op. The drop is deliberately awkward: it must
    name the exact function it is aimed at, so a payload copied from one
    environment cannot destroy another.
    """
    mode = event.get("migrate")
    destructive = isinstance(mode, str) and mode.lower() in ("drop", "teardown")

    if not PG_HOST or not PG_PASSWORD:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing Postgres config"})}

    if destructive:
        confirm = str(event.get("confirm") or "")
        if confirm != LAMBDA_FUNCTION_NAME or not LAMBDA_FUNCTION_NAME:
            logger.error("Refused destructive migrate: confirm=%r != function %r",
                         confirm, LAMBDA_FUNCTION_NAME)
            return {"statusCode": 400, "body": json.dumps({
                "error": "Destructive migrate requires confirm == the function's own name",
                "status": "refused",
            })}
        statements = list(TEARDOWN_OBJECTS)
        logger.warning("DESTRUCTIVE MIGRATE: dropping pod_scores on %s", LAMBDA_FUNCTION_NAME)
    else:
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
                statements = [fh.read()]
        except OSError as e:
            logger.error("Cannot read schema at %s: %s", SCHEMA_PATH, e)
            return {"statusCode": 500, "body": json.dumps({
                "error": f"Schema not found at {SCHEMA_PATH}", "status": "failed"})}

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        logger.error("Migrate failed: %s", e)
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "status": "failed"})}
    finally:
        release_db_connection(conn)

    status = "dropped" if destructive else "applied"
    logger.info("MIGRATE %s", status.upper())
    return {"statusCode": 200, "body": json.dumps({
        "status": status, "destructive": destructive,
        "statements": len(statements),
    })}


# --------------------------------------------------------------------------- #
# Warm pool
# --------------------------------------------------------------------------- #

def handle_warmup(event: dict) -> dict:
    """Keep a small pool of containers hot so rider-triggered scoring is instant.

    A cold container pays for the image pull, the torch import and the
    checkpoint load — tens of seconds. This ping does exactly that work and
    returns, leaving the container warm.

    Lambda routes concurrent invocations to *different* containers, so warming
    N of them means self-invoking N-1 times and holding each briefly: without
    the hold they would all be served by the same container. The fan-out
    invocations carry fanout=0 so they never recurse.
    """
    get_model()                 # the expensive bit
    build_session()
    warmed = 1
    fanout = int(event.get("fanout", WARM_FANOUT))
    try:
        get_db_connection()
    except Exception as e:  # noqa: BLE001 — a warm container without the DB is still useful
        logger.warning("warmup: DB dial failed: %s", e)

    if fanout > 1 and LAMBDA_FUNCTION_NAME:
        client = boto3.client("lambda")
        for _ in range(fanout - 1):
            try:
                client.invoke(
                    FunctionName=LAMBDA_FUNCTION_NAME, InvocationType="Event",
                    Payload=json.dumps({"warmup": True, "fanout": 0}).encode("utf-8"),
                )
                warmed += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("warmup fan-out failed: %s", e)
                break
        # Occupy this container so the siblings land on their own.
        time.sleep(float(os.environ.get("WARM_HOLD_SECONDS", "1.5")))

    logger.info("WARMUP ok containers_requested=%d", warmed)
    return {"statusCode": 200, "body": json.dumps({"status": "warm", "warmed": warmed})}


def _normalise_event(event: Any) -> dict:
    """Accept a dict, a JSON string, or an SQS/EventBridge-style envelope."""
    if event is None:
        return {}
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except Exception:  # noqa: BLE001
            return {}
    if not isinstance(event, dict):
        return {}
    # EventBridge "detail" / SQS single-record envelopes.
    if "trip_id" not in event and isinstance(event.get("detail"), dict):
        return event["detail"]
    records = event.get("Records")
    if "trip_id" not in event and isinstance(records, list) and records:
        body = records[0].get("body")
        if isinstance(body, str):
            try:
                return json.loads(body)
            except Exception:  # noqa: BLE001
                return {}
    return event


# --------------------------------------------------------------------------- #
# Single-trip handler (rider-triggered POD verification)
# --------------------------------------------------------------------------- #

def handle_single_trip(event: dict, context: Any) -> dict:
    """Score the POD images of exactly one trip.

    Deliberately simple compared to the batch path: a trip has a handful of
    images, so there is no window loop and no continuation — one invocation
    always finishes the trip.

    Re-requests are cheap: links this trip has already scored are skipped, and
    only links that are new or changed since the last request are downloaded.
    """
    t_start = time.time()
    trip_id = str(event["trip_id"]).strip()
    run_date = event.get("run_date") or date.today().isoformat()
    run_id = event.get("run_id") or f"trip-{trip_id}-{uuid.uuid4().hex[:8]}"

    if not PG_HOST or not PG_PASSWORD:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing Postgres config"})}

    source = "trip_query"
    rows: list[dict] = []
    if TRIP_QUERY:
        try:
            raw = fetch_trip_pod_data(trip_id)
            if not raw.empty:
                rows = expand_pod_links(raw).to_dict("records")
        except Exception as e:  # noqa: BLE001 — fall back to the event payload
            logger.warning("TRIP_QUERY failed for trip %s: %s", trip_id, e)

    if not rows:
        rows = rows_from_event(event, trip_id)
        source = "event_payload"

    if not rows:
        logger.warning("No POD links for trip %s (run=%s)", trip_id, run_id)
        return {"statusCode": 200, "body": json.dumps({
            "message": "No POD links for trip", "trip_id": trip_id, "run_id": run_id,
            "status": "no_data",
        })}

    # Trip id from the event always wins, so scores stay joinable to the trip
    # that triggered them even if the source row carries a different value.
    for r in rows:
        r["trip_id"] = trip_id

    model, device = get_model()
    session = build_session(TRIP_MAX_DOWNLOAD_WORKERS)
    conn = get_db_connection()
    try:
        # A rider re-raising the request usually changed only some of the
        # photos. Score what is new; leave what is unchanged alone.
        already = load_scored_links(conn, trip_id, [r["pod_link"] for r in rows])
        pending = [r for r in rows if r["pod_link"] not in already]

        if not pending:
            logger.info("trip %s: all %d POD links already scored (run=%s)",
                        trip_id, len(rows), run_id)
            return {"statusCode": 200, "body": json.dumps({
                "run_id": run_id, "trip_id": trip_id, "mode": "single_trip",
                "source": source, "total_images": len(rows),
                "skipped_already_scored": len(already),
                "scored": 0, "failed": 0, "status": "already_scored",
                "invocation_duration_s": round(time.time() - t_start, 3),
            })}

        successes, failures = download_window(session, pending, TRIP_MAX_DOWNLOAD_WORKERS)
        scored = score_prepared(model, device, successes)
        upsert_results(conn, scored + failures, run_date)
    finally:
        release_db_connection(conn)

    summary = {
        "run_id": run_id, "run_date": run_date, "trip_id": trip_id,
        "mode": "single_trip", "source": source,
        "total_images": len(rows),
        "skipped_already_scored": len(already),
        "scored": len(scored), "failed": len(failures),
        "status": "complete",
        "invocation_duration_s": round(time.time() - t_start, 3),
    }
    logger.info("Trip scoring COMPLETE: %s", json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}


# --------------------------------------------------------------------------- #
# Batch handler (backfill / manual whole-dataset re-run)
# --------------------------------------------------------------------------- #

def handle_batch(event: dict, context: Any) -> dict:
    t_start = time.time()
    run_date = event.get("run_date") or date.today().isoformat()
    run_id = event.get("run_id") or f"{run_date}_{uuid.uuid4().hex[:8]}"
    continuation = int(event.get("continuation", 0))

    if not PG_HOST or not PG_PASSWORD:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing Postgres config"})}

    # What to score: ad-hoc SQL from the event > date range > the default
    # SOURCE_QUERY. A continuation carries the same selection forward so a
    # resumed run keeps scoring the same set of rows.
    try:
        sql, params, source = resolve_batch_source(event)
    except ValueError as e:
        logger.error("Bad batch request: %s", e)
        return {"statusCode": 400, "body": json.dumps({"error": str(e), "run_id": run_id})}
    logger.info("run=%s batch source=%s params=%s", run_id, source, params)

    # Read + expand the selected POD rows (cheap, one shot).
    raw = fetch_pod_data(sql, params)
    if raw.empty:
        return {"statusCode": 200, "body": json.dumps({"message": "No data", "run_id": run_id})}
    expanded = expand_pod_links(raw)
    total = len(expanded)
    if total == 0:
        return {"statusCode": 200, "body": json.dumps({"message": "No POD links", "run_id": run_id})}

    model, device = get_model()
    session = build_session()
    conn = get_db_connection()
    scored_total = failed_total = 0
    hit_time_limit = False
    try:
        done = load_done_keys(conn, run_date)
        pending = [r for r in expanded.to_dict("records")
                   if (r["awb"], r["pod_link"]) not in done]
        logger.info("run=%s total=%d done=%d pending=%d continuation=%d",
                    run_id, total, len(done), len(pending), continuation)

        for w in range(0, len(pending), WINDOW_SIZE):
            if _remaining_ms(context) < CONTINUATION_SAFETY_MS:
                hit_time_limit = True
                break
            window = pending[w:w + WINDOW_SIZE]
            successes, failures = download_window(session, window)
            scored = score_prepared(model, device, successes)
            upsert_results(conn, scored + failures, run_date)
            scored_total += len(scored)
            failed_total += len(failures)
            del successes, failures, scored
            logger.info("window %d-%d done (scored=%d failed=%d)",
                        w, w + len(window), scored_total, failed_total)
    finally:
        release_db_connection(conn)

    if hit_time_limit and continuation < MAX_CONTINUATIONS:
        selection = {k: event[k] for k in ("query", "start_date", "end_date") if event.get(k)}
        invoke_continuation(run_id, run_date, continuation + 1, selection)
        status = "continuing"
    else:
        # Done, or out of continuations: emit coverage so an alarm can fire if
        # scored + failed < total (i.e. we gave up with work remaining).
        emit_coverage(total, len(done) + scored_total, failed_total)
        status = "incomplete" if hit_time_limit else "complete"

    summary = {
        "run_id": run_id, "run_date": run_date, "mode": "batch", "source": source,
        "total_images": total,
        "scored_this_invocation": scored_total, "failed_this_invocation": failed_total,
        "already_done_at_start": len(done),
        "status": status,
        "continuation": continuation,
        "invocation_duration_s": round(time.time() - t_start, 3),
    }
    logger.info("Pipeline %s: %s", status.upper(), json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}
