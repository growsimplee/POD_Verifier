"""
POD scoring Lambda — event-driven per trip, with a batch mode for backfills.

The scorer owns no data. It asks sarathy what to score and posts back what it found;
sarathy is the only writer to its own tables. There is no database driver here.

PRIMARY (event-driven, one trip):
    Sarathy invokes this function asynchronously when a rider raises a POD
    verification request:

        {"trip_id": 12345, "pod_links": ["https://..."], "awb": "..."}   # links optional

    Links come from the event when it carries them — that is the trip as it was at
    the instant of the request — and from sarathy otherwise. Images already scored
    are skipped. Results are posted back in one call.

SECONDARY (batch, from Sarathy's admin API or a manual invoke):
    An event with no trip_id scores a date range, paging through sarathy:

        {"start_date": "2026-09-01", "end_date": "2026-09-09"}
        {}                                    -> today

WARMUP:
    {"warmup": true} loads the checkpoint and returns, keeping a small pool of
    containers hot so rider-triggered scoring does not pay a cold start.

Design guarantees:
  * one writer      — every row reaches Postgres through sarathy, never from here.
  * warm reuse      — model and HTTP pools live for the life of the container.
  * no rework       — links sarathy reports as already scored are not downloaded.
  * bounded memory  — only one WINDOW_SIZE of images is held at a time.
  * every input gets an outcome — a failed download is recorded, never dropped.
  * time-safe       — a clock check hands off before the 15-minute wall.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
import uuid
from datetime import date
from typing import Any, Optional

import boto3
import cv2
import numpy as np
import requests
import torch

from sarathy_client import SarathyClient, SarathyError
from src.model import ATTRIBUTE_NAMES, ATTRIBUTE_WEIGHTS, MultiHeadEfficientNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Configuration (all from environment; set by CloudFormation)
# --------------------------------------------------------------------------- #

# Sarathy's internal base URL, e.g. http://<internal-nlb>:8080. Reachable only
# from inside the VPC.
SARATHY_BASE_URL = os.environ.get("SARATHY_BASE_URL", "")
SARATHY_TIMEOUT = int(os.environ.get("SARATHY_TIMEOUT", "15"))
SARATHY_RETRIES = int(os.environ.get("SARATHY_RETRIES", "3"))
SARATHY_PAGE_SIZE = int(os.environ.get("SARATHY_PAGE_SIZE", "500"))

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

# Warm-pool: how many containers a warmup ping keeps alive (1 = just this one).
WARM_FANOUT = int(os.environ.get("WARM_FANOUT", "3"))

# Resilience.
CONTINUATION_SAFETY_MS = int(os.environ.get("CONTINUATION_SAFETY_MS", "90000"))
MAX_CONTINUATIONS = int(os.environ.get("MAX_CONTINUATIONS", "5"))

LAMBDA_FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
CW_NAMESPACE = os.environ.get("CW_NAMESPACE", "PODPipeline")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Warm-container state
#
# Built once per container and reused by every later invocation that lands on it:
# the EfficientNet checkpoint (seconds to load), the image-download connection
# pool, and the sarathy client's own pool. Under load Lambda runs several
# containers side by side and each pays this once.
# --------------------------------------------------------------------------- #

_model: Optional[MultiHeadEfficientNet] = None
_device: Optional[torch.device] = None
_session: Optional[requests.Session] = None
_sarathy: Optional[SarathyClient] = None


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


def get_sarathy() -> SarathyClient:
    global _sarathy
    if _sarathy is None:
        _sarathy = SarathyClient(SARATHY_BASE_URL, timeout=SARATHY_TIMEOUT,
                                 retries=SARATHY_RETRIES, page_size=SARATHY_PAGE_SIZE)
    return _sarathy


def build_session(max_workers: int = MAX_DOWNLOAD_WORKERS) -> requests.Session:
    """HTTP session for image downloads, cached for the life of the container."""
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


# --------------------------------------------------------------------------- #
# Rows carried in the trigger event
# --------------------------------------------------------------------------- #

def rows_from_event(event: dict, trip_id: str) -> list[dict]:
    """POD links carried in the trigger event itself.

    Sarathy sends these from the trip row at the moment the rider raised the
    request, so they are the freshest view there is.
    """
    links = event.get("pod_links") or event.get("pod") or []
    if isinstance(links, str):
        links = [l.strip() for l in links.split(",")]
    awb = str(event.get("awb") or "").strip() or f"TRIP-{trip_id}"
    seen, rows = set(), []
    for link in (str(l).strip() for l in links):
        if link.startswith("http") and link not in seen:
            seen.add(link)
            rows.append({"awb": awb, "trip_id": str(trip_id),
                         "pod_link": link, "already_scored": False})
    return rows


# --------------------------------------------------------------------------- #
# Download + preprocess + score
# --------------------------------------------------------------------------- #

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


def score_and_record(rows: list[dict], run_date: str, max_workers: int) -> tuple[int, int]:
    """Download, score and post one batch back to sarathy. Returns (scored, failed)."""
    if not rows:
        return 0, 0
    model, device = get_model()
    session = build_session(max_workers)
    successes, failures = download_window(session, rows, max_workers)
    scored = score_prepared(model, device, successes)
    get_sarathy().write_scores(run_date, scored + failures)
    return len(scored), len(failures)


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
    """Queue exactly one checkpointed continuation before the time limit."""
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
# Dispatch
# --------------------------------------------------------------------------- #

def handler(event: Any, context: Any) -> dict:
    """Entry point. Routes per-trip events to the single-trip path, everything
    else (manual/backfill) to the date-range batch path."""
    event = _normalise_event(event)
    if event.get("warmup"):
        return handle_warmup(event)
    if event.get("trip_id") not in (None, ""):
        return handle_single_trip(event, context)
    return handle_batch(event, context)


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
# Warm pool
# --------------------------------------------------------------------------- #

def handle_warmup(event: dict) -> dict:
    """Keep a small pool of containers hot so rider-triggered scoring is instant.

    A cold container pays for the image pull, the torch import and the checkpoint
    load — tens of seconds. This ping does exactly that work and returns.

    Lambda routes concurrent invocations to *different* containers, so warming N
    of them means self-invoking N-1 times and holding each briefly: without the
    hold they would all be served by the same container.
    """
    get_model()                 # the expensive bit
    build_session()
    warmed = 1
    fanout = int(event.get("fanout", WARM_FANOUT))

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
        time.sleep(float(os.environ.get("WARM_HOLD_SECONDS", "1.5")))

    logger.info("WARMUP ok containers_requested=%d", warmed)
    return {"statusCode": 200, "body": json.dumps({"status": "warm", "warmed": warmed})}


# --------------------------------------------------------------------------- #
# Single trip
# --------------------------------------------------------------------------- #

def handle_single_trip(event: dict, context: Any) -> dict:
    """Score the POD images of exactly one trip.

    A trip has a handful of images, so there is no window loop and no
    continuation — one invocation always finishes it.
    """
    t_start = time.time()
    trip_id = str(event["trip_id"]).strip()
    run_date = event.get("run_date") or date.today().isoformat()
    run_id = event.get("run_id") or f"trip-{trip_id}-{uuid.uuid4().hex[:8]}"

    if not SARATHY_BASE_URL:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing SARATHY_BASE_URL"})}

    # Links: the event carries them as they were at the instant the rider raised
    # the request, so it wins. Sarathy is asked when the event has none — a
    # console invoke with only a trip_id, say — and also for the AWB, which the
    # event does not carry and which a TRIP-<id> placeholder cannot substitute
    # for downstream.
    event_awb = str(event.get("awb") or "").strip()
    rows = rows_from_event(event, trip_id)
    source = "event_payload"
    api_rows: list[dict] = []

    if not rows or not event_awb:
        try:
            api_rows = get_sarathy().trip_rows(trip_id)
        except SarathyError as e:
            if not rows:
                logger.error("sarathy lookup failed for trip %s: %s", trip_id, e)
                return {"statusCode": 502, "body": json.dumps({
                    "error": str(e), "trip_id": trip_id, "run_id": run_id, "status": "failed"})}
            logger.warning("sarathy lookup failed for trip %s: %s", trip_id, e)

    if api_rows:
        if not rows:
            rows, source = api_rows, "sarathy"
        else:
            # Keep the event's links; take the AWB and the already-scored flags.
            scored_links = {r["pod_link"] for r in api_rows if r.get("already_scored")}
            for r in rows:
                if not event_awb:
                    r["awb"] = api_rows[0]["awb"]
                r["already_scored"] = r["pod_link"] in scored_links

    if not rows:
        logger.warning("No POD links for trip %s (run=%s)", trip_id, run_id)
        return {"statusCode": 200, "body": json.dumps({
            "message": "No POD links for trip", "trip_id": trip_id, "run_id": run_id,
            "status": "no_data",
        })}

    for r in rows:
        r["trip_id"] = trip_id

    already = [r for r in rows if r.get("already_scored")]
    pending = [r for r in rows if not r.get("already_scored")]

    if not pending:
        logger.info("trip %s: all %d POD links already scored (run=%s)",
                    trip_id, len(rows), run_id)
        return {"statusCode": 200, "body": json.dumps({
            "run_id": run_id, "trip_id": trip_id, "mode": "single_trip", "source": source,
            "total_images": len(rows), "skipped_already_scored": len(already),
            "scored": 0, "failed": 0, "status": "already_scored",
            "invocation_duration_s": round(time.time() - t_start, 3),
        })}

    scored, failed = score_and_record(pending, run_date, TRIP_MAX_DOWNLOAD_WORKERS)

    summary = {
        "run_id": run_id, "run_date": run_date, "trip_id": trip_id,
        "mode": "single_trip", "source": source,
        "total_images": len(rows),
        "skipped_already_scored": len(already),
        "scored": scored, "failed": failed,
        "status": "complete",
        "invocation_duration_s": round(time.time() - t_start, 3),
    }
    logger.info("Trip scoring COMPLETE: %s", json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}


# --------------------------------------------------------------------------- #
# Batch (backfill over a date range)
# --------------------------------------------------------------------------- #

def resolve_range(event: dict) -> tuple[str, str]:
    """The date range to score. An empty event means today.

    Raises ValueError on a malformed range so the caller can answer 400.
    """
    start = event.get("start_date") or event.get("end_date") or date.today().isoformat()
    end = event.get("end_date") or start
    for label, value in (("start_date", start), ("end_date", end)):
        try:
            date.fromisoformat(str(value))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{label} must be ISO YYYY-MM-DD") from exc
    if str(start) > str(end):
        raise ValueError("start_date must not be after end_date")
    return str(start), str(end)


def handle_batch(event: dict, context: Any) -> dict:
    t_start = time.time()
    run_date = event.get("run_date") or date.today().isoformat()
    run_id = event.get("run_id") or f"{run_date}_{uuid.uuid4().hex[:8]}"
    continuation = int(event.get("continuation", 0))

    if not SARATHY_BASE_URL:
        return {"statusCode": 500, "body": json.dumps({"error": "Missing SARATHY_BASE_URL"})}

    try:
        start_date, end_date = resolve_range(event)
    except ValueError as e:
        logger.error("Bad batch request: %s", e)
        return {"statusCode": 400, "body": json.dumps({"error": str(e), "run_id": run_id})}
    logger.info("run=%s batch range=%s..%s continuation=%d",
                run_id, start_date, end_date, continuation)

    total = skipped_total = scored_total = failed_total = 0
    hit_time_limit = False

    try:
        for page in get_sarathy().range_rows(start_date, end_date):
            total += len(page)
            # Sarathy flags what it has already scored, which is also what makes a
            # continuation resume: re-paging the range costs a query and skips the
            # work that is already done.
            pending = [r for r in page if not r.get("already_scored")]
            skipped_total += len(page) - len(pending)

            for w in range(0, len(pending), WINDOW_SIZE):
                if _remaining_ms(context) < CONTINUATION_SAFETY_MS:
                    hit_time_limit = True
                    break
                window = pending[w:w + WINDOW_SIZE]
                scored, failed = score_and_record(window, run_date, MAX_DOWNLOAD_WORKERS)
                scored_total += scored
                failed_total += failed
                logger.info("window %d-%d done (scored=%d failed=%d)",
                            w, w + len(window), scored_total, failed_total)
            if hit_time_limit:
                break
    except SarathyError as e:
        logger.error("run=%s aborted: %s", run_id, e)
        return {"statusCode": 502, "body": json.dumps({
            "error": str(e), "run_id": run_id, "status": "failed",
            "scored_this_invocation": scored_total, "failed_this_invocation": failed_total})}

    if hit_time_limit and continuation < MAX_CONTINUATIONS:
        invoke_continuation(run_id, run_date, continuation + 1,
                            {"start_date": start_date, "end_date": end_date})
        status = "continuing"
    else:
        emit_coverage(total, scored_total + skipped_total, failed_total)
        status = "incomplete" if hit_time_limit else "complete"

    summary = {
        "run_id": run_id, "run_date": run_date, "mode": "batch",
        "start_date": start_date, "end_date": end_date,
        "total_images": total,
        "skipped_already_scored": skipped_total,
        "scored_this_invocation": scored_total,
        "failed_this_invocation": failed_total,
        "status": status,
        "continuation": continuation,
        "invocation_duration_s": round(time.time() - t_start, 3),
    }
    logger.info("Pipeline %s: %s", status.upper(), json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}
