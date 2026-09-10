"""
Test suite for the single-invocation POD pipeline handler.

Design goals covered:
  * expand/dedup correctness
  * concurrent download partitions success vs failure, never drops an input
  * idempotent upsert SQL shape (ON CONFLICT) + failure rows recorded
  * resume-from-checkpoint skips already-scored rows
  * WHOLE-DATASET COVERAGE in one invocation with ample time (no timeout)
  * clock-aware continuation fires (and only fires) near the wall
  * preprocessing shape + normalization

Pipeline-logic tests mock the model and DB, so they run without torch or a
database. A separate torch-gated test exercises real inference wiring.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler as H  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers / fakes
# --------------------------------------------------------------------------- #

def _png_bytes(w=32, h=32):
    import cv2
    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class FakeResp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status


class FakeSession:
    """Returns a valid PNG for most URLs; configurable failures."""
    def __init__(self, fail_urls=None, http_status=None):
        self.fail_urls = fail_urls or set()
        self.http_status = http_status or {}

    def get(self, url, timeout=0):
        if url in self.fail_urls:
            raise ConnectionError("boom")
        if url in self.http_status:
            return FakeResp(_png_bytes(), self.http_status[url])
        return FakeResp(_png_bytes(), 200)


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._rows = []
        self.description = [("col",)]
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def execute(self, sql, params=None):
        self.conn._last_sql = sql
        self.conn.executed.append((sql, params))
        if sql.strip().upper().startswith("SELECT AWB"):
            # a source query — mirror fetch_pod_data's column contract
            self.description = [("awb",), ("trip_id",), ("pod",)]
            self._rows = list(self.conn.done_rows)
        elif "SELECT pod_link FROM pod_scores" in sql:
            # load_scored_links: only links this trip already scored
            links = set(params["links"]) if params else set()
            self._rows = [(l,) for l in self.conn.scored_links if l in links]
        elif sql.strip() == "SELECT 1":
            self._rows = [(1,)]
        else:
            self._rows = list(self.conn.done_rows)
    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, done_rows=None, scored_links=()):
        self.done_rows = done_rows or []
        self.scored_links = set(scored_links)
        self.upserted = []
        self.executed = []
        self.committed = 0
        self.rolled_back = 0
        self.closed = 0
        self.autocommit = True
        self.autocommit_history = []
    def cursor(self):
        return FakeCursor(self)
    def commit(self):
        self.committed += 1
    def rollback(self):
        self.rolled_back += 1
    def close(self):
        self.closed = 1


class FakeContext:
    def __init__(self, remaining_ms):
        self._r = remaining_ms
    def get_remaining_time_in_millis(self):
        return self._r


def _fake_score_prepared(model, device, successes):
    out = []
    for s in successes:
        out.append({
            "awb": s["awb"], "trip_id": s["trip_id"], "pod_link": s["pod_link"],
            "status": "scored", "failure_reason": None, "pod_score": 0.9,
            "context_valid_prob": 0.9, "package_visible_prob": 0.9,
            "label_readable_prob": 0.9, "image_clarity_prob": 0.9,
        })
    return out


@pytest.fixture
def wire(monkeypatch):
    conn = FakeConn()
    # Warm-container caches are module globals — clear them so tests are isolated.
    monkeypatch.setattr(H, "_model", None, raising=False)
    monkeypatch.setattr(H, "_session", None, raising=False)
    monkeypatch.setattr(H, "_conn", None, raising=False)
    monkeypatch.setattr(H, "get_model", lambda: (object(), "cpu"))
    monkeypatch.setattr(H, "get_db_connection", lambda: conn)
    monkeypatch.setattr(H, "release_db_connection", lambda c: None)
    monkeypatch.setattr(H, "score_prepared", _fake_score_prepared)
    monkeypatch.setattr(H, "emit_coverage", lambda *a, **k: None)

    def _capture_upsert(c, rows, run_date):
        c.upserted.extend(rows)
        return len(rows)
    monkeypatch.setattr(H, "upsert_results", _capture_upsert)
    monkeypatch.setattr(H, "SOURCE_QUERY", "SELECT 1", raising=False)
    monkeypatch.setattr(H, "PG_HOST", "db", raising=False)
    monkeypatch.setattr(H, "PG_PASSWORD", "pw", raising=False)
    return conn


def _mb_df(n):
    return pd.DataFrame([
        {"AWB": f"AWB{i}", "Trip Id": f"T{i}", "POD": f"http://img/{i}.png"}
        for i in range(n)
    ])


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #

def test_expand_dedup_and_filter():
    df = pd.DataFrame([
        {"AWB": "A1", "Trip Id": "T1", "POD": "http://a/1.png, http://a/2.png"},
        {"AWB": "A1", "Trip Id": "T1", "POD": "http://a/1.png"},
        {"AWB": "A2", "Trip Id": "T2", "POD": "not_a_url, http://a/3.png"},
        {"AWB": "A3", "Trip Id": "T3", "POD": ""},
    ])
    out = H.expand_pod_links(df)
    assert set(out["pod_link"]) == {"http://a/1.png", "http://a/2.png", "http://a/3.png"}
    assert len(out) == 3


def test_preprocess_shape_and_normalization():
    img = (np.random.rand(50, 40, 3) * 255).astype(np.uint8)
    chw = H.preprocess_image(img, size=224, normalize=True)
    assert chw.shape == (3, 224, 224)
    assert chw.dtype == np.float32
    raw = H.preprocess_image(img, size=224, normalize=False)
    assert raw.min() >= 0.0 and raw.max() <= 1.0
    assert chw.min() < 0.0


def test_download_and_prepare_outcomes():
    s = FakeSession(fail_urls={"http://img/x.png"}, http_status={"http://img/y.png": 404})
    ok = H.download_and_prepare(s, {"awb": "a", "trip_id": "t", "pod_link": "http://img/ok.png"})
    assert "chw" in ok and "status" not in ok   # success = prepared tensor, no status yet
    exc = H.download_and_prepare(s, {"awb": "a", "trip_id": "t", "pod_link": "http://img/x.png"})
    assert exc["status"] == "download_failed" and exc["failure_reason"] == "ConnectionError"
    http = H.download_and_prepare(s, {"awb": "a", "trip_id": "t", "pod_link": "http://img/y.png"})
    assert http["status"] == "download_failed" and http["failure_reason"] == "http_404"


def test_download_window_partitions_all_inputs():
    rows = [{"awb": f"a{i}", "trip_id": "t", "pod_link": f"http://img/{i}.png"} for i in range(20)]
    s = FakeSession(fail_urls={"http://img/3.png", "http://img/9.png"})
    succ, fail = H.download_window(s, rows)
    assert len(succ) + len(fail) == 20
    assert {f["pod_link"] for f in fail} == {"http://img/3.png", "http://img/9.png"}


def test_upsert_sql_is_idempotent(monkeypatch):
    captured = {}
    def fake_execute_values(cur, sql, tuples, page_size=100):
        captured["sql"] = sql
        captured["tuples"] = tuples
    monkeypatch.setattr(H.psycopg2.extras, "execute_values", fake_execute_values)
    conn = FakeConn()
    rows = [{"awb": "a", "trip_id": "t", "pod_link": "u", "status": "scored",
             "pod_score": 0.8, "context_valid_prob": 0.1, "package_visible_prob": 0.2,
             "label_readable_prob": 0.3, "image_clarity_prob": 0.4}]
    n = H.upsert_results(conn, rows, "2026-07-05")
    assert n == 1
    assert "ON CONFLICT (awb, pod_link, run_date) DO UPDATE" in captured["sql"]
    assert conn.committed == 1


def test_load_done_keys():
    conn = FakeConn(done_rows=[("A1", "u1"), ("A2", "u2")])
    assert H.load_done_keys(conn, "2026-07-05") == {("A1", "u1"), ("A2", "u2")}


# --------------------------------------------------------------------------- #
# Handler-level: coverage, resume, continuation
# --------------------------------------------------------------------------- #

def test_handler_covers_whole_dataset_one_invocation(wire, monkeypatch):
    N = 2500
    monkeypatch.setattr(H, "fetch_pod_data", lambda sql=None, params=None: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(H, "WINDOW_SIZE", 800, raising=False)

    resp = H.handler({}, FakeContext(600_000))
    body = json.loads(resp["body"])
    assert body["status"] == "complete"
    assert body["scored_this_invocation"] == N
    assert body["failed_this_invocation"] == 0
    assert len(wire.upserted) == N
    assert body["continuation"] == 0


def test_handler_records_failures_as_outcomes(wire, monkeypatch):
    N = 100
    fail = {f"http://img/{i}.png" for i in (5, 10, 42)}
    monkeypatch.setattr(H, "fetch_pod_data", lambda sql=None, params=None: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda: FakeSession(fail_urls=fail))
    resp = H.handler({}, FakeContext(600_000))
    body = json.loads(resp["body"])
    assert body["scored_this_invocation"] == 97
    assert body["failed_this_invocation"] == 3
    assert body["scored_this_invocation"] + body["failed_this_invocation"] == N
    assert len([r for r in wire.upserted if r["status"] == "download_failed"]) == 3


def test_handler_resume_skips_done(wire, monkeypatch):
    N = 50
    monkeypatch.setattr(H, "fetch_pod_data", lambda sql=None, params=None: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    done = {(f"AWB{i}", f"http://img/{i}.png") for i in range(20)}
    monkeypatch.setattr(H, "load_done_keys", lambda c, d: done)
    resp = H.handler({}, FakeContext(600_000))
    body = json.loads(resp["body"])
    assert body["already_done_at_start"] == 20
    assert body["scored_this_invocation"] == 30
    assert len(wire.upserted) == 30


def test_handler_continuation_near_wall(wire, monkeypatch):
    N = 3000
    monkeypatch.setattr(H, "fetch_pod_data", lambda sql=None, params=None: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(H, "WINDOW_SIZE", 500, raising=False)
    monkeypatch.setattr(H, "CONTINUATION_SAFETY_MS", 90_000, raising=False)
    called = {}
    monkeypatch.setattr(H, "invoke_continuation",
                        lambda rid, rd, c, sel=None: called.update(run_id=rid, cont=c))

    class Ctx:
        def __init__(self):
            self.calls = 0
        def get_remaining_time_in_millis(self):
            self.calls += 1
            return 600_000 if self.calls == 1 else 10_000

    resp = H.handler({}, Ctx())
    body = json.loads(resp["body"])
    assert body["status"] == "continuing"
    assert called["cont"] == 1
    assert body["scored_this_invocation"] == 500


# --------------------------------------------------------------------------- #
# Torch-gated: real inference wiring
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch", reason="torch not installed")

def test_score_prepared_real_math():
    import torch as T
    from src.model import ATTRIBUTE_NAMES

    class FakeModel:
        def __call__(self, batch):
            b = batch.shape[0]
            return {n: T.zeros(b) for n in ATTRIBUTE_NAMES}

    successes = [{"awb": "a", "trip_id": "t", "pod_link": "u",
                  "chw": np.zeros((3, 224, 224), dtype=np.float32)} for _ in range(3)]
    out = H.score_prepared(FakeModel(), T.device("cpu"), successes)
    assert len(out) == 3
    assert abs(out[0]["pod_score"] - 0.5) < 1e-6
    assert abs(out[0]["context_valid_prob"] - 0.5) < 1e-6


# --------------------------------------------------------------------------- #
# Single-trip (event-driven) mode
# --------------------------------------------------------------------------- #

def _trip_df(trip_id="T99", n=2):
    return pd.DataFrame([
        {"awb": f"AWB{i}", "trip_id": trip_id,
         "pod": f"http://img/{trip_id}_{i}.png"}
        for i in range(n)
    ])


def test_normalise_event_unwraps_envelopes():
    assert H._normalise_event({"trip_id": 5})["trip_id"] == 5
    assert H._normalise_event('{"trip_id": 5}')["trip_id"] == 5
    assert H._normalise_event({"detail": {"trip_id": 5}})["trip_id"] == 5
    assert H._normalise_event(
        {"Records": [{"body": json.dumps({"trip_id": 5})}]})["trip_id"] == 5
    assert H._normalise_event(None) == {}
    assert H._normalise_event({}) == {}


def test_trip_event_routes_to_single_trip(wire, monkeypatch):
    called = {}
    monkeypatch.setattr(H, "handle_single_trip",
                        lambda e, c: called.setdefault("trip", e["trip_id"]))
    monkeypatch.setattr(H, "handle_batch",
                        lambda e, c: called.setdefault("batch", True))
    H.handler({"trip_id": 4242}, FakeContext(900_000))
    assert called == {"trip": 4242}


def test_empty_event_still_routes_to_batch(wire, monkeypatch):
    called = {}
    monkeypatch.setattr(H, "handle_batch", lambda e, c: called.setdefault("batch", True))
    H.handler({}, FakeContext(900_000))
    assert called == {"batch": True}


def test_single_trip_scores_only_that_trip(wire, monkeypatch):
    """No links in the event — e.g. a console invoke — so the fallback resolves them."""
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: _trip_df("T99", 3))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    resp = H.handler({"trip_id": "T99"}, FakeContext(900_000))
    body = json.loads(resp["body"])

    assert resp["statusCode"] == 200
    assert body["mode"] == "single_trip"
    assert body["source"] == "trip_query"
    assert body["trip_id"] == "T99"
    assert body["total_images"] == 3
    assert body["scored"] == 3 and body["failed"] == 0
    # every upserted row is bound to the triggering trip — no other trip touched
    assert {r["trip_id"] for r in wire.upserted} == {"T99"}
    assert len(wire.upserted) == 3


def test_event_links_win_over_the_derived_table(wire, monkeypatch):
    """kaptaan lags the trip table, so a stale link must not beat a fresh one.

    Only reproducible on prod, where the derived pipeline actually runs: the
    rider replaces a photo, re-raises the request, and the old link is still
    what kaptaan holds.
    """
    called = {"n": 0}

    def _stale(tid):
        called["n"] += 1
        return pd.DataFrame([
            {"awb": "A1", "trip_id": tid, "pod": "http://img/stale.png"},
        ])

    monkeypatch.setattr(H, "fetch_trip_pod_data", _stale)
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    body = json.loads(H.handler({
        "trip_id": 90, "pod_links": ["http://img/fresh.png"],
    }, FakeContext(900_000))["body"])

    assert body["source"] == "event_payload"
    # the event's link is scored, not the table's
    assert [r["pod_link"] for r in wire.upserted] == ["http://img/fresh.png"]
    # ...but the AWB is still taken from the table, since the event has none
    assert [r["awb"] for r in wire.upserted] == ["A1"]
    assert called["n"] == 1


def test_single_trip_falls_back_to_event_links(wire, monkeypatch):
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: pd.DataFrame())
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    body = json.loads(H.handler({
        "trip_id": 77,
        "awb": "AWB-77",
        "pod_links": ["http://img/a.png", "http://img/a.png", "http://img/b.png", "junk"],
    }, FakeContext(900_000))["body"])

    assert body["source"] == "event_payload"
    assert body["total_images"] == 2          # deduped, non-http dropped
    assert {r["awb"] for r in wire.upserted} == {"AWB-77"}


def test_single_trip_falls_back_when_trip_query_errors(wire, monkeypatch):
    def _boom(tid):
        raise RuntimeError("db down")
    monkeypatch.setattr(H, "fetch_trip_pod_data", _boom)
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    body = json.loads(H.handler({
        "trip_id": 77, "pod_links": ["http://img/a.png"],
    }, FakeContext(900_000))["body"])

    assert body["source"] == "event_payload"
    assert body["scored"] == 1


def test_single_trip_without_awb_uses_trip_placeholder(wire, monkeypatch):
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: pd.DataFrame())
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    H.handler({"trip_id": 77, "pod_links": ["http://img/a.png"]}, FakeContext(900_000))
    assert {r["awb"] for r in wire.upserted} == {"TRIP-77"}


def test_single_trip_no_links_is_a_clean_no_op(wire, monkeypatch):
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: pd.DataFrame())
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    resp = H.handler({"trip_id": 77}, FakeContext(900_000))
    body = json.loads(resp["body"])

    assert resp["statusCode"] == 200
    assert body["status"] == "no_data"
    assert wire.upserted == []


def test_single_trip_records_download_failures(wire, monkeypatch):
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: _trip_df("T5", 2))
    monkeypatch.setattr(H, "build_session",
                        lambda *a, **k: FakeSession(fail_urls={"http://img/T5_1.png"}))

    body = json.loads(H.handler({"trip_id": "T5"}, FakeContext(900_000))["body"])

    assert body["scored"] == 1 and body["failed"] == 1
    assert len(wire.upserted) == 2            # every input still gets an outcome
    assert {r["status"] for r in wire.upserted} == {"scored", "download_failed"}


# --------------------------------------------------------------------------- #
# Re-requests for the same trip
# --------------------------------------------------------------------------- #

def test_repeat_request_skips_links_already_scored(wire, monkeypatch):
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: _trip_df("T7", 2))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    first = json.loads(H.handler({"trip_id": "T7"}, FakeContext(900_000))["body"])
    assert first["scored"] == 2
    # the DB now holds those two links as scored
    wire.scored_links |= {r["pod_link"] for r in wire.upserted}
    wire.upserted.clear()

    second = json.loads(H.handler({"trip_id": "T7"}, FakeContext(900_000))["body"])
    assert second["status"] == "already_scored"
    assert second["scored"] == 0
    assert second["skipped_already_scored"] == 2
    assert wire.upserted == []                # nothing re-downloaded, nothing re-written


def test_repeat_request_scores_only_the_changed_links(wire, monkeypatch):
    """The rider replaced one photo — score that one, leave the other alone."""
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: pd.DataFrame([
        {"awb": "A1", "trip_id": "T8", "pod": "http://img/old.png"},
        {"awb": "A2", "trip_id": "T8", "pod": "http://img/new.png"},
    ]))
    wire.scored_links = {"http://img/old.png"}

    body = json.loads(H.handler({"trip_id": "T8"}, FakeContext(900_000))["body"])

    assert body["total_images"] == 2
    assert body["skipped_already_scored"] == 1
    assert body["scored"] == 1
    assert [r["pod_link"] for r in wire.upserted] == ["http://img/new.png"]


def test_scored_links_lookup_is_scoped_to_the_trip(wire, monkeypatch):
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: _trip_df("T9", 1))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    H.handler({"trip_id": "T9"}, FakeContext(900_000))
    lookup = [(sql, prm) for sql, prm in wire.executed
              if "SELECT pod_link FROM pod_scores" in sql]
    assert lookup, "expected an already-scored lookup"
    sql, params = lookup[0]
    assert params["trip_id"] == "T9"
    assert "status = 'scored'" in sql
    assert "run_date >=" in sql               # bounded by RESCORE_LOOKBACK_DAYS


def test_rescore_lookback_zero_looks_back_forever(wire, monkeypatch):
    monkeypatch.setattr(H, "RESCORE_LOOKBACK_DAYS", 0, raising=False)
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: _trip_df("TA", 1))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    H.handler({"trip_id": "TA"}, FakeContext(900_000))
    sql = [s for s, _ in wire.executed if "SELECT pod_link FROM pod_scores" in s][0]
    assert "run_date >=" not in sql


# --------------------------------------------------------------------------- #
# Warm pool
# --------------------------------------------------------------------------- #

def test_warmup_loads_the_model_and_does_not_score(wire, monkeypatch):
    loaded = {"n": 0}
    monkeypatch.setattr(H, "get_model", lambda: (loaded.__setitem__("n", loaded["n"] + 1), (object(), "cpu"))[1])
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(H, "WARM_FANOUT", 1, raising=False)

    body = json.loads(H.handler({"warmup": True}, FakeContext(900_000))["body"])

    assert body["status"] == "warm"
    assert loaded["n"] == 1
    assert wire.upserted == []


def test_warmup_fans_out_to_sibling_containers(wire, monkeypatch):
    invokes = []

    class FakeLambda:
        def invoke(self, **kw):
            invokes.append(json.loads(kw["Payload"].decode()))
            return {}

    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(H.boto3, "client", lambda name, **k: FakeLambda())
    monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "pod-pipeline-stg", raising=False)
    monkeypatch.setattr(H.time, "sleep", lambda *_: None)

    body = json.loads(H.handler({"warmup": True, "fanout": 3}, FakeContext(900_000))["body"])

    assert body["warmed"] == 3
    assert len(invokes) == 2                       # this container + 2 siblings
    assert all(i["fanout"] == 0 for i in invokes)  # siblings never recurse


def test_warm_connection_is_reused_across_invocations(monkeypatch):
    """A warm container must not redial Postgres for every trip."""
    conn = FakeConn()
    dials = {"n": 0}

    def _dial(*a, **k):
        dials["n"] += 1
        return conn

    monkeypatch.setattr(H, "_conn", None, raising=False)
    monkeypatch.setattr(H, "_connect", _dial)
    assert H.get_db_connection() is conn
    assert H.get_db_connection() is conn
    assert dials["n"] == 1                         # dialled once, reused after
    H.release_db_connection(conn)
    assert conn.closed == 0                        # the warm handle stays open


def test_warm_connection_redials_when_stale(monkeypatch):
    dead, fresh = FakeConn(), FakeConn()
    dead.closed = 1
    monkeypatch.setattr(H, "_conn", dead, raising=False)
    monkeypatch.setattr(H, "_connect", lambda *a, **k: fresh)
    assert H.get_db_connection() is fresh


# --------------------------------------------------------------------------- #
# Batch selection: date range / ad-hoc SQL
# --------------------------------------------------------------------------- #

def test_resolve_batch_source_defaults_to_source_query(monkeypatch):
    monkeypatch.setattr(H, "SOURCE_QUERY", "SELECT 1", raising=False)
    sql, params, source = H.resolve_batch_source({})
    assert (sql, params, source) == ("SELECT 1", None, "source_query")


def test_resolve_batch_source_binds_date_range():
    sql, params, source = H.resolve_batch_source(
        {"start_date": "2026-08-01", "end_date": "2026-08-20"})
    assert source == "date_range"
    assert params == {"start_date": "2026-08-01", "end_date": "2026-08-20"}
    assert "%(start_date)s" in sql and "%(end_date)s" in sql


def test_resolve_batch_source_single_date_is_a_one_day_range():
    _, params, _ = H.resolve_batch_source({"start_date": "2026-08-01"})
    assert params == {"start_date": "2026-08-01", "end_date": "2026-08-01"}


@pytest.mark.parametrize("event", [
    {"start_date": "01-08-2026", "end_date": "2026-08-20"},
    {"start_date": "2026-08-20", "end_date": "2026-08-01"},
])
def test_resolve_batch_source_rejects_bad_ranges(event):
    with pytest.raises(ValueError):
        H.resolve_batch_source(event)


def test_adhoc_query_accepts_read_only_sql(monkeypatch):
    monkeypatch.setattr(H, "ALLOW_ADHOC_QUERY", True, raising=False)
    sql, params, source = H.resolve_batch_source(
        {"query": "SELECT awb, trip_id, pod FROM kaptaan WHERE awb = 'X';"})
    assert source == "adhoc_query" and params is None
    assert sql.endswith("'X'")            # trailing semicolon stripped
    assert H.resolve_batch_source({"query": "WITH x AS (SELECT 1) SELECT * FROM x"})[2] == "adhoc_query"


@pytest.mark.parametrize("bad", [
    "DELETE FROM pod_scores",
    "SELECT 1; DROP TABLE pod_scores",
    "UPDATE pod_scores SET pod_score = 1",
    "SELECT 1 -- comment",
    "INSERT INTO pod_scores VALUES (1)",
    "TRUNCATE pod_scores",
    "",
])
def test_adhoc_query_rejects_anything_that_writes(monkeypatch, bad):
    monkeypatch.setattr(H, "ALLOW_ADHOC_QUERY", True, raising=False)
    with pytest.raises(ValueError):
        H.resolve_batch_source({"query": bad})


def test_adhoc_query_can_be_disabled(monkeypatch):
    monkeypatch.setattr(H, "ALLOW_ADHOC_QUERY", False, raising=False)
    with pytest.raises(ValueError):
        H.resolve_batch_source({"query": "SELECT 1"})


def test_batch_handler_runs_a_date_range(wire, monkeypatch):
    seen = {}

    def _fetch(sql=None, params=None):
        seen["sql"], seen["params"] = sql, params
        return _mb_df(3)

    monkeypatch.setattr(H, "fetch_pod_data", _fetch)
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    body = json.loads(H.handler(
        {"start_date": "2026-08-01", "end_date": "2026-08-02"}, FakeContext(900_000))["body"])

    assert body["mode"] == "batch" and body["source"] == "date_range"
    assert seen["params"] == {"start_date": "2026-08-01", "end_date": "2026-08-02"}
    assert body["scored_this_invocation"] == 3


def test_batch_handler_rejects_a_bad_request_with_400(wire, monkeypatch):
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    resp = H.handler({"query": "DROP TABLE pod_scores"}, FakeContext(900_000))
    assert resp["statusCode"] == 400
    assert wire.upserted == []


def test_continuation_carries_the_selection_forward(wire, monkeypatch):
    payloads = []
    monkeypatch.setattr(H, "invoke_continuation",
                        lambda rid, rd, c, sel=None: payloads.append(sel))
    monkeypatch.setattr(H, "fetch_pod_data", lambda sql=None, params=None: _mb_df(2000))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    H.handler({"start_date": "2026-08-01", "end_date": "2026-08-02"}, FakeContext(1_000))
    assert payloads == [{"start_date": "2026-08-01", "end_date": "2026-08-02"}]


# --------------------------------------------------------------------------- #
# Schema migration / teardown
# --------------------------------------------------------------------------- #

@pytest.fixture
def schema_file(tmp_path, monkeypatch):
    f = tmp_path / "schema.sql"
    f.write_text("CREATE TABLE IF NOT EXISTS pod_scores (id serial);")
    monkeypatch.setattr(H, "SCHEMA_PATH", str(f), raising=False)
    return f


def test_migrate_applies_the_baked_in_schema(wire, schema_file):
    body = json.loads(H.handler({"migrate": True}, FakeContext(900_000))["body"])

    assert body["status"] == "applied"
    assert body["destructive"] is False
    assert wire.committed == 1
    assert any("CREATE TABLE IF NOT EXISTS pod_scores" in sql for sql, _ in wire.executed)


def test_migrate_is_idempotent(wire, schema_file):
    """schema.sql is IF NOT EXISTS throughout — CI applies it on every deploy."""
    for _ in range(3):
        body = json.loads(H.handler({"migrate": True}, FakeContext(900_000))["body"])
        assert body["status"] == "applied"
    assert wire.committed == 3


def test_migrate_reports_a_missing_schema_file(wire, monkeypatch):
    monkeypatch.setattr(H, "SCHEMA_PATH", "/nope/schema.sql", raising=False)
    resp = H.handler({"migrate": True}, FakeContext(900_000))

    assert resp["statusCode"] == 500
    assert json.loads(resp["body"])["status"] == "failed"
    assert wire.committed == 0


def test_migrate_never_scores(wire, schema_file):
    H.handler({"migrate": True}, FakeContext(900_000))
    assert wire.upserted == []


def test_destructive_drop_requires_the_functions_own_name(wire, monkeypatch):
    monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "pod-pipeline-stg", raising=False)

    for bad in ({"migrate": "drop"},
                {"migrate": "drop", "confirm": ""},
                {"migrate": "drop", "confirm": "pod-pipeline-prod"},   # another env
                {"migrate": "drop", "confirm": "yes"}):
        resp = H.handler(bad, FakeContext(900_000))
        assert resp["statusCode"] == 400, bad
        assert json.loads(resp["body"])["status"] == "refused"

    assert wire.committed == 0
    assert not any("DROP" in sql for sql, _ in wire.executed)


def test_destructive_drop_runs_when_confirmed(wire, monkeypatch):
    monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "pod-pipeline-stg", raising=False)

    body = json.loads(H.handler(
        {"migrate": "drop", "confirm": "pod-pipeline-stg"}, FakeContext(900_000))["body"])

    assert body["status"] == "dropped"
    assert body["destructive"] is True
    dropped = [sql for sql, _ in wire.executed if sql.startswith("DROP")]
    assert dropped == ["DROP VIEW IF EXISTS pod_scores_flagged",
                       "DROP TABLE IF EXISTS pod_scores"]      # view before table


def test_destructive_drop_is_refused_when_the_function_is_unnamed(wire, monkeypatch):
    """Locally / in tests there is no function name — nothing to confirm against."""
    monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "", raising=False)
    resp = H.handler({"migrate": "drop", "confirm": ""}, FakeContext(900_000))
    assert resp["statusCode"] == 400


def test_migrate_wins_over_the_other_lanes(wire, schema_file, monkeypatch):
    """A migrate event carrying a stray trip_id must not start scoring."""
    monkeypatch.setattr(H, "handle_single_trip",
                        lambda e, c: pytest.fail("should not have scored"))
    body = json.loads(H.handler(
        {"migrate": True, "trip_id": 5}, FakeContext(900_000))["body"])
    assert body["status"] == "applied"


# --------------------------------------------------------------------------- #
# Source database resolution
# --------------------------------------------------------------------------- #

def _reload_with(monkeypatch, **env):
    """Re-import the handler with a given environment (module-level config)."""
    import importlib
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(H)


def test_source_db_inherits_results_db_when_unset(monkeypatch):
    m = _reload_with(monkeypatch, PG_HOST="db.internal", PG_DATABASE="pod_classifier",
                     PG_USER="postgres", PG_PASSWORD="pw",
                     SOURCE_PG_HOST="", SOURCE_PG_DATABASE="",
                     SOURCE_PG_USER="", SOURCE_PG_PASSWORD="")
    try:
        # CloudFormation always sets these, empty when unset — empty must inherit
        # rather than blank the connection out.
        assert m.SOURCE_PG_HOST == "db.internal"
        assert m.SOURCE_PG_DATABASE == "pod_classifier"
        assert m.SOURCE_PG_USER == "postgres"
        assert m.SOURCE_PG_PASSWORD == "pw"
    finally:
        importlib_reload_clean(monkeypatch)


def test_source_db_can_be_a_sibling_database_on_the_same_cluster(monkeypatch):
    """The real shape: POD rows in `kaptaan` in the app DB, scores written apart."""
    m = _reload_with(monkeypatch, PG_HOST="db.internal", PG_DATABASE="pod_classifier",
                     PG_USER="postgres", PG_PASSWORD="pw",
                     SOURCE_PG_HOST="", SOURCE_PG_DATABASE="sarathy",
                     SOURCE_PG_USER="", SOURCE_PG_PASSWORD="")
    try:
        assert m.SOURCE_PG_DATABASE == "sarathy"      # the one thing that differs
        assert m.SOURCE_PG_HOST == "db.internal"      # same cluster
        assert m.SOURCE_PG_PASSWORD == "pw"           # same credentials
        assert m.PG_DATABASE == "pod_classifier"      # results stay put
    finally:
        importlib_reload_clean(monkeypatch)


def importlib_reload_clean(monkeypatch):
    """Restore the module to the ambient test environment."""
    import importlib
    for k in ("PG_HOST", "PG_DATABASE", "PG_USER", "PG_PASSWORD",
              "SOURCE_PG_HOST", "SOURCE_PG_DATABASE",
              "SOURCE_PG_USER", "SOURCE_PG_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(H)


# --------------------------------------------------------------------------- #
# Locking / session hygiene on a shared operational cluster
# --------------------------------------------------------------------------- #

def test_transaction_commits_and_restores_autocommit():
    conn = FakeConn()
    with H._transaction(conn) as tx:
        assert tx.autocommit is False        # inside: a real transaction
    assert conn.committed == 1
    assert conn.autocommit is True           # out: nothing left open


def test_transaction_rolls_back_and_still_restores_autocommit():
    conn = FakeConn()
    with pytest.raises(RuntimeError):
        with H._transaction(conn):
            raise RuntimeError("boom")
    assert conn.rolled_back == 1
    assert conn.committed == 0
    assert conn.autocommit is True           # never left idle in transaction


def test_reads_do_not_open_a_transaction(wire, monkeypatch):
    """A SELECT must not hold a transaction across downloads and inference."""
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: _trip_df("T1", 1))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    H.handler({"trip_id": "T1"}, FakeContext(900_000))
    # autocommit is only ever toggled off inside upsert_results' transaction,
    # and is back on afterwards.
    assert wire.autocommit is True


def test_already_scored_path_leaves_no_open_transaction(wire, monkeypatch):
    """The early return skips the upsert — it must not strand a transaction."""
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: _trip_df("T2", 1))
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    wire.scored_links = {"http://img/T2_0.png"}

    body = json.loads(H.handler({"trip_id": "T2"}, FakeContext(900_000))["body"])

    assert body["status"] == "already_scored"
    assert wire.autocommit is True           # connection is idle, not idle-in-tx
    assert wire.committed == 0


def test_connect_sets_timeouts_and_read_only_for_the_source(monkeypatch):
    calls = []

    def _fake_connect(**kw):
        calls.append(kw)
        return FakeConn()

    monkeypatch.setattr(H.psycopg2, "connect", _fake_connect)
    monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "pod-pipeline-stg", raising=False)

    results = H._connect("h", "5432", "sarathy", "u", "p")
    source = H._connect("h", "5432", "sarathy", "u", "p", read_only=True)

    for kw in calls:
        assert "statement_timeout" in kw["options"]
        assert "idle_in_transaction_session_timeout" in kw["options"]
        assert kw["application_name"] == "pod-pipeline-stg"   # visible in pg_stat_activity
    assert "default_transaction_read_only=on" not in calls[0]["options"]
    assert "default_transaction_read_only=on" in calls[1]["options"]
    assert results.autocommit is True and source.autocommit is True


def test_source_connection_is_closed_before_the_slow_work(monkeypatch):
    """kaptaan must not be held while images download and score."""
    conn = FakeConn()
    monkeypatch.setattr(H, "_connect", lambda *a, **k: conn)
    H.fetch_pod_data("SELECT awb, trip_id, pod FROM kaptaan")
    assert conn.closed == 1


def test_event_awb_is_used_as_is_without_a_lookup(wire, monkeypatch):
    called = {"n": 0}

    def _counted(tid):
        called["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(H, "fetch_trip_pod_data", _counted)
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())

    H.handler({"trip_id": 91, "awb": "GS123", "pod_links": ["http://img/a.png"]},
              FakeContext(900_000))

    assert [r["awb"] for r in wire.upserted] == ["GS123"]
    assert called["n"] == 0, "nothing to resolve — the caller supplied the AWB"


def test_awb_is_resolved_from_the_trip_table_when_the_event_omits_it(wire, monkeypatch):
    """Sarathy sends links but no AWB; TRIP-<id> would join to nothing."""
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: pd.DataFrame([
        {"awb": "GS4321728395", "trip_id": tid, "pod": "http://img/from_table.png"},
    ]))

    body = json.loads(H.handler({
        "trip_id": 116126,
        "pod_links": ["http://img/1.webp", "http://img/2.webp"],
    }, FakeContext(900_000))["body"])

    assert body["source"] == "event_payload"
    assert body["scored"] == 2
    assert {r["awb"] for r in wire.upserted} == {"GS4321728395"}
    assert {r["pod_link"] for r in wire.upserted} == {
        "http://img/1.webp", "http://img/2.webp"}


def test_placeholder_awb_survives_when_the_trip_row_is_gone(wire, monkeypatch):
    """Nothing to resolve from — the row must still be written, keyed on the trip."""
    monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
    monkeypatch.setattr(H, "fetch_trip_pod_data", lambda tid: pd.DataFrame())

    H.handler({"trip_id": 92, "pod_links": ["http://img/a.png"]}, FakeContext(900_000))
    assert [r["awb"] for r in wire.upserted] == ["TRIP-92"]
