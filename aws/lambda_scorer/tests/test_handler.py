"""
Test suite for the POD scoring Lambda handler.

The handler owns no data: it asks sarathy what to score and posts the results
back. Every test here therefore stands a fake SarathyClient in front of the
handler and asserts on what it was asked for and what it was handed — that is
the whole of the service boundary, so it is the whole of what these tests need
to pin down.

Covered:
  * event routing (warmup / single trip / batch, and the envelope shapes)
  * event links win over sarathy's; sarathy still supplies the AWB
  * already-scored links are never downloaded, and re-requests re-score
    replaced photos
  * a failed download is recorded as a row, never silently dropped
  * batch paging, windowing, date-range validation and resume
  * clock-aware continuation fires only near the wall
  * sarathy being down is a 502, not a half-written run
  * preprocessing shape + normalization

Pipeline-logic tests fake the model, so they run without a real checkpoint. A
separate torch-gated test exercises the real inference wiring.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler as H  # noqa: E402
from sarathy_client import SarathyError  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers / fakes
# --------------------------------------------------------------------------- #

def _png_bytes(w=32, h=32):
    import cv2
    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class FakeResp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status


class FakeSession:
    """Returns a valid PNG for most URLs; configurable failures."""

    def __init__(self, fail_urls=None, http_status=None, tiny_urls=None):
        self.fail_urls = set(fail_urls or ())
        self.http_status = http_status or {}
        self.tiny_urls = set(tiny_urls or ())
        self.requested = []

    def get(self, url, timeout=0):
        self.requested.append(url)
        if url in self.fail_urls:
            raise ConnectionError("boom")
        if url in self.tiny_urls:
            return FakeResp(b"x", 200)
        if url in self.http_status:
            return FakeResp(_png_bytes(), self.http_status[url])
        return FakeResp(_png_bytes(), 200)


class FakeSarathy:
    """Stands in for SarathyClient. Records every call and every row written."""

    def __init__(self, trip_items=None, pages=None, fail_on=()):
        self._trip_items = trip_items or {}
        self._pages = pages or []
        self._fail_on = set(fail_on)
        self.trip_calls = []
        self.range_calls = []
        self.writes = []          # [(run_date, [row, ...]), ...]

    # reads
    def trip_rows(self, trip_id):
        self.trip_calls.append(str(trip_id))
        if "trip_rows" in self._fail_on:
            raise SarathyError("sarathy unreachable")
        return [dict(r) for r in self._trip_items.get(str(trip_id), [])]

    def range_rows(self, start_date, end_date):
        self.range_calls.append((start_date, end_date))
        if "range_rows" in self._fail_on:
            raise SarathyError("sarathy unreachable")
        for page in self._pages:
            yield [dict(r) for r in page]

    # write
    def write_scores(self, run_date, results):
        if "write_scores" in self._fail_on:
            raise SarathyError("sarathy refused the write")
        self.writes.append((run_date, [dict(r) for r in results]))
        return len(results)

    # convenience
    @property
    def written_rows(self):
        return [r for _, rows in self.writes for r in rows]

    @property
    def written_links(self):
        return [r["pod_link"] for r in self.written_rows]


class FakeContext:
    def __init__(self, remaining_ms=10 ** 9):
        self._remaining = remaining_ms

    def get_remaining_time_in_millis(self):
        return self._remaining


def _row(link, awb="AWB1", trip_id="1", already=False):
    return {"awb": awb, "trip_id": trip_id, "pod_link": link, "already_scored": already}


def _body(resp):
    return json.loads(resp["body"])


@pytest.fixture
def wired(monkeypatch):
    """Handler wired to a fake sarathy, a fake download session and a fake model.

    Returns the FakeSarathy so a test can assert on the traffic across the
    boundary; the session is reachable as `.session`.
    """
    sar = FakeSarathy()
    session = FakeSession()

    monkeypatch.setattr(H, "SARATHY_BASE_URL", "http://sarathy.internal:8080")
    monkeypatch.setattr(H, "get_sarathy", lambda: sar)
    monkeypatch.setattr(H, "build_session", lambda *a, **k: session)
    monkeypatch.setattr(H, "get_model", lambda: ("model", "cpu"))
    monkeypatch.setattr(
        H, "score_prepared",
        lambda model, device, successes: [
            {"awb": s["awb"], "trip_id": s["trip_id"], "pod_link": s["pod_link"],
             "status": "scored", "failure_reason": None, "pod_score": 0.9,
             "context_valid_prob": 0.9, "package_visible_prob": 0.9,
             "label_readable_prob": 0.9, "image_clarity_prob": 0.9}
            for s in successes
        ],
    )
    monkeypatch.setattr(H, "emit_coverage", lambda *a, **k: None)
    sar.session = session
    return sar


# --------------------------------------------------------------------------- #
# Event normalisation and routing
# --------------------------------------------------------------------------- #

class TestEventNormalisation:
    def test_none_and_garbage_become_empty(self):
        assert H._normalise_event(None) == {}
        assert H._normalise_event("not json") == {}
        assert H._normalise_event(["a", "list"]) == {}

    def test_json_string_is_parsed(self):
        assert H._normalise_event('{"trip_id": 7}') == {"trip_id": 7}

    def test_eventbridge_detail_is_unwrapped(self):
        assert H._normalise_event({"detail": {"trip_id": 7}}) == {"trip_id": 7}

    def test_sqs_record_body_is_unwrapped(self):
        event = {"Records": [{"body": json.dumps({"trip_id": 7})}]}
        assert H._normalise_event(event) == {"trip_id": 7}

    def test_sqs_record_with_bad_body_is_empty(self):
        assert H._normalise_event({"Records": [{"body": "{oops"}]}) == {}

    def test_top_level_trip_id_beats_the_envelope(self):
        event = {"trip_id": 1, "detail": {"trip_id": 2}}
        assert H._normalise_event(event)["trip_id"] == 1


class TestRouting:
    def test_warmup_routes_to_warmup(self, monkeypatch):
        monkeypatch.setattr(H, "handle_warmup", lambda e: {"routed": "warmup"})
        assert H.handler({"warmup": True}, None) == {"routed": "warmup"}

    def test_trip_id_routes_to_single_trip(self, monkeypatch):
        monkeypatch.setattr(H, "handle_single_trip", lambda e, c: {"routed": "trip"})
        assert H.handler({"trip_id": 5}, None) == {"routed": "trip"}

    def test_empty_event_routes_to_batch(self, monkeypatch):
        monkeypatch.setattr(H, "handle_batch", lambda e, c: {"routed": "batch"})
        assert H.handler({}, None) == {"routed": "batch"}

    def test_blank_trip_id_routes_to_batch(self, monkeypatch):
        monkeypatch.setattr(H, "handle_batch", lambda e, c: {"routed": "batch"})
        assert H.handler({"trip_id": ""}, None) == {"routed": "batch"}


# --------------------------------------------------------------------------- #
# Links carried in the trigger event
# --------------------------------------------------------------------------- #

class TestRowsFromEvent:
    def test_list_of_links(self):
        rows = H.rows_from_event({"pod_links": ["http://a", "http://b"]}, "9")
        assert [r["pod_link"] for r in rows] == ["http://a", "http://b"]
        assert all(r["trip_id"] == "9" for r in rows)

    def test_comma_separated_string_is_expanded(self):
        rows = H.rows_from_event({"pod_links": "http://a, http://b"}, "9")
        assert [r["pod_link"] for r in rows] == ["http://a", "http://b"]

    def test_pod_key_is_accepted_too(self):
        rows = H.rows_from_event({"pod": ["http://a"]}, "9")
        assert len(rows) == 1

    def test_duplicates_are_dropped(self):
        rows = H.rows_from_event({"pod_links": ["http://a", "http://a"]}, "9")
        assert len(rows) == 1

    def test_non_http_values_are_ignored(self):
        rows = H.rows_from_event({"pod_links": ["", "null", "/local/path"]}, "9")
        assert rows == []

    def test_awb_falls_back_to_a_trip_placeholder(self):
        rows = H.rows_from_event({"pod_links": ["http://a"]}, "9")
        assert rows[0]["awb"] == "TRIP-9"

    def test_event_awb_is_used_when_present(self):
        rows = H.rows_from_event({"pod_links": ["http://a"], "awb": " AWB7 "}, "9")
        assert rows[0]["awb"] == "AWB7"

    def test_event_rows_are_never_pre_marked_as_scored(self):
        rows = H.rows_from_event({"pod_links": ["http://a"]}, "9")
        assert rows[0]["already_scored"] is False


# --------------------------------------------------------------------------- #
# Download + preprocess
# --------------------------------------------------------------------------- #

class TestPreprocess:
    def test_shape_is_chw_float32(self):
        out = H.preprocess_image(np.zeros((64, 48, 3), dtype=np.uint8), size=224)
        assert out.shape == (3, 224, 224)
        assert out.dtype == np.float32

    def test_normalisation_shifts_the_values(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        plain = H.preprocess_image(img, size=32, normalize=False)
        norm = H.preprocess_image(img, size=32, normalize=True)
        assert 0.0 <= plain.min() and plain.max() <= 1.0
        assert not np.allclose(plain, norm)


class TestDownload:
    def test_success_carries_a_prepared_tensor(self):
        out = H.download_and_prepare(FakeSession(), _row("http://ok"))
        assert out["chw"].shape == (3, H.INPUT_SIZE, H.INPUT_SIZE)

    def test_http_error_is_an_outcome_not_an_exception(self):
        session = FakeSession(http_status={"http://bad": 404})
        out = H.download_and_prepare(session, _row("http://bad"))
        assert out["status"] == "download_failed"
        assert out["failure_reason"] == "http_404"

    def test_truncated_body_is_rejected(self):
        session = FakeSession(tiny_urls={"http://tiny"})
        out = H.download_and_prepare(session, _row("http://tiny"))
        assert out["failure_reason"] == "too_small"

    def test_undecodable_body_is_rejected(self):
        class Junk(FakeSession):
            def get(self, url, timeout=0):
                return FakeResp(b"z" * 5000, 200)

        out = H.download_and_prepare(Junk(), _row("http://junk"))
        assert out["failure_reason"] == "decode_failed"

    def test_transport_error_is_recorded_by_type(self):
        session = FakeSession(fail_urls={"http://dead"})
        out = H.download_and_prepare(session, _row("http://dead"))
        assert out["failure_reason"] == "ConnectionError"

    def test_window_partitions_and_loses_nothing(self):
        rows = [_row(f"http://img{i}") for i in range(6)]
        session = FakeSession(fail_urls={"http://img2"}, http_status={"http://img4": 500})
        prepared, failures = H.download_window(session, rows, max_workers=4)
        assert len(prepared) + len(failures) == 6
        assert {f["pod_link"] for f in failures} == {"http://img2", "http://img4"}

    def test_empty_window_is_a_no_op(self):
        prepared, failures = H.download_window(FakeSession(), [], max_workers=4)
        assert (prepared, failures) == ([], [])


# --------------------------------------------------------------------------- #
# score_and_record — the one place rows cross back to sarathy
# --------------------------------------------------------------------------- #

class TestScoreAndRecord:
    def test_empty_input_writes_nothing(self, wired):
        assert H.score_and_record([], "2026-09-10", 4) == (0, 0)
        assert wired.writes == []

    def test_successes_and_failures_are_written_together(self, wired):
        wired.session.fail_urls = {"http://b"}
        rows = [_row("http://a"), _row("http://b")]
        scored, failed = H.score_and_record(rows, "2026-09-10", 4)
        assert (scored, failed) == (1, 1)
        run_date, written = wired.writes[0]
        assert run_date == "2026-09-10"
        assert {r["pod_link"] for r in written} == {"http://a", "http://b"}
        statuses = {r["pod_link"]: r["status"] for r in written}
        assert statuses == {"http://a": "scored", "http://b": "download_failed"}

    def test_every_input_gets_exactly_one_row(self, wired):
        wired.session.fail_urls = {"http://img1", "http://img3"}
        rows = [_row(f"http://img{i}") for i in range(5)]
        H.score_and_record(rows, "2026-09-10", 4)
        assert sorted(wired.written_links) == sorted(r["pod_link"] for r in rows)


# --------------------------------------------------------------------------- #
# Warmup
# --------------------------------------------------------------------------- #

class TestWarmup:
    def test_warms_this_container_without_fanning_out(self, monkeypatch):
        monkeypatch.setattr(H, "get_model", lambda: ("m", "cpu"))
        monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
        body = _body(H.handle_warmup({"warmup": True, "fanout": 1}))
        assert body == {"status": "warm", "warmed": 1}

    def test_fanout_self_invokes_for_the_rest_of_the_pool(self, monkeypatch):
        monkeypatch.setattr(H, "get_model", lambda: ("m", "cpu"))
        monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
        monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "pod-pipeline-stg")
        monkeypatch.setenv("WARM_HOLD_SECONDS", "0")
        calls = []

        class FakeLambda:
            def invoke(self, **kw):
                calls.append(kw)

        monkeypatch.setattr(H.boto3, "client", lambda name: FakeLambda())
        body = _body(H.handle_warmup({"warmup": True, "fanout": 3}))
        assert body["warmed"] == 3
        assert len(calls) == 2
        assert all(kw["InvocationType"] == "Event" for kw in calls)

    def test_a_failed_fanout_still_reports_this_container(self, monkeypatch):
        monkeypatch.setattr(H, "get_model", lambda: ("m", "cpu"))
        monkeypatch.setattr(H, "build_session", lambda *a, **k: FakeSession())
        monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "pod-pipeline-stg")
        monkeypatch.setenv("WARM_HOLD_SECONDS", "0")

        class Broken:
            def invoke(self, **kw):
                raise RuntimeError("throttled")

        monkeypatch.setattr(H.boto3, "client", lambda name: Broken())
        assert _body(H.handle_warmup({"fanout": 4}))["warmed"] == 1


# --------------------------------------------------------------------------- #
# Single trip
# --------------------------------------------------------------------------- #

class TestSingleTrip:
    def test_event_links_are_scored_without_asking_sarathy_for_links(self, wired):
        wired._trip_items = {"42": [_row("http://old", awb="AWB42", trip_id="42")]}
        resp = H.handle_single_trip(
            {"trip_id": 42, "pod_links": ["http://new"], "awb": "AWB42"}, FakeContext())
        body = _body(resp)
        assert body["status"] == "complete"
        assert body["source"] == "event_payload"
        assert wired.written_links == ["http://new"]
        # the event carried an AWB, so there was nothing to look up
        assert wired.trip_calls == []

    def test_sarathy_supplies_links_when_the_event_has_none(self, wired):
        wired._trip_items = {"42": [_row("http://a", awb="AWB42", trip_id="42")]}
        body = _body(H.handle_single_trip({"trip_id": 42}, FakeContext()))
        assert body["source"] == "sarathy"
        assert wired.written_links == ["http://a"]

    def test_awb_comes_from_sarathy_when_the_event_omits_it(self, wired):
        """A TRIP-<id> placeholder is not an AWB — the trip row is the source."""
        wired._trip_items = {"42": [_row("http://old", awb="REALAWB", trip_id="42")]}
        H.handle_single_trip({"trip_id": 42, "pod_links": ["http://new"]}, FakeContext())
        assert wired.trip_calls == ["42"]
        assert {r["awb"] for r in wired.written_rows} == {"REALAWB"}

    def test_already_scored_links_are_not_downloaded_again(self, wired):
        wired._trip_items = {"42": [
            _row("http://a", awb="AWB42", trip_id="42", already=True),
            _row("http://b", awb="AWB42", trip_id="42"),
        ]}
        body = _body(H.handle_single_trip({"trip_id": 42}, FakeContext()))
        assert body["skipped_already_scored"] == 1
        assert body["scored"] == 1
        assert wired.written_links == ["http://b"]
        assert "http://a" not in wired.session.requested

    def test_a_replaced_photo_is_scored_even_on_a_re_request(self, wired):
        """The rider swapped one image; only the new link is unscored."""
        wired._trip_items = {"42": [
            _row("http://kept", awb="AWB42", trip_id="42", already=True),
            _row("http://fresh", awb="AWB42", trip_id="42", already=False),
        ]}
        H.handle_single_trip(
            {"trip_id": 42, "pod_links": ["http://kept", "http://fresh"]}, FakeContext())
        assert wired.written_links == ["http://fresh"]

    def test_all_links_already_scored_short_circuits(self, wired):
        wired._trip_items = {"42": [
            _row("http://a", awb="AWB42", trip_id="42", already=True)]}
        body = _body(H.handle_single_trip({"trip_id": 42}, FakeContext()))
        assert body["status"] == "already_scored"
        assert body["scored"] == 0
        assert wired.writes == []

    def test_no_links_anywhere_is_no_data(self, wired):
        body = _body(H.handle_single_trip({"trip_id": 999}, FakeContext()))
        assert body["status"] == "no_data"
        assert wired.writes == []

    def test_sarathy_down_with_no_event_links_is_a_502(self, wired):
        wired._fail_on = {"trip_rows"}
        resp = H.handle_single_trip({"trip_id": 42}, FakeContext())
        assert resp["statusCode"] == 502
        assert _body(resp)["status"] == "failed"

    def test_sarathy_down_still_scores_the_links_the_event_carried(self, wired):
        wired._fail_on = {"trip_rows"}
        body = _body(H.handle_single_trip(
            {"trip_id": 42, "pod_links": ["http://a"]}, FakeContext()))
        assert body["status"] == "complete"
        assert wired.written_links == ["http://a"]

    def test_missing_base_url_is_a_500(self, wired, monkeypatch):
        monkeypatch.setattr(H, "SARATHY_BASE_URL", "")
        resp = H.handle_single_trip({"trip_id": 42}, FakeContext())
        assert resp["statusCode"] == 500
        assert "SARATHY_BASE_URL" in _body(resp)["error"]

    def test_trip_id_is_stamped_on_every_written_row(self, wired):
        body = _body(H.handle_single_trip(
            {"trip_id": 77, "pod_links": ["http://a"], "awb": "A"}, FakeContext()))
        assert body["trip_id"] == "77"
        assert {r["trip_id"] for r in wired.written_rows} == {"77"}


# --------------------------------------------------------------------------- #
# Batch date range
# --------------------------------------------------------------------------- #

class TestResolveRange:
    def test_empty_event_is_today(self):
        from datetime import date
        today = date.today().isoformat()
        assert H.resolve_range({}) == (today, today)

    def test_explicit_range_is_kept(self):
        assert H.resolve_range(
            {"start_date": "2026-09-01", "end_date": "2026-09-09"}
        ) == ("2026-09-01", "2026-09-09")

    def test_start_alone_means_a_single_day(self):
        assert H.resolve_range({"start_date": "2026-09-01"}) == ("2026-09-01", "2026-09-01")

    def test_end_alone_means_a_single_day(self):
        assert H.resolve_range({"end_date": "2026-09-01"}) == ("2026-09-01", "2026-09-01")

    def test_malformed_date_is_rejected(self):
        with pytest.raises(ValueError, match="start_date"):
            H.resolve_range({"start_date": "01-09-2026"})

    def test_inverted_range_is_rejected(self):
        with pytest.raises(ValueError, match="after"):
            H.resolve_range({"start_date": "2026-09-09", "end_date": "2026-09-01"})


class TestBatch:
    def test_pages_are_scored_and_summed(self, wired):
        wired._pages = [
            [_row("http://a"), _row("http://b")],
            [_row("http://c")],
        ]
        body = _body(H.handle_batch({"start_date": "2026-09-01",
                                     "end_date": "2026-09-02"}, FakeContext()))
        assert body["status"] == "complete"
        assert body["total_images"] == 3
        assert body["scored_this_invocation"] == 3
        assert sorted(wired.written_links) == ["http://a", "http://b", "http://c"]
        assert wired.range_calls == [("2026-09-01", "2026-09-02")]

    def test_already_scored_rows_are_skipped_not_re_downloaded(self, wired):
        wired._pages = [[_row("http://a", already=True), _row("http://b")]]
        body = _body(H.handle_batch({}, FakeContext()))
        assert body["skipped_already_scored"] == 1
        assert body["scored_this_invocation"] == 1
        assert "http://a" not in wired.session.requested

    def test_failures_are_counted_and_still_written(self, wired):
        wired._pages = [[_row("http://a"), _row("http://b")]]
        wired.session.fail_urls = {"http://b"}
        body = _body(H.handle_batch({}, FakeContext()))
        assert (body["scored_this_invocation"], body["failed_this_invocation"]) == (1, 1)
        assert len(wired.written_rows) == 2

    def test_a_page_larger_than_the_window_is_split(self, wired, monkeypatch):
        monkeypatch.setattr(H, "WINDOW_SIZE", 2)
        wired._pages = [[_row(f"http://img{i}") for i in range(5)]]
        body = _body(H.handle_batch({}, FakeContext()))
        assert body["scored_this_invocation"] == 5
        assert len(wired.writes) == 3          # 2 + 2 + 1
        assert len(wired.written_rows) == 5

    def test_empty_range_completes_with_nothing_written(self, wired):
        body = _body(H.handle_batch({}, FakeContext()))
        assert body["status"] == "complete"
        assert body["total_images"] == 0
        assert wired.writes == []

    def test_bad_range_is_a_400(self, wired):
        resp = H.handle_batch({"start_date": "nope"}, FakeContext())
        assert resp["statusCode"] == 400
        assert wired.range_calls == []

    def test_missing_base_url_is_a_500(self, wired, monkeypatch):
        monkeypatch.setattr(H, "SARATHY_BASE_URL", "")
        assert H.handle_batch({}, FakeContext())["statusCode"] == 500

    def test_sarathy_failing_mid_run_is_a_502_reporting_partial_work(self, wired):
        wired._fail_on = {"range_rows"}
        resp = H.handle_batch({}, FakeContext())
        assert resp["statusCode"] == 502
        assert _body(resp)["status"] == "failed"


class TestContinuation:
    def test_ample_time_never_continues(self, wired, monkeypatch):
        fired = []
        monkeypatch.setattr(H, "invoke_continuation",
                            lambda *a, **k: fired.append(a))
        wired._pages = [[_row(f"http://img{i}") for i in range(4)]]
        body = _body(H.handle_batch({}, FakeContext(remaining_ms=10 ** 9)))
        assert body["status"] == "complete"
        assert fired == []

    def test_near_the_wall_it_hands_off(self, wired, monkeypatch):
        fired = []
        monkeypatch.setattr(H, "invoke_continuation",
                            lambda *a, **k: fired.append((a, k)))
        wired._pages = [[_row("http://a")]]
        body = _body(H.handle_batch({}, FakeContext(remaining_ms=1000)))
        assert body["status"] == "continuing"
        assert len(fired) == 1
        assert wired.writes == []

    def test_the_continuation_carries_the_range_forward(self, wired, monkeypatch):
        fired = []
        monkeypatch.setattr(H, "invoke_continuation",
                            lambda run_id, run_date, cont, selection=None:
                            fired.append((cont, selection)))
        wired._pages = [[_row("http://a")]]
        H.handle_batch({"start_date": "2026-09-01", "end_date": "2026-09-05"},
                       FakeContext(remaining_ms=1000))
        cont, selection = fired[0]
        assert cont == 1
        assert selection == {"start_date": "2026-09-01", "end_date": "2026-09-05"}

    def test_the_chain_stops_at_the_cap(self, wired, monkeypatch):
        fired = []
        monkeypatch.setattr(H, "invoke_continuation", lambda *a, **k: fired.append(a))
        wired._pages = [[_row("http://a")]]
        body = _body(H.handle_batch({"continuation": H.MAX_CONTINUATIONS},
                                    FakeContext(remaining_ms=1000)))
        assert body["status"] == "incomplete"
        assert fired == []

    def test_a_missing_context_is_treated_as_ample_time(self):
        assert H._remaining_ms(None) == 10 ** 9

    def test_a_real_context_is_read(self):
        assert H._remaining_ms(FakeContext(remaining_ms=4242)) == 4242


class TestInvokeContinuation:
    def test_payload_carries_the_checkpoint(self, monkeypatch):
        sent = {}

        class FakeLambda:
            def invoke(self, **kw):
                sent.update(kw)

        monkeypatch.setattr(H.boto3, "client", lambda name: FakeLambda())
        monkeypatch.setattr(H, "LAMBDA_FUNCTION_NAME", "pod-pipeline-stg")
        H.invoke_continuation("run-1", "2026-09-10", 2,
                              {"start_date": "2026-09-01", "end_date": "2026-09-02"})
        payload = json.loads(sent["Payload"].decode("utf-8"))
        assert payload["run_id"] == "run-1"
        assert payload["continuation"] == 2
        assert payload["start_date"] == "2026-09-01"
        assert sent["InvocationType"] == "Event"


class TestCoverageMetrics:
    def test_metrics_are_emitted(self, monkeypatch):
        sent = {}

        class FakeCw:
            def put_metric_data(self, **kw):
                sent.update(kw)

        monkeypatch.setattr(H.boto3, "client", lambda name: FakeCw())
        H.emit_coverage(10, 7, 2)
        names = {m["MetricName"]: m["Value"] for m in sent["MetricData"]}
        assert names["ImagesTotal"] == 10
        assert names["ImagesUncovered"] == 1

    def test_a_metrics_failure_never_fails_the_run(self, monkeypatch):
        class Broken:
            def put_metric_data(self, **kw):
                raise RuntimeError("no perms")

        monkeypatch.setattr(H.boto3, "client", lambda name: Broken())
        H.emit_coverage(1, 1, 0)          # must not raise


# --------------------------------------------------------------------------- #
# The boundary itself: no database, anywhere
# --------------------------------------------------------------------------- #

class TestNoDatabaseAccess:
    def test_the_handler_imports_no_database_driver(self):
        import inspect
        source = inspect.getsource(H)
        for forbidden in ("psycopg2", "PG_HOST", "SOURCE_QUERY", "TRIP_QUERY",
                          "RANGE_QUERY", "ALLOW_ADHOC_QUERY"):
            assert forbidden not in source, f"{forbidden} is back in handler.py"

    def test_no_database_driver_is_declared_as_a_dependency(self):
        """The image is built from requirements.txt — that is where a driver
        would have to reappear before it could reach the Lambda."""
        req = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "requirements.txt")
        text = open(req).read().lower()
        assert "psycopg" not in text
        assert "sqlalchemy" not in text

    def test_the_client_is_the_only_route_to_sarathy(self):
        import inspect
        source = inspect.getsource(H)
        assert "from sarathy_client import" in source
        # no raw SQL anywhere in the handler
        for verb in ("select ", "insert into", "on conflict", "create table"):
            assert verb not in source.lower(), f"SQL ({verb!r}) is back in handler.py"


# --------------------------------------------------------------------------- #
# Warm-container state
# --------------------------------------------------------------------------- #

class TestWarmState:
    def test_the_sarathy_client_is_built_once_and_reused(self, monkeypatch):
        monkeypatch.setattr(H, "_sarathy", None)
        monkeypatch.setattr(H, "SARATHY_BASE_URL", "http://sarathy.internal:8080")
        first = H.get_sarathy()
        assert first.base_url == "http://sarathy.internal:8080"
        assert H.get_sarathy() is first

    def test_a_missing_base_url_fails_the_client_loudly(self, monkeypatch):
        monkeypatch.setattr(H, "_sarathy", None)
        monkeypatch.setattr(H, "SARATHY_BASE_URL", "")
        with pytest.raises(SarathyError):
            H.get_sarathy()

    def test_the_download_session_is_built_once_and_reused(self, monkeypatch):
        monkeypatch.setattr(H, "_session", None)
        first = H.build_session(8)
        assert H.build_session(8) is first
        adapter = first.get_adapter("https://example.com")
        assert adapter.max_retries.total == 3

    def test_the_model_is_loaded_once_and_reused(self, monkeypatch, tmp_path):
        """No checkpoint is read twice, however many invocations land here."""
        loads = []

        class FakeModel:
            def load_state_dict(self, state):
                loads.append(state)

            def to(self, device):
                return self

            def eval(self):
                return self

        monkeypatch.setattr(H, "_model", None)
        monkeypatch.setattr(H, "MultiHeadEfficientNet",
                            lambda **kw: FakeModel())
        monkeypatch.setattr(H.torch, "load", lambda *a, **k: {"model_state_dict": {"w": 1}})
        model, device = H.get_model()
        assert loads == [{"w": 1}]
        assert H.get_model()[0] is model
        assert str(device) == "cpu"


# --------------------------------------------------------------------------- #
# Inference wiring, against a stub model
# --------------------------------------------------------------------------- #

class TestScorePrepared:
    class StubModel:
        """Returns fixed logits per head, so the composite is checkable by hand."""

        def __init__(self, logit=0.0):
            self.logit = logit
            self.batch_sizes = []

        def __call__(self, batch):
            import torch as T
            self.batch_sizes.append(batch.shape[0])
            n = batch.shape[0]
            from src.model import ATTRIBUTE_NAMES
            return {name: T.full((n,), self.logit) for name in ATTRIBUTE_NAMES}

    def _prepared(self, n):
        return [{"awb": "A", "trip_id": "1", "pod_link": f"http://img{i}",
                 "chw": np.zeros((3, H.INPUT_SIZE, H.INPUT_SIZE), dtype=np.float32)}
                for i in range(n)]

    def test_every_prepared_image_gets_a_scored_row(self):
        out = H.score_prepared(self.StubModel(), "cpu", self._prepared(3))
        assert len(out) == 3
        assert {r["status"] for r in out} == {"scored"}
        assert [r["pod_link"] for r in out] == ["http://img0", "http://img1", "http://img2"]

    def test_probabilities_are_the_sigmoid_of_the_logits(self):
        out = H.score_prepared(self.StubModel(logit=0.0), "cpu", self._prepared(1))
        assert out[0]["context_valid_prob"] == pytest.approx(0.5, abs=1e-5)
        # weights sum to 1, so an all-0.5 head set gives a 0.5 composite
        assert out[0]["pod_score"] == pytest.approx(0.5, abs=1e-5)

    def test_inference_runs_in_batches(self, monkeypatch):
        monkeypatch.setattr(H, "INFERENCE_BATCH_SIZE", 2)
        model = self.StubModel()
        H.score_prepared(model, "cpu", self._prepared(5))
        assert model.batch_sizes == [2, 2, 1]

    def test_nothing_prepared_means_nothing_scored(self):
        assert H.score_prepared(self.StubModel(), "cpu", []) == []


# --------------------------------------------------------------------------- #
# Real inference wiring (skipped without the checkpoint)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(os.environ.get("MODEL_PATH", "/opt/model/best.pt")),
                    reason="checkpoint not present")
def test_real_model_scores_a_prepared_image():
    model, device = H.get_model()
    prepared = [{"awb": "A", "trip_id": "1", "pod_link": "http://a",
                 "chw": H.preprocess_image(np.zeros((64, 64, 3), dtype=np.uint8))}]
    out = H.score_prepared(model, device, prepared)
    assert len(out) == 1
    assert out[0]["status"] == "scored"
    assert 0.0 <= out[0]["pod_score"] <= 1.0
    for key in ("context_valid_prob", "package_visible_prob",
                "label_readable_prob", "image_clarity_prob"):
        assert 0.0 <= out[0][key] <= 1.0
