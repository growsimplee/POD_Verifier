"""
Tests for the sarathy API client — the scorer's only route to any database.

The client is deliberately thin, so what is worth pinning down is the contract
it assumes on sarathy's side: the ServiceResponse envelope, the camelCase field
names, cursor paging, and the fact that every failure surfaces as SarathyError
rather than as a bare requests exception the handler would not catch.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sarathy_client as SC  # noqa: E402
from sarathy_client import SarathyClient, SarathyError  # noqa: E402


class FakeResp:
    def __init__(self, payload=None, status=200, text="", raw=None):
        self._payload = payload
        self.status_code = status
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self._raw = raw

    def json(self):
        if self._raw is not None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Replays queued responses and records every request made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def mount(self, prefix, adapter):
        pass

    def request(self, method, url, timeout=None, **kw):
        self.calls.append({"method": method, "url": url, "timeout": timeout, **kw})
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def client(responses, **kw):
    c = SarathyClient("http://sarathy.internal:8080/", **kw)
    c._session = FakeSession(responses)
    return c


def envelope(data):
    return FakeResp({"message": "ok", "data": data})


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_an_empty_base_url_is_refused_up_front(self):
        with pytest.raises(SarathyError, match="SARATHY_BASE_URL"):
            SarathyClient("")

    def test_a_trailing_slash_is_normalised_away(self):
        assert SarathyClient("http://x:8080/").base_url == "http://x:8080"

    def test_retries_are_configured_for_both_verbs(self):
        c = SarathyClient("http://x:8080", retries=5)
        adapter = c._session.get_adapter("http://x:8080")
        assert adapter.max_retries.total == 5
        assert set(adapter.max_retries.allowed_methods) >= {"GET", "POST"}


# --------------------------------------------------------------------------- #
# Envelope + error handling
# --------------------------------------------------------------------------- #

class TestCall:
    def test_the_service_response_envelope_is_unwrapped(self):
        c = client([envelope({"awb": "A"})])
        assert c._call("GET", "/trips/1") == {"awb": "A"}

    def test_a_bare_body_is_passed_through(self):
        c = client([FakeResp({"awb": "A"})])
        assert c._call("GET", "/trips/1") == {"awb": "A"}

    def test_the_path_is_namespaced_internal(self):
        c = client([envelope({})])
        c._call("GET", "/trips/1")
        assert c._session.calls[0]["url"] == \
            "http://sarathy.internal:8080/internal/pod-scoring/trips/1"

    def test_the_timeout_is_applied(self):
        c = client([envelope({})], timeout=7)
        c._call("GET", "/trips/1")
        assert c._session.calls[0]["timeout"] == 7

    def test_an_http_error_becomes_a_sarathy_error(self):
        c = client([FakeResp(status=500, text="boom")])
        with pytest.raises(SarathyError, match="500"):
            c._call("GET", "/trips/1")

    def test_a_transport_error_becomes_a_sarathy_error(self):
        c = client([SC.requests.ConnectionError("no route")])
        with pytest.raises(SarathyError, match="ConnectionError"):
            c._call("GET", "/trips/1")

    def test_a_non_json_body_becomes_a_sarathy_error(self):
        c = client([FakeResp(raw="<html>", text="<html>")])
        with pytest.raises(SarathyError, match="non-JSON"):
            c._call("GET", "/trips/1")


# --------------------------------------------------------------------------- #
# Row shaping
# --------------------------------------------------------------------------- #

class TestRowShaping:
    def test_one_row_per_link(self):
        rows = SarathyClient._rows(
            {"awb": "AWB1", "tripId": 5, "podLinks": ["http://a", "http://b"]})
        assert [r["pod_link"] for r in rows] == ["http://a", "http://b"]
        assert all(r["awb"] == "AWB1" and r["trip_id"] == "5" for r in rows)

    def test_already_scored_links_are_flagged(self):
        rows = SarathyClient._rows({
            "awb": "AWB1", "tripId": 5,
            "podLinks": ["http://a", "http://b"],
            "alreadyScoredLinks": ["http://a"],
        })
        assert [r["already_scored"] for r in rows] == [True, False]

    def test_a_missing_awb_falls_back_to_a_trip_placeholder(self):
        rows = SarathyClient._rows({"tripId": 5, "podLinks": ["http://a"]})
        assert rows[0]["awb"] == "TRIP-5"

    def test_no_links_means_no_rows(self):
        assert SarathyClient._rows({"awb": "A", "tripId": 5}) == []

    def test_the_trip_id_override_wins(self):
        rows = SarathyClient._rows(
            {"awb": "A", "tripId": 5, "podLinks": ["http://a"]}, trip_id_override="9")
        assert rows[0]["trip_id"] == "9"


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #

class TestTripRows:
    def test_a_trip_is_expanded_into_rows(self):
        c = client([envelope({"awb": "AWB1", "tripId": 5,
                              "podLinks": ["http://a"], "alreadyScoredLinks": []})])
        rows = c.trip_rows("5")
        assert len(rows) == 1
        assert rows[0]["awb"] == "AWB1"

    def test_an_unknown_trip_yields_nothing(self):
        c = client([envelope(None)])
        assert c.trip_rows("999") == []


class TestRangeRows:
    def test_a_single_page_is_yielded_and_paging_stops(self):
        c = client([envelope({"items": [
            {"awb": "A", "tripId": 1, "podLinks": ["http://a"]}], "nextCursor": None})])
        pages = list(c.range_rows("2026-09-01", "2026-09-02"))
        assert len(pages) == 1 and len(pages[0]) == 1
        assert len(c._session.calls) == 1

    def test_the_cursor_is_carried_into_the_next_request(self):
        c = client([
            envelope({"items": [{"awb": "A", "tripId": 1, "podLinks": ["http://a"]}],
                      "nextCursor": 42}),
            envelope({"items": [{"awb": "B", "tripId": 2, "podLinks": ["http://b"]}],
                      "nextCursor": None}),
        ])
        pages = list(c.range_rows("2026-09-01", "2026-09-02"))
        assert [r["pod_link"] for page in pages for r in page] == ["http://a", "http://b"]
        assert "cursor" not in c._session.calls[0]["params"]
        assert c._session.calls[1]["params"]["cursor"] == 42

    def test_the_date_range_and_page_size_are_sent(self):
        c = client([envelope({"items": [], "nextCursor": None})], page_size=250)
        list(c.range_rows("2026-09-01", "2026-09-02"))
        params = c._session.calls[0]["params"]
        assert params["startDate"] == "2026-09-01"
        assert params["endDate"] == "2026-09-02"
        assert params["limit"] == 250

    def test_an_empty_page_is_not_yielded(self):
        c = client([envelope({"items": [], "nextCursor": None})])
        assert list(c.range_rows("2026-09-01", "2026-09-02")) == []

    def test_a_trip_with_no_links_contributes_no_rows(self):
        c = client([envelope({"items": [{"awb": "A", "tripId": 1, "podLinks": []}],
                              "nextCursor": None})])
        assert list(c.range_rows("2026-09-01", "2026-09-02")) == []


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #

class TestWriteScores:
    def test_nothing_to_write_makes_no_call(self):
        c = client([])
        assert c.write_scores("2026-09-10", []) == 0
        assert c._session.calls == []

    def test_rows_are_translated_to_the_api_field_names(self):
        c = client([envelope(1)])
        c.write_scores("2026-09-10", [{
            "awb": "AWB1", "trip_id": "5", "pod_link": "http://a",
            "status": "scored", "failure_reason": None, "pod_score": 0.91,
            "context_valid_prob": 0.8, "package_visible_prob": 0.7,
            "label_readable_prob": 0.6, "image_clarity_prob": 0.5,
        }])
        payload = c._session.calls[0]["json"]
        assert payload["runDate"] == "2026-09-10"
        result = payload["results"][0]
        assert result["tripId"] == "5"
        assert result["podLink"] == "http://a"
        assert result["podScore"] == 0.91
        assert result["labelReadableProb"] == 0.6

    def test_a_failure_row_is_written_with_its_reason(self):
        c = client([envelope(1)])
        c.write_scores("2026-09-10", [{
            "awb": "AWB1", "trip_id": "5", "pod_link": "http://a",
            "status": "download_failed", "failure_reason": "http_404",
        }])
        result = c._session.calls[0]["json"]["results"][0]
        assert result["status"] == "download_failed"
        assert result["failureReason"] == "http_404"
        assert result["podScore"] is None

    def test_the_written_count_comes_back_from_sarathy(self):
        c = client([envelope(3)])
        rows = [{"awb": "A", "trip_id": "1", "pod_link": f"http://{i}",
                 "status": "scored"} for i in range(3)]
        assert c.write_scores("2026-09-10", rows) == 3

    def test_a_non_numeric_answer_falls_back_to_the_row_count(self):
        c = client([envelope({"ok": True})])
        rows = [{"awb": "A", "trip_id": "1", "pod_link": "http://a", "status": "scored"}]
        assert c.write_scores("2026-09-10", rows) == 1

    def test_a_rejected_write_raises(self):
        c = client([FakeResp(status=400, text="bad request")])
        rows = [{"awb": "A", "trip_id": "1", "pod_link": "http://a", "status": "scored"}]
        with pytest.raises(SarathyError, match="400"):
            c.write_scores("2026-09-10", rows)
