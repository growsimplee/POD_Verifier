"""HTTP client for sarathy's /internal/pod-scoring API.

The scorer used to connect straight to sarathy's Postgres — holding its credentials, creating its
own tables and writing rows. Two services owning one service's schema is the problem this module
closes: sarathy is now the only writer, and this is the whole of our access to it.

Reachable only inside the VPC over the internal NLB. There is no token: the endpoints are
namespaced /internal and protected by network isolation, matching the rest of sarathy's
service-to-service surface.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import requests

logger = logging.getLogger()


class SarathyError(RuntimeError):
    """Sarathy could not answer. Callers decide whether that is fatal."""


class SarathyClient:
    """Thin, retrying wrapper. One instance per warm container."""

    def __init__(self, base_url: str, timeout: int = 15, retries: int = 3,
                 page_size: int = 500) -> None:
        if not base_url:
            raise SarathyError("SARATHY_BASE_URL is not configured")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.page_size = page_size

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=retries, backoff_factor=0.5,
                # 500 is deliberately NOT retried. Sarathy returns it for a query that failed on its
                # own terms -- a statement timeout, say -- and repeating that query repeats the
                # failure while charging its full cost to the database again. A batch read that
                # timed out once turned into four identical timeouts here, every 30 minutes, until
                # the query behind it was fixed. The codes left are the ones that mean "try again":
                # rate limiting and the load balancer failing to reach an instance.
                status_forcelist=[429, 502, 503, 504],
                allowed_methods=["GET", "POST"],   # the write is idempotent, so retrying is safe
            ),
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _call(self, method: str, path: str, **kw: Any) -> Any:
        url = f"{self.base_url}/internal/pod-scoring{path}"
        try:
            resp = self._session.request(method, url, timeout=self.timeout, **kw)
        except requests.RequestException as e:
            raise SarathyError(f"{method} {path} failed: {type(e).__name__}: {e}") from e

        if resp.status_code >= 400:
            raise SarathyError(f"{method} {path} returned {resp.status_code}: {resp.text[:300]}")
        try:
            body = resp.json()
        except ValueError as e:
            raise SarathyError(f"{method} {path} returned non-JSON") from e

        # ServiceResponse wraps everything as {"message": ..., "data": ...}
        return body.get("data") if isinstance(body, dict) and "data" in body else body

    @staticmethod
    def _rows(item: dict, trip_id_override: Optional[str] = None) -> list[dict]:
        """One API item -> the row shape the scorer works in."""
        awb = str(item.get("awb") or "").strip()
        trip_id = str(trip_id_override if trip_id_override is not None else item.get("tripId") or "")
        already = set(item.get("alreadyScoredLinks") or [])
        return [
            {"awb": awb or f"TRIP-{trip_id}", "trip_id": trip_id,
             "pod_link": link, "already_scored": link in already}
            for link in (item.get("podLinks") or [])
        ]

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def trip_rows(self, trip_id: str) -> list[dict]:
        """The POD images for one trip, each flagged with whether it is already scored."""
        item = self._call("GET", f"/trips/{trip_id}")
        return self._rows(item or {}, trip_id_override=trip_id)

    def _paged(self, path: str, params: dict, label: str) -> Iterator[list[dict]]:
        """Walk a keyset-paginated endpoint, yielding one page of API items at a time.

        A generator rather than a list: a wide window is exactly the case where holding every row
        in memory is the thing to avoid, and the caller already scores in windows.

        The cursor is an opaque token — echo sarathy's ``nextCursor`` back verbatim and stop when
        it comes back null. It used to be a trip id and is now a (timestamp, id) pair; treating it
        as opaque is why that change did not reach this file. The same token shape serves the trip
        feed and both retry feeds, so this one loop covers all three.
        """
        cursor: Optional[str] = None
        pages = 0
        while True:
            page_params = dict(params, limit=self.page_size)
            if cursor is not None:
                page_params["cursor"] = cursor
            page = self._call("GET", path, params=page_params) or {}

            items = page.get("items") or []
            pages += 1
            if items:
                yield items

            cursor = page.get("nextCursor")
            if cursor is None:
                logger.info("sarathy: %s exhausted after %d page(s)", label, pages)
                return

    def range_rows(self, start_date: str, end_date: str) -> Iterator[list[dict]]:
        """Page through a date range — whole days, end inclusive. Used by manual backfills."""
        for items in self._paged("/trips",
                                 {"startDate": start_date, "endDate": end_date},
                                 f"range {start_date}..{end_date}"):
            rows: list[dict] = []
            for item in items:
                rows.extend(self._rows(item))
            if rows:
                yield rows

    def window_rows(self, from_iso: str, to_iso: str) -> Iterator[list[dict]]:
        """Page through an exact instant window, upper bound exclusive.

        What the half-hourly sweep uses. Days are the wrong unit for it: a run at 23:30 asking for
        "today" covers to 23:30, and every trip completing between then and midnight would fall
        into no run at all, because the next day's runs look at the next day's timestamps. Asking
        for the last N hours instead has no such seam.
        """
        for items in self._paged("/trips", {"from": from_iso, "to": to_iso},
                                 f"window {from_iso}..{to_iso}"):
            rows: list[dict] = []
            for item in items:
                rows.extend(self._rows(item))
            if rows:
                yield rows

    def retry_rows(self, force: bool = False,
                   since: Optional[str] = None,
                   until: Optional[str] = None) -> Iterator[list[dict]]:
        """Page through POD links whose download failed and which are due another attempt.

        These are mostly not broken links. sarathy records the presigned URLs the rider's app
        announces when it calls /app/save-info, before the image has finished uploading, so a
        failure usually means the photo had not landed in S3 yet. Sarathy owns the schedule — it
        decides which links are due — and this just walks what it hands back.

        ``force`` abandons that schedule and asks for every still-unscored failure in an explicit
        window, for links the scheduled sweep can no longer reach.
        """
        params: dict = {}
        if force:
            params["force"] = "true"
            if since:
                params["since"] = since
            if until:
                params["until"] = until

        for items in self._paged("/retry-links", params,
                                 "forced retry set" if force else "retry set"):
            rows = [
                {"awb": str(item.get("awb") or "").strip() or f"TRIP-{item.get('tripId')}",
                 "trip_id": str(item.get("tripId") or ""),
                 "pod_link": item.get("podLink"),
                 "attempts": int(item.get("attempts") or 0),
                 "already_scored": False}
                for item in items
                if item.get("podLink")
            ]
            if rows:
                yield rows

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def write_scores(self, run_date: str, results: list[dict]) -> int:
        """Post a batch of scores. Idempotent on (awb, podLink, runDate)."""
        if not results:
            return 0
        payload = {
            "runDate": run_date,
            "results": [
                {
                    "awb": r["awb"],
                    "tripId": r.get("trip_id"),
                    "podLink": r["pod_link"],
                    "status": r["status"],
                    "failureReason": r.get("failure_reason"),
                    "podScore": r.get("pod_score"),
                    "contextValidProb": r.get("context_valid_prob"),
                    "packageVisibleProb": r.get("package_visible_prob"),
                    "labelReadableProb": r.get("label_readable_prob"),
                    "imageClarityProb": r.get("image_clarity_prob"),
                }
                for r in results
            ],
        }
        written = self._call("POST", "/scores", json=payload)
        return int(written) if isinstance(written, int) else len(results)
