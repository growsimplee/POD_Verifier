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
                status_forcelist=[429, 500, 502, 503, 504],
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

    def range_rows(self, start_date: str, end_date: str) -> Iterator[list[dict]]:
        """Page through a date range, yielding one page of rows at a time.

        A generator rather than a list: a wide range is exactly the case where holding every row
        in memory is the thing to avoid, and the caller already scores in windows.
        """
        cursor: Optional[int] = None
        pages = 0
        while True:
            params = {"startDate": start_date, "endDate": end_date, "limit": self.page_size}
            if cursor is not None:
                params["cursor"] = cursor
            page = self._call("GET", "/trips", params=params) or {}

            rows: list[dict] = []
            for item in page.get("items") or []:
                rows.extend(self._rows(item))
            pages += 1
            if rows:
                yield rows

            cursor = page.get("nextCursor")
            if cursor is None:
                logger.info("sarathy: range exhausted after %d page(s)", pages)
                return

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
