#!/usr/bin/env python3
"""Assert that a Lambda invocation returned an expected status.

    check_lambda_response.py <response.json> <expected-status>

`aws lambda invoke` writes the function's return value to a file. Our handlers
return {"statusCode": …, "body": "<json string>"} — note that `body` is a JSON
*string*, so the file on disk contains escaped quotes:

    {"statusCode": 200, "body": "{\\"status\\": \\"applied\\", …}"}

which means grepping it for the unescaped `"status": "applied"` never matches,
even on a completely successful run. Parse it instead.

Also distinguishes the two failure shapes that look alike from the outside: a
handler that ran and reported a problem, versus a Lambda-level error where
there is no body at all (timeout, unhandled exception, init failure).
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    path, want = argv[1], argv[2]

    try:
        raw = open(path, encoding="utf-8").read()
    except OSError as e:
        print(f"Could not read {path}: {e}")
        return 1

    print(raw.strip() or "(empty response)")

    try:
        resp = json.loads(raw)
    except ValueError:
        print(f"FAILED — response was not JSON.")
        return 1

    # Unhandled exception / timeout: the payload is an error object, no body.
    if isinstance(resp, dict) and "errorMessage" in resp:
        print(f"FAILED — the function raised "
              f"{resp.get('errorType', 'an error')}: {resp['errorMessage']}")
        for line in (resp.get("stackTrace") or [])[:10]:
            print(f"    {line}".rstrip())
        return 1

    body = resp.get("body") if isinstance(resp, dict) else None
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            print("FAILED — body was not JSON.")
            return 1
    if not isinstance(body, dict):
        body = {}

    got = body.get("status")
    if got != want:
        detail = body.get("error") or body.get("message")
        print(f"FAILED — expected status={want!r}, got {got!r}."
              + (f" ({detail})" if detail else ""))
        return 1

    print(f"OK — status={got}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
