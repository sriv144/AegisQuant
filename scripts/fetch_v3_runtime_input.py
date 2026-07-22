"""Fetch one content-addressed, self-contained v3 runtime input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


MAX_BUNDLE_BYTES = 50 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, code, "redirects are disabled", headers, fp)


def fetch(*, url: str, expected_sha256: str, output: Path) -> str:
    """Download, verify, validate, and atomically persist a frozen bundle."""

    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("bundle URL must be credential-free HTTPS")
    digest = expected_sha256.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("bundle SHA-256 must be 64 lowercase hexadecimal characters")

    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "AegisQuant-v3-input/1"},
        method="GET",
    )
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"bundle download returned HTTP {response.status}")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > MAX_BUNDLE_BYTES:
            raise ValueError("bundle exceeds the 50 MiB limit")
        payload = response.read(MAX_BUNDLE_BYTES + 1)
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError("bundle exceeds the 50 MiB limit")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest:
        raise ValueError("bundle SHA-256 mismatch")

    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("bundle root must be a JSON object")
    if "total_return_prices_csv" in document:
        raise ValueError("workflow bundles must inline total-return prices")
    if "total_return_prices" not in document:
        raise ValueError("bundle is missing inline total-return prices")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(fetch(url=arguments.url, expected_sha256=arguments.sha256, output=arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
