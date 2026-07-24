"""robots.txt enforcement.

The crawler calls `allowed(url)` before every fetch. Rules are fetched once per
origin and cached. Failure policy follows the spirit of RFC 9309:

  - 2xx robots.txt: obey its rules.
  - 4xx (typically 404, no robots.txt): allow everything.
  - 5xx or network error: disallow. If a site's robots endpoint is failing we
    do not get to assume permission — the polite default is to back off.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from src.ingest.http import UA


@lru_cache(maxsize=64)
def _rules(origin: str) -> RobotFileParser:
    rp = RobotFileParser()
    try:
        resp = httpx.get(
            f"{origin}/robots.txt",
            headers={"User-Agent": UA},
            timeout=15.0,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        rp.disallow_all = True  # unreachable robots.txt -> back off
        return rp

    if resp.status_code >= 500:
        rp.disallow_all = True
    elif resp.status_code >= 400:
        rp.parse([])            # no robots.txt -> nothing disallowed
    else:
        rp.parse(resp.text.splitlines())
    return rp


def allowed(url: str, agent: str = UA) -> bool:
    """True if `agent` may fetch `url` under the origin's robots.txt."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return _rules(origin).can_fetch(agent, url)
