"""Containment for filesystem paths that arrive in a request body.

Three endpoints take a path from the caller and open it: `/telemetry/replay`
(`log_path`), `/telemetry/serial/replay` (`capture_path`) and the DBC used by
`/telemetry/coverage` and the replay endpoints (`dbc_path`). Without a check,
any client that can reach the service can ask it to open any file the process
can read - `~/.ssh/id_rsa`, `/etc/passwd`, a `.env` two directories up. The
contents are not echoed back wholesale, but decode statistics and rejection
reasons are, and "does this path exist" is itself an answer the caller should
not get for arbitrary paths.

The fix is a root allowlist, not a pattern blocklist. Blocklists on `..` lose to
symlinks, UNC paths, `%2e%2e`, and on Windows to short 8.3 names; resolving the
path and asking whether the result sits under a permitted root does not.

**Defaults are chosen for what this service actually is**: a local bench and
demo tool. They admit the repository's own data directories and the system
temporary directory, because recording a capture to a temp file and replaying it
is the normal workflow and the test suite does exactly that. They exclude
everything else, which is enough to stop the cases above.

A deployment that exposes this service should replace the defaults:

    BEACON_ALLOWED_DATA_ROOTS=/srv/beacon/data:/srv/beacon/captures

Set that and the repository and temp roots are no longer permitted - the
variable replaces the defaults rather than extending them, so a deployment
cannot accidentally keep a development convenience.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import HTTPException

#: Environment variable that replaces the default roots entirely.
ROOTS_ENV = "BEACON_ALLOWED_DATA_ROOTS"

REPO = Path(__file__).resolve().parents[3]

#: Repository-relative roots a request may reference by default.
_DEFAULT_REPO_ROOTS: tuple[str, ...] = (
    "data",
    "reports",
    "tests",
    "src/bms/io/dbc_examples",
)


def allowed_roots() -> tuple[Path, ...]:
    """Directories a request-supplied path may resolve under."""
    configured = os.environ.get(ROOTS_ENV, "").strip()
    if configured:
        return tuple(
            Path(entry).expanduser().resolve()
            for entry in configured.split(os.pathsep)
            if entry.strip()
        )

    roots = [(REPO / rel).resolve() for rel in _DEFAULT_REPO_ROOTS]
    # Captures are routinely written to a temp file and replayed from there.
    roots.append(Path(tempfile.gettempdir()).resolve())
    return tuple(roots)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_request_path(raw: str, field: str) -> Path:
    """Resolve a caller-supplied path, or refuse and say why.

    Returns the resolved path. Existence is deliberately *not* checked here:
    the caller reports a missing file as 404, and a containment failure must
    surface as 400 before that, so an out-of-bounds path cannot be used to
    probe which files exist.
    """
    if not raw or not raw.strip():
        raise HTTPException(
            status_code=422, detail=f"{field} must not be empty"
        )

    try:
        candidate = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"{field} is not a usable path: {exc}"
        ) from exc

    roots = allowed_roots()
    if any(_is_within(candidate, root) for root in roots):
        return candidate

    raise HTTPException(
        status_code=400,
        detail=(
            f"{field} resolves outside the directories this service is "
            f"permitted to read. Allowed roots: "
            f"{', '.join(str(root) for root in roots)}. "
            f"Set {ROOTS_ENV} to change them. This is a containment refusal, "
            f"not a statement about whether the file exists."
        ),
    )
