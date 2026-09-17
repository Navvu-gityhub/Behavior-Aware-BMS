"""Three endpoints open a path the caller supplies. This is what stops them.

The property under test is not "rejects strings containing `..`" - that is a
blocklist, and blocklists on paths lose to symlinks, UNC paths and Windows 8.3
short names. The property is that the *resolved* path sits under a permitted
root, which is decided after resolution and therefore after any such trick has
already been applied.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.api.paths import (  # noqa: E402
    REPO,
    ROOTS_ENV,
    allowed_roots,
    resolve_request_path,
)


def test_a_path_inside_the_repository_data_root_is_allowed():
    target = REPO / "tests" / "fixtures" / "beacon_rig_capture.txt"
    assert resolve_request_path(str(target), "capture_path") == target.resolve()


def test_a_path_in_the_temp_directory_is_allowed():
    """Recording a capture to a temp file and replaying it is the normal
    bench workflow, and the test suite does exactly that."""
    target = Path(tempfile.gettempdir()) / "beacon_capture.txt"
    assert resolve_request_path(str(target), "capture_path") == target.resolve()


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "/etc/shadow",
        "C:/Windows/win.ini",
        "~/.ssh/id_rsa",
        "~/.aws/credentials",
    ],
)
def test_sensitive_host_paths_are_refused(hostile):
    with pytest.raises(HTTPException) as excinfo:
        resolve_request_path(hostile, "log_path")
    assert excinfo.value.status_code == 400


def test_traversal_out_of_an_allowed_root_is_refused_after_resolution():
    """`data/../../..` resolves above the repository and must not be admitted.

    Checked post-resolution rather than by pattern, so an equivalent path
    expressed some other way is caught by the same rule.
    """
    escape = str(REPO / "data" / ".." / ".." / ".." / "secrets.txt")
    with pytest.raises(HTTPException) as excinfo:
        resolve_request_path(escape, "log_path")
    assert excinfo.value.status_code == 400


def test_traversal_that_lands_back_inside_a_root_is_allowed():
    """Containment is about where the path ends up, not how it is written."""
    circuitous = str(REPO / "data" / ".." / "tests" / "fixtures")
    assert resolve_request_path(circuitous, "capture_path") == (
        REPO / "tests" / "fixtures"
    ).resolve()


def test_an_empty_path_is_a_422_not_a_containment_error():
    with pytest.raises(HTTPException) as excinfo:
        resolve_request_path("   ", "log_path")
    assert excinfo.value.status_code == 422


def test_the_refusal_names_the_field_and_the_allowed_roots():
    with pytest.raises(HTTPException) as excinfo:
        resolve_request_path("/etc/passwd", "capture_path")
    detail = excinfo.value.detail
    assert "capture_path" in detail
    assert ROOTS_ENV in detail
    assert "not a statement about whether the file exists" in detail


def test_the_environment_variable_replaces_the_defaults_rather_than_extending(
    monkeypatch, tmp_path
):
    """A deployment must not silently keep a development convenience.

    If the variable extended the defaults, setting it on a production host
    would leave the repository and temp roots readable as well.
    """
    allowed = tmp_path / "captures"
    allowed.mkdir()
    monkeypatch.setenv(ROOTS_ENV, str(allowed))

    assert allowed_roots() == (allowed.resolve(),)
    assert resolve_request_path(str(allowed / "run.txt"), "capture_path")

    with pytest.raises(HTTPException) as excinfo:
        resolve_request_path(
            str(REPO / "tests" / "fixtures" / "beacon_rig_capture.txt"),
            "capture_path",
        )
    assert excinfo.value.status_code == 400


def test_multiple_roots_can_be_configured(monkeypatch, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(ROOTS_ENV, os.pathsep.join([str(first), str(second)]))

    assert resolve_request_path(str(first / "x.txt"), "log_path")
    assert resolve_request_path(str(second / "y.txt"), "log_path")
