"""The web console is a single embedded HTML document.

Guards against a class of bug that breaks the entire page: a JavaScript syntax
error anywhere in the embedded <script> block stops the browser from running
*any* of it, which looks exactly like "nothing works".
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from weft.api.console import INDEX_HTML

NODE = shutil.which("node")


def _script() -> str:
    m = re.search(r"(?s)<script>(.*)</script>", INDEX_HTML)
    assert m, "console must contain a <script> block"
    return m.group(1)


def test_console_is_html_and_has_one_script_block():
    assert INDEX_HTML.lstrip().lower().startswith("<!doctype html>")
    assert INDEX_HTML.count("<script>") == 1
    assert len(_script()) > 0


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_embedded_script_parses():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "console.js"
        p.write_text(_script(), encoding="utf-8")
        res = subprocess.run([NODE, "--check", str(p)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
