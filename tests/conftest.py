"""Shared pytest configuration. Inserts server/ and client/ on sys.path so
`import app.X` resolves correctly when tests reference both packages.

NOTE: server/app and client/app are SEPARATE namespaces. Tests that touch
both must use full paths (server.app.model.X) — done via the conftest
INSERT below which adds repo root to path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so `import server.app.X` and `import client.app.X` both work
