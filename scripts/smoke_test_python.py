"""Confirm Python runs, can read sample_config.yaml, and can write to outputs/."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    config_path = REPO_ROOT / "sample_config.yaml"
    if not config_path.exists():
        print(f"ERROR: missing config file: {config_path}", file=sys.stderr)
        return 1

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        print(f"ERROR: PyYAML not installed: {exc}", file=sys.stderr)
        print("Install: pip install pyyaml", file=sys.stderr)
        return 2

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "ok": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "project_name": (cfg.get("project") or {}).get("name"),
        "config_top_level_keys": sorted(cfg.keys()),
    }

    out_path = output_dir / "smoke_test_python.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
