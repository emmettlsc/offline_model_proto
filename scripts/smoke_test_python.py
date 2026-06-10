import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg_path = ROOT / "sample_config.yaml"
    if not cfg_path.exists():
        sys.exit(f"missing: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / "smoke_test_python.json"
    out.write_text(json.dumps({
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "project": (cfg.get("project") or {}).get("name"),
        "config_keys": sorted(cfg.keys()),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
