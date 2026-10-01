"""Verify that the configured model artifacts are present and readable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON manifest: {path}: {exc}") from exc


def verify(root: Path, manifest: dict) -> list[str]:
    missing = []
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(f"{artifact['name']}: {artifact['path']}")
    return missing


def checksums(root: Path, manifest: dict) -> dict:
    result = {
        "manifest_schema_version": manifest.get("schema_version"),
        "default_department_variant": manifest.get("default_department_variant"),
        "artifacts": [],
    }
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["artifacts"].append(
            {
                "name": artifact["name"],
                "path": artifact["path"],
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check model files required by the default API configuration."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project or container root containing the models directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "config"
        / "model_artifacts.json",
        help="Artifact manifest JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failure exit code when artifacts are missing.",
    )
    parser.add_argument(
        "--write-checksums",
        type=Path,
        help="Write SHA-256 checksums for present artifacts to this JSON file.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    missing = verify(args.root, manifest)
    if args.write_checksums:
        args.write_checksums.parent.mkdir(parents=True, exist_ok=True)
        args.write_checksums.write_text(
            json.dumps(checksums(args.root, manifest), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote artifact checksums to {args.write_checksums}")
    if missing:
        print("Missing model artifacts:")
        print("\n".join(f"- {item}" for item in missing))
        try:
            manifest_name = args.manifest.relative_to(args.root)
        except ValueError:
            manifest_name = args.manifest
        print(
            "Provision artifacts with the commands in "
            f"{manifest_name}."
        )
        return 1 if args.strict else 0

    print(
        "Model artifacts verified for "
        f"{manifest.get('default_department_variant', 'default')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
