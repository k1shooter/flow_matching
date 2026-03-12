from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download evaluation assets (e.g., latent/chord encoder checkpoints)."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloaded = []

    for item in manifest.get("files", []):
        url = item["url"]
        rel_path = item["path"]
        expected_sha256 = item.get("sha256")

        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {url} -> {dst}")
        urlretrieve(url, dst)

        if expected_sha256:
            got = sha256sum(dst)
            if got.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"Checksum mismatch for {dst}. expected={expected_sha256}, got={got}"
                )

        downloaded.append(str(dst))

    print("Downloaded files:")
    for x in downloaded:
        print(f"- {x}")


if __name__ == "__main__":
    main()
