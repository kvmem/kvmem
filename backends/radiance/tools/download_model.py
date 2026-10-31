"""Download explicit files from a pinned HF revision; resume and verify, never execute repo code."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = "https://huggingface.co"
    api = f"{base}/api/models/{args.repo}/revision/{args.revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != args.revision:
        parser.error("use an exact immutable repository revision")
    names = set(args.file)
    entries = {item["rfilename"]: item for item in metadata["siblings"] if item["rfilename"] in names}
    if entries.keys() != names:
        parser.error("requested file missing from the pinned revision")
    args.output.mkdir(parents=True, exist_ok=True)
    destinations = {}
    needed = 0
    root = args.output.resolve()
    for name, item in entries.items():
        rel = PurePosixPath(name)
        path = root.joinpath(*rel.parts).resolve()
        if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(root):
            parser.error("unsafe file path")
        path.parent.mkdir(parents=True, exist_ok=True)
        present = path.stat().st_size if path.exists() else 0
        if present > item["size"]:
            raise RuntimeError(f"existing file exceeds expected size; preserve and inspect: {path}")
        needed += item["size"] - present
        destinations[name] = path
    if shutil.disk_usage(root).free < needed + 1024**3:
        raise RuntimeError("not enough free space for remaining bytes plus 1 GiB; preserving files")
    manifest = {"repo": args.repo, "revision": args.revision, "files": []}
    for name, item in entries.items():
        path = destinations[name]
        for attempt in range(8):
            if path.exists() and path.stat().st_size == item["size"]:
                break
            print(f"download {name}, attempt {attempt + 1}, expected bytes {item['size']}", flush=True)
            curl = "curl.exe" if os.name == "nt" else "curl"
            # Let this outer loop retry. curl --retry can truncate back to the
            # offset at the start of its invocation, losing bytes downloaded
            # during a timed-out attempt. A fresh -C - recomputes the offset.
            subprocess.run([curl, "-f", "-L", "-C", "-", "--connect-timeout", "30",
                            "--max-time", "1800", "--speed-limit", "1024", "--speed-time", "60",
                            "-o", str(path),
                            f"{base}/{args.repo}/resolve/{args.revision}/{urllib.parse.quote(name)}"], check=False)
        if not path.exists() or path.stat().st_size != item["size"]:
            raise RuntimeError(f"incomplete download preserved: {path}")
        print(f"hashing {name}", flush=True)
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = (item.get("lfs") or {}).get("sha256")
        if expected and sha != expected:
            raise RuntimeError(f"hash mismatch, preserve for inspection: {path}")
        manifest["files"].append({"file": name, "bytes": item["size"], "sha256": sha})
        print(json.dumps(manifest["files"][-1]), flush=True)
    (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("all requested files verified", flush=True)


if __name__ == "__main__":
    main()
