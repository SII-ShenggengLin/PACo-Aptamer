#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://zenodo.org/api/deposit/depositions"
ARCHIVE = HERE / "PACoAptamer_research_data_v1.tar.gz"


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def response_json(response, expected):
    if response.status_code not in expected:
        raise RuntimeError(f"Zenodo HTTP {response.status_code}: {response.text[:1000]}")
    return response.json() if response.content else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deposition-id", type=int)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    token = os.environ.get("ZENODO_ACCESS_TOKEN") or os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("Set ZENODO_ACCESS_TOKEN with deposit:write and deposit:actions scopes")
    if not args.archive.is_file():
        raise SystemExit(f"Archive not found: {args.archive}; run build_package.py first")
    headers = {"Authorization": f"Bearer {token}"}
    json_headers = {**headers, "Content-Type": "application/json"}

    if args.deposition_id:
        dep_id = args.deposition_id
        dep = response_json(requests.get(f"{API}/{dep_id}", headers=headers, timeout=60), {200})
    else:
        dep = response_json(requests.post(API, json={}, headers=json_headers, timeout=60), {201})
        dep_id = dep["id"]
    if dep.get("submitted"):
        raise SystemExit(f"Deposition {dep_id} is already published")

    metadata = json.loads((HERE / "zenodo_metadata.json").read_text())
    metadata["metadata"]["prereserve_doi"] = True
    dep = response_json(requests.put(f"{API}/{dep_id}", json=metadata, headers=json_headers, timeout=60), {200})
    print(f"Draft: https://zenodo.org/deposit/{dep_id}", flush=True)
    reserved = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    if reserved:
        print(f"Reserved DOI: {reserved} (not active until publication)", flush=True)

    filename = args.archive.name
    expected_size = args.archive.stat().st_size
    expected_md5 = digest(args.archive, "md5")
    if dep.get("files"):
        if len(dep["files"]) != 1 or dep["files"][0].get("filename") != filename:
            raise SystemExit("Draft contains other files; use a new empty draft or remove them in Zenodo UI")
        print("Archive already present; skipping upload and verifying it", flush=True)
    else:
        bucket = dep["links"]["bucket"]
        uploaded = None
        for attempt in range(1, 6):
            try:
                with args.archive.open("rb") as f:
                    response = requests.put(
                        f"{bucket}/{filename}", data=f, headers=headers, timeout=(30, 600))
                uploaded = response_json(response, {200, 201})
                break
            except (requests.RequestException, RuntimeError) as exc:
                if attempt == 5:
                    raise
                print(f"Upload attempt {attempt} failed: {exc}; retrying", flush=True)
                time.sleep(20 * attempt)
        if uploaded.get("size") != expected_size or uploaded.get("checksum") != f"md5:{expected_md5}":
            raise SystemExit("Upload response checksum/size mismatch; deposition left unpublished")

    dep = response_json(requests.get(f"{API}/{dep_id}", headers=headers, timeout=60), {200})
    files = [f for f in dep.get("files", []) if f.get("filename") == filename]
    if len(files) != 1:
        raise SystemExit("Read-back did not find exactly one uploaded archive; deposition left unpublished")
    remote = files[0]
    if remote.get("filesize") != expected_size or remote.get("checksum") != f"md5:{expected_md5}":
        raise SystemExit("Read-back checksum/size mismatch; deposition left unpublished")
    print(f"Verified upload: {filename}, {expected_size} bytes, md5 {expected_md5}", flush=True)

    if args.publish:
        dep = response_json(requests.post(
            f"{API}/{dep_id}/actions/publish", headers=headers, timeout=120), {200, 201, 202})
        print(json.dumps({
            "id": dep_id,
            "state": dep.get("state"),
            "submitted": dep.get("submitted"),
            "doi": dep.get("doi") or dep.get("metadata", {}).get("doi"),
            "record_url": dep.get("record_url") or dep.get("links", {}).get("html"),
        }, indent=2), flush=True)
    else:
        print("Upload remains an unpublished draft; rerun with --deposition-id and --publish after review")


if __name__ == "__main__":
    main()
