#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://zenodo.org/api/deposit/depositions"


def checksum(value):
    return (value or "").removeprefix("md5:").strip('"')


def remote_matches(dep_id, item, headers):
    try:
        response = requests.get(f"{API}/{dep_id}", headers=headers, timeout=60)
        if response.status_code != 200:
            return False
        remote = {f["filename"]: f for f in response.json().get("files", [])}.get(item["filename"])
        return bool(remote and remote.get("filesize") == item["size"]
                    and checksum(remote.get("checksum")) == item["md5"])
    except requests.RequestException:
        return False


def upload(dep_id, bucket, item, headers):
    path = HERE / "upload_parts" / item["filename"]
    for attempt in range(1, 6):
        if attempt > 1 and remote_matches(dep_id, item, headers):
            print(f"Remote verification accepted {item['filename']} without reupload", flush=True)
            return item["filename"], item["size"]
        try:
            data = path.read_bytes()
            response = requests.put(
                f"{bucket}/{item['filename']}", data=data, headers=headers,
                timeout=(30, 240))
            if response.status_code not in (200, 201):
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            result = response.json()
            if result.get("size") != item["size"] or checksum(result.get("checksum")) != item["md5"]:
                raise RuntimeError("upload response size/MD5 mismatch")
            return item["filename"], item["size"]
        except (requests.RequestException, RuntimeError) as exc:
            print(f"{item['filename']} attempt {attempt} issue: {exc}", flush=True)
            if remote_matches(dep_id, item, headers):
                print(f"Remote verification accepted {item['filename']} after response issue", flush=True)
                return item["filename"], item["size"]
            if attempt == 5:
                raise RuntimeError(f"{item['filename']} failed after 5 attempts: {exc}") from exc
            time.sleep(min(60, attempt * 5))
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deposition-id", type=int, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    token = os.environ.get("ZENODO_ACCESS_TOKEN") or os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("Set ZENODO_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    json_headers = {**headers, "Content-Type": "application/json"}
    directory = HERE / "upload_parts"
    manifest = json.loads((directory / "upload_manifest.json").read_text())
    items = list(manifest["parts"])
    for name in ("README.md", "upload_manifest.json"):
        path = directory / name
        data = path.read_bytes()
        items.append({"filename": name, "size": len(data), "md5": hashlib.md5(data).hexdigest()})

    dep = requests.get(f"{API}/{args.deposition_id}", headers=headers, timeout=60)
    if dep.status_code != 200:
        raise SystemExit(f"Zenodo HTTP {dep.status_code}: {dep.text[:1000]}")
    dep = dep.json()
    if dep.get("submitted"):
        raise SystemExit("Deposition is already published")
    metadata = json.loads((HERE / "zenodo_metadata.json").read_text())
    metadata["metadata"]["prereserve_doi"] = True
    response = requests.put(f"{API}/{args.deposition_id}", json=metadata, headers=json_headers, timeout=60)
    if response.status_code != 200:
        raise SystemExit(f"Metadata update failed: HTTP {response.status_code}: {response.text[:1000]}")

    existing = {f["filename"]: f for f in dep.get("files", [])}
    expected_names = {item["filename"] for item in items}
    if not set(existing).issubset(expected_names):
        raise SystemExit("Draft contains unexpected files; inspect it before continuing")
    pending = []
    completed_bytes = 0
    for item in items:
        remote = existing.get(item["filename"])
        if remote:
            if remote.get("filesize") != item["size"] or checksum(remote.get("checksum")) != item["md5"]:
                raise SystemExit(f"Existing file mismatch: {item['filename']}; remove it before retrying")
            completed_bytes += item["size"]
        else:
            pending.append(item)
    print(f"Files already verified: {len(items) - len(pending)}/{len(items)}", flush=True)

    bucket = dep["links"]["bucket"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, args.deposition_id, bucket, item, headers) for item in pending]
        for future in as_completed(futures):
            name, size = future.result()
            completed_bytes += size
            print(f"Uploaded {name}; {completed_bytes}/{sum(i['size'] for i in items)} bytes verified", flush=True)

    dep = requests.get(f"{API}/{args.deposition_id}", headers=headers, timeout=60).json()
    remote_files = {f["filename"]: f for f in dep.get("files", [])}
    if set(remote_files) != expected_names:
        raise SystemExit("Read-back file list mismatch; deposition left unpublished")
    for item in items:
        remote = remote_files[item["filename"]]
        if remote.get("filesize") != item["size"] or checksum(remote.get("checksum")) != item["md5"]:
            raise SystemExit(f"Read-back mismatch: {item['filename']}; deposition left unpublished")
    print(f"Verified all {len(items)} public-package files", flush=True)

    if args.publish:
        response = requests.post(
            f"{API}/{args.deposition_id}/actions/publish", headers=headers, timeout=120)
        if response.status_code not in (200, 201, 202):
            raise SystemExit(f"Publish failed: HTTP {response.status_code}: {response.text[:1000]}")
        dep = response.json()
        print(json.dumps({
            "id": args.deposition_id,
            "state": dep.get("state"),
            "submitted": dep.get("submitted"),
            "doi": dep.get("doi") or dep.get("metadata", {}).get("doi"),
            "record_url": dep.get("record_url") or dep.get("links", {}).get("html"),
        }, indent=2), flush=True)
    else:
        print(f"Verified unpublished draft: https://zenodo.org/deposit/{args.deposition_id}")


if __name__ == "__main__":
    main()
