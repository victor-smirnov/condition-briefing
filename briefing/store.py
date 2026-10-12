"""Run storage. Local directory for docker compose, GCS bucket for Cloud Run.

Layout (same for both backends):
    <root>/<run_id>/run.json
    <root>/<run_id>/steps/<NN>_<step>.json
"""

import json
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class Store(Protocol):
    def read(self, key: str) -> dict | None: ...
    def write(self, key: str, obj: dict) -> None: ...
    def list_runs(self) -> list[str]: ...


class LocalStore:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, key: str) -> dict | None:
        p = self.root / key
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, key: str, obj: dict) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        tmp.replace(p)

    def list_runs(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/run.json"))


class GCSStore:
    def __init__(self, bucket: str, prefix: str) -> None:
        import os

        from google.cloud import storage

        if os.getenv("STORAGE_EMULATOR_HOST"):  # local fake-gcs-server
            from google.auth.credentials import AnonymousCredentials

            client = storage.Client(project="local", credentials=AnonymousCredentials())
            if not client.bucket(bucket).exists():
                client.create_bucket(bucket)
        else:  # Cloud Run: service account via ADC
            client = storage.Client()
        self.bucket = client.bucket(bucket)
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""

    def read(self, key: str) -> dict | None:
        blob = self.bucket.blob(self.prefix + key)
        return json.loads(blob.download_as_text()) if blob.exists() else None

    def write(self, key: str, obj: dict) -> None:
        self.bucket.blob(self.prefix + key).upload_from_string(
            json.dumps(obj, indent=2, ensure_ascii=False), content_type="application/json"
        )

    def list_runs(self) -> list[str]:
        it = self.bucket.client.list_blobs(self.bucket, prefix=self.prefix, delimiter="/")
        list(it)  # prefixes are filled while iterating
        return sorted(p[len(self.prefix) :].rstrip("/") for p in it.prefixes)


def make_store(url: str) -> Store:
    u = urlparse(url)
    if u.scheme == "gs":
        return GCSStore(u.netloc, u.path)
    if u.scheme == "file":
        return LocalStore(u.netloc + u.path)
    raise ValueError(f"unsupported STORE_URL: {url}")
