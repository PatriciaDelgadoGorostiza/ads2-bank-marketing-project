import os
from pathlib import Path

from src.config import MODELS_DIR
from src.pipeline import load_model_manifest


DEFAULT_LOCAL_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
MODEL_MANIFEST_URI_ENV = "MODEL_MANIFEST_URI"


def load_serving_manifest() -> dict:
    """Load the manifest that tells the API which model artifacts to use.

    Local teaching setup:
        No environment variable is set. The API reads `models/model_manifest.json`
        and uses model/preprocessor files mounted into the container.

    Professional deployment setup:
        `MODEL_MANIFEST_URI` points to a manifest in a model registry or object
        storage. That manifest would then point to the actual model and
        preprocessor artifacts.
    """
    manifest_uri = os.getenv(MODEL_MANIFEST_URI_ENV)

    if manifest_uri:
        return load_remote_serving_manifest(manifest_uri)

    return load_local_serving_manifest(DEFAULT_LOCAL_MANIFEST_PATH)


def load_local_serving_manifest(manifest_path: Path = DEFAULT_LOCAL_MANIFEST_PATH) -> dict:
    """Load a manifest from the local filesystem.

    This is the path used by Docker Compose in this project. The local `models/`
    directory is mounted into the API container as read-only volume.
    """
    return load_model_manifest(manifest_path)


def load_remote_serving_manifest(manifest_uri: str) -> dict:
    """Prepare the extension point for loading artifacts from registry/storage.

    A real implementation would typically:

    1. download the manifest from `manifest_uri`;
    2. read `model_uri` and `preprocessor_uri` from that manifest;
    3. download both artifacts into a local cache directory, for example
       `/tmp/models`;
    4. return the same structure as the local manifest loader, but with local
       `model_path` and `preprocessor_path` values.

    We intentionally do not implement a concrete provider here. Azure Blob
    Storage, S3, and MLflow Registry all need different clients, credentials,
    and URI conventions. Keeping this function explicit makes the deployment
    boundary visible without making the local demo depend on a cloud account.
    """
    raise NotImplementedError(
        f"Remote model artifact loading is not implemented yet. "
        f"Unset {MODEL_MANIFEST_URI_ENV} to use the local manifest, or implement "
        f"provider-specific loading for: {manifest_uri}"
    )
