"""
Thin helpers for reading/writing Azure Blob Storage using full blob URLs, e.g.

    https://<account>.blob.core.windows.net/ingestion-pipeline/source-documents/breakdown_policy_booklet.pdf
    \____________ account_url ____________/\___ container __/\____________ blob name ____________/

Authentication: BLOB_CONNECTION_STRING is used when it is set and points at the
same storage account as the URL; otherwise DefaultAzureCredential is used.
"""

from dataclasses import dataclass
from urllib.parse import urlparse, unquote
import json
import os
import threading

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient, ContentSettings

JSON_CONTENT_TYPE = "application/json; charset=utf-8"
MARKDOWN_CONTENT_TYPE = "text/markdown; charset=utf-8"

_credential: DefaultAzureCredential | None = None
_services: dict[str, BlobServiceClient] = {}
_lock = threading.Lock()


def _default_credential() -> DefaultAzureCredential:
    """One shared credential — token acquisition/caching is not free."""
    global _credential
    with _lock:
        if _credential is None:
            _credential = DefaultAzureCredential()
        return _credential


@dataclass(frozen=True)
class BlobLocation:
    """A blob URL split into its parts. Also used for prefixes ("directories")."""

    account_url: str   # https://<account>.blob.core.windows.net
    container: str     # ingestion-pipeline
    name: str          # source-documents/breakdown_policy_booklet.pdf

    @property
    def url(self) -> str:
        return f"{self.account_url}/{self.container}/{self.name}"

    @property
    def directory_url(self) -> str:
        """URL of this location treated as a prefix, with a trailing slash."""
        return f"{self.url.rstrip('/')}/"

    @property
    def account_name(self) -> str:
        return urlparse(self.account_url).netloc.split(".")[0]

    @property
    def filename(self) -> str:
        return self.name.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        return self.filename.rsplit(".", 1)[0]

    def sibling(self, container: str, prefix: str, filename: str) -> "BlobLocation":
        """Another blob in the same storage account."""
        name = f"{prefix.strip('/')}/{filename}" if prefix else filename
        return BlobLocation(self.account_url, container, name)

    def child(self, filename: str) -> "BlobLocation":
        """A blob underneath this location, treating it as a prefix."""
        return BlobLocation(self.account_url, self.container, f"{self.name.rstrip('/')}/{filename}")

    def __str__(self) -> str:
        return self.url


def account_url() -> str:
    """
    Storage account URL from configuration.

    The account name is deployment configuration, not source: it comes from
    BLOB_ACCOUNT_URL, or from the AccountName in BLOB_CONNECTION_STRING, or
    from STORAGE_ACCOUNT_NAME.
    """
    configured = os.getenv("BLOB_ACCOUNT_URL")
    if configured:
        return configured.rstrip("/")

    connection_string = os.getenv("BLOB_CONNECTION_STRING")
    if connection_string:
        settings = dict(
            pair.split("=", 1) for pair in connection_string.split(";") if "=" in pair
        )
        if settings.get("BlobEndpoint"):
            return settings["BlobEndpoint"].rstrip("/")
        if settings.get("AccountName"):
            suffix = settings.get("EndpointSuffix", "core.windows.net")
            protocol = settings.get("DefaultEndpointsProtocol", "https")
            return f"{protocol}://{settings['AccountName']}.blob.{suffix}"

    account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    if account_name:
        return f"https://{account_name}.blob.core.windows.net"

    raise ValueError(
        "No storage account configured. Set BLOB_ACCOUNT_URL, "
        "BLOB_CONNECTION_STRING or STORAGE_ACCOUNT_NAME in .env"
    )


def parse_blob_url(url: str) -> BlobLocation:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Not an Azure Blob Storage URL: {url!r}")

    path = unquote(parsed.path).lstrip("/")
    container, _, name = path.partition("/")
    if not container or not name:
        raise ValueError(
            f"Blob URL must contain a container and a blob name: {url!r}"
        )
    return BlobLocation(f"{parsed.scheme}://{parsed.netloc}", container, name)


# ---------------------------------------------------------------------------
# Clients — cached per account, since a pipeline run makes hundreds of calls
# ---------------------------------------------------------------------------

def service_client(account_url: str) -> BlobServiceClient:
    with _lock:
        service = _services.get(account_url)
        if service is not None:
            return service

        account_name = urlparse(account_url).netloc.split(".")[0]
        connection_string = os.getenv("BLOB_CONNECTION_STRING")
        service = None
        if connection_string:
            from_conn = BlobServiceClient.from_connection_string(connection_string)
            if from_conn.account_name == account_name:
                service = from_conn
        if service is None:
            service = BlobServiceClient(account_url, credential=_default_credential())

        _services[account_url] = service
        return service


def container_client(location: BlobLocation) -> ContainerClient:
    return service_client(location.account_url).get_container_client(location.container)


def blob_client(location: BlobLocation) -> BlobClient:
    return service_client(location.account_url).get_blob_client(
        location.container, location.name
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def download_blob(location: BlobLocation) -> bytes:
    return blob_client(location).download_blob().readall()


def download_json(location: BlobLocation) -> dict:
    return json.loads(download_blob(location).decode("utf-8"))


def blob_exists(location: BlobLocation) -> bool:
    try:
        return blob_client(location).exists()
    except ResourceNotFoundError:      # container itself is missing
        return False


def list_blobs(prefix: BlobLocation) -> list[BlobLocation]:
    """Every blob underneath `prefix`, sorted by name."""
    container = container_client(prefix)
    if not container.exists():
        return []
    name_prefix = f"{prefix.name.rstrip('/')}/"
    return sorted(
        (
            BlobLocation(prefix.account_url, prefix.container, blob.name)
            for blob in container.list_blobs(name_starts_with=name_prefix)
        ),
        key=lambda loc: loc.name,
    )


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def ensure_container(location: BlobLocation) -> None:
    container = container_client(location)
    if not container.exists():
        container.create_container()


def upload_text(location: BlobLocation, text: str, content_type: str) -> str:
    """Upload UTF-8 text, creating the container if needed. Returns the blob URL."""
    ensure_container(location)
    blob_client(location).upload_blob(
        data=text.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    return location.url


def upload_json(location: BlobLocation, payload: dict) -> str:
    """Upload a JSON document. Assumes the container already exists."""
    blob_client(location).upload_blob(
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type=JSON_CONTENT_TYPE),
    )
    return location.url


def delete_blobs(locations: list[BlobLocation]) -> int:
    for location in locations:
        blob_client(location).delete_blob()
    return len(locations)


def upload_bytes(location: BlobLocation, data: bytes, content_type: str) -> str:
    """Upload binary content (a source PDF, say), creating the container if needed."""
    ensure_container(location)
    blob_client(location).upload_blob(
        data=data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    return location.url
